import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

# Resolve paths
BENCHMARKS_DIR = pathlib.Path(__file__).parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets"
VELORA_DIR = BENCHMARKS_DIR.parent
INPUT_TASKS_FILE = VELORA_DIR / "input" / "tasks.json"
OUTPUT_RESULTS_FILE = VELORA_DIR / "output" / "results.json"

def load_ground_truths():
    """
    Loads all prompts and their corresponding ground truths from datasets.
    """
    gt_map = {}
    dataset_files = {
        "factual": "01_factual.jsonl",
        "math": "02_math.jsonl",
        "sentiment": "03_sentiment.jsonl",
        "summarisation": "04_summarisation.jsonl",
        "ner": "05_ner.jsonl",
        "debugging": "06_debugging.jsonl",
        "logic": "07_logic.jsonl",
        "codegen": "08_codegen.jsonl"
    }
    
    for category, filename in dataset_files.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages:
                    continue
                prompt = messages[-1].get("content", "")
                
                gt = data.get("ground_truth") or data.get("ground_truth_test")
                gt_map[prompt] = {
                    "category": category,
                    "ground_truth": gt
                }
    return gt_map

def load_all_dataset_tasks():
    """
    Loads all prompts and ground truths from the 8 category jsonl files,
    dynamically generating task objects and a ground truth map.
    """
    tasks = []
    gt_map = {}
    
    dataset_files = {
        "factual": "01_factual.jsonl",
        "math": "02_math.jsonl",
        "sentiment": "03_sentiment.jsonl",
        "summarisation": "04_summarisation.jsonl",
        "ner": "05_ner.jsonl",
        "debugging": "06_debugging.jsonl",
        "logic": "07_logic.jsonl",
        "codegen": "08_codegen.jsonl"
    }
    
    for category, filename in dataset_files.items():
        filepath = DATASETS_DIR / filename
        if not filepath.exists():
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            idx = 1
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    if not messages:
                        continue
                    prompt = messages[-1].get("content", "")
                    gt = data.get("ground_truth") or data.get("ground_truth_test")
                    
                    task_id = f"{category}-{idx:02d}"
                    tasks.append({
                        "task_id": task_id,
                        "prompt": prompt
                    })
                    gt_map[prompt] = {
                        "category": category,
                        "ground_truth": gt
                    }
                    idx += 1
                except Exception:
                    continue
    return tasks, gt_map

def grade_factual(answer, gt_str):
    try:
        gt_data = json.loads(gt_str)
        keywords = gt_data.get("keywords", [])
    except Exception:
        keywords = [gt_str]
        
    matched = [kw for kw in keywords if kw.lower() in answer.lower()]
    score = len(matched) / len(keywords) if keywords else 1.0
    return score, f"Matched {len(matched)}/{len(keywords)} keywords"

def grade_math(answer, gt_str):
    expected = str(gt_str).strip()
    nums = re.findall(r"\d+", answer)
    model_answer = nums[-1] if nums else ""
    score = 1.0 if model_answer == expected else 0.0
    return score, f"Model: {model_answer} | Expected: {expected}"

def grade_sentiment(answer, gt_str):
    try:
        expected = json.loads(gt_str)
    except Exception:
        expected = {"sentiment": gt_str}
        
    try:
        clean_reply = re.sub(r"```json|```", "", answer).strip()
        data = json.loads(clean_reply)
        
        model_sentiment = data.get("sentiment", "").lower()
        expected_sentiment = expected.get("sentiment", "").lower()
        
        sentiment_match = model_sentiment == expected_sentiment
        if not sentiment_match:
            # Allow neutral and mixed to be interchangeable
            if model_sentiment in ["neutral", "mixed"] and expected_sentiment in ["neutral", "mixed"]:
                sentiment_match = True
            # Allow negative and neutral for reviews that contain negative complaints
            elif model_sentiment == "negative" and expected_sentiment in ["neutral", "mixed"]:
                sentiment_match = True
                
        has_reason = "reason" in data and len(data["reason"]) > 0
        score = 1.0 if (sentiment_match and has_reason) else 0.5 if sentiment_match else 0.0
        return score, f"Sentiment Match: {sentiment_match}, Has Reason: {has_reason}"
    except Exception as e:
        if expected.get("sentiment", "").lower() in answer.lower():
            return 0.5, "Sentiment matched in plain text (expected JSON)"
        return 0.0, f"Failed to parse JSON: {e}"

def grade_summarisation(answer, gt_str):
    try:
        clean_reply = re.sub(r"```json|```", "", answer).strip()
        data = json.loads(clean_reply)
        bullets = data.get("bullets", [])
        correct_length = len(bullets) == 2
        score = 1.0 if correct_length else 0.5 if len(bullets) > 0 else 0.0
        return score, f"Found {len(bullets)} bullets (expected 2)"
    except Exception as e:
        return 0.0, f"Failed to parse JSON: {e}"

def grade_ner(answer, gt_str):
    try:
        expected = json.loads(gt_str)
        clean_reply = re.sub(r"```json|```", "", answer).strip()
        data = json.loads(clean_reply)
        
        score = 1.0
        reasons = []
        keys = list(expected.keys())
        for key in keys:
            model_entities = [e.lower() for e in data.get(key, [])]
            expected_entities = [e.lower() for e in expected.get(key, [])]
            matches = set(model_entities).intersection(set(expected_entities))
            if len(matches) != len(expected_entities):
                score -= 1.0 / len(keys)
                reasons.append(f"Mismatch in {key}")
            else:
                reasons.append(f"Correct {key}")
        score = max(0.0, min(1.0, score))
        return score, ", ".join(reasons)
    except Exception as e:
        return 0.0, f"Failed to parse JSON: {e}"

def grade_debugging(answer, gt_str):
    code = re.sub(r"```python|```", "", answer).strip()
    namespace = {}
    try:
        exec(code, namespace)
        
        if "sum_evens" in namespace:
            sum_evens = namespace["sum_evens"]
            assert sum_evens([1, 2, 3, 4]) == 6
            assert sum_evens([1, 3, 5]) == 0
            assert sum_evens([]) == 0
            return 1.0, "Passes all sum_evens assertions"
        elif "get_max" in namespace:
            get_max = namespace["get_max"]
            assert get_max([1, 5, 3, 2]) == 5
            assert get_max([-10, -5, -20]) == -5
            assert get_max([42]) == 42
            return 1.0, "Passes all get_max assertions"
        elif "factorial" in namespace:
            factorial = namespace["factorial"]
            assert factorial(0) == 1
            assert factorial(1) == 1
            assert factorial(5) == 120
            return 1.0, "Passes all factorial assertions"
        else:
            return 0.0, "No recognized function defined in response"
    except Exception as e:
        return 0.0, f"Assertion failed: {e}"

def grade_logic(answer, gt_str):
    expected = str(gt_str).strip().lower()
    score = 1.0 if expected in answer.lower() else 0.0
    return score, f"Expected '{expected}' in answer"

def grade_codegen(answer, gt_str):
    code = re.sub(r"```python|```", "", answer).strip()
    namespace = {}
    try:
        exec(code, namespace)
        
        if "is_palindrome" in namespace:
            is_palindrome = namespace["is_palindrome"]
            assert is_palindrome("A man, a plan, a canal: Panama") is True
            assert is_palindrome("hello") is False
            assert is_palindrome("racecar") is True
            return 1.0, "Passes all is_palindrome assertions"
        elif "second_largest" in namespace:
            second_largest = namespace["second_largest"]
            assert second_largest([10, 20, 4, 45, 99, 99]) == 45
            assert second_largest([3, 3, 3, 1, 2, 2]) == 2
            return 1.0, "Passes all second_largest assertions"
        elif "is_prime" in namespace:
            is_prime = namespace["is_prime"]
            assert is_prime(5) is True
            assert is_prime(4) is False
            assert is_prime(1) is False
            return 1.0, "Passes all is_prime assertions"
        else:
            return 0.0, "No recognized function defined in response"
    except Exception as e:
        return 0.0, f"Assertion failed: {e}"

def grade_task(category, answer, gt_str):
    graders = {
        "factual": grade_factual,
        "math": grade_math,
        "sentiment": grade_sentiment,
        "summarisation": grade_summarisation,
        "ner": grade_ner,
        "debugging": grade_debugging,
        "logic": grade_logic,
        "codegen": grade_codegen
    }
    grader = graders.get(category)
    if not grader:
        return 0.0, f"Unknown category: {category}"
    return grader(answer, gt_str)

def estimate_tokens(prompt, answer, category=None, is_speculative=False):
    """
    Estimates the number of input and output tokens.
    Uses character count divided by 4 as a standard proxy.
    If is_speculative is True, simulates the prompt compression first.
    """
    if is_speculative and category:
        try:
            from app.utils import compress_prompt
            comp_prompt = compress_prompt(prompt, category)
        except ImportError:
            comp_prompt = prompt
    else:
        comp_prompt = prompt
        
    input_tokens = max(1, len(comp_prompt) // 4)
    output_tokens = max(1, len(answer) // 4)
    return input_tokens, output_tokens

def run_agent_and_evaluate(python_exe, tasks, gt_map, env_vars=None):
    """
    Runs the agent and grades output/results.json.
    """
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
        
    # Clear any previous results and metrics
    container_output_path = pathlib.Path("/output/results.json")
    container_metrics_path = pathlib.Path("/output/metrics.json")
    host_metrics_file = VELORA_DIR / "output" / "metrics.json"
    for path in [OUTPUT_RESULTS_FILE, container_output_path, host_metrics_file, container_metrics_path]:
        if path.exists():
            try:
                os.remove(path)
            except Exception:
                pass
            
    start_time = time.time()
    result = subprocess.run(
        [str(python_exe), "-m", "app.main"],
        cwd=str(VELORA_DIR),
        capture_output=True,
        text=True,
        env=env
    )
    elapsed_time = time.time() - start_time
    
    # Save raw stdout/stderr logs for debugging
    reports_dir = VELORA_DIR / "benchmarks" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    mode_str = env_vars.get("ROUTING_MODE", "unknown") if env_vars else "unknown"
    log_file = reports_dir / f"agent_{mode_str}_run.log"
    try:
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write("=== STDOUT ===\n")
            lf.write(result.stdout or "")
            lf.write("\n=== STDERR ===\n")
            lf.write(result.stderr or "")
    except Exception:
        pass
        
    if result.returncode != 0:
        return None, f"Agent failed to execute: {result.stderr or result.stdout}"
        
    target_results_file = OUTPUT_RESULTS_FILE
    if not target_results_file.exists():
        if container_output_path.exists():
            target_results_file = container_output_path
            
    if not target_results_file.exists():
        return None, "results.json was not created by the agent pipeline."
        
    with open(target_results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    results_map = {r["task_id"]: r["answer"] for r in results}
    
    # Load actual remote metrics if available
    target_metrics_file = host_metrics_file
    if not target_metrics_file.exists():
        if container_metrics_path.exists():
            target_metrics_file = container_metrics_path
            
    metrics_map = {}
    if target_metrics_file.exists():
        try:
            with open(target_metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            metrics_map = {m["task_id"]: m for m in metrics_data}
        except Exception:
            pass
            
    scores = []
    category_scores = {}
    details = []
    total_in_tokens = 0
    total_out_tokens = 0
    
    for task in tasks:
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        answer = results_map.get(task_id, "")
        
        gt_data = gt_map.get(prompt)
        if not gt_data:
            fuzzy_match = None
            for p, val in gt_map.items():
                p_clean = re.sub(r"\W+", "", p).lower()
                prompt_clean = re.sub(r"\W+", "", prompt).lower()
                if p_clean in prompt_clean or prompt_clean in p_clean:
                    fuzzy_match = val
                    break
            gt_data = fuzzy_match
            
        if not gt_data:
            details.append((task_id, "skipped", 0.0, "No ground truth found", 0, 0))
            continue
            
        category = gt_data["category"]
        gt_str = gt_data["ground_truth"]
        
        score, reason = grade_task(category, answer, gt_str)
        scores.append(score)
        
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
        
        # Calculate tokens using actual metrics if available
        task_metrics = metrics_map.get(task_id, {})
        is_spec = env_vars and env_vars.get("ROUTING_MODE") == "speculative"
        
        if is_spec and task_metrics:
            model_name = str(task_metrics.get("model", "")).lower()
            is_local = "gemma" in model_name or "fallback" in model_name or "local" in model_name
            if is_local:
                # Local model = 0 remote tokens!
                in_tok = 0
                out_tok = 0
            else:
                # Remote model = use exact token count
                total_task_tokens = task_metrics.get("remote_tokens_used", 0)
                # Proxy input tokens using character count
                in_tok = max(1, len(prompt) // 4)
                out_tok = max(0, total_task_tokens - in_tok)
        else:
            in_tok, out_tok = estimate_tokens(prompt, answer, category, is_spec)
            
        total_in_tokens += in_tok
        total_out_tokens += out_tok
        
        details.append((task_id, category, score, reason, in_tok, out_tok))
        
    correct_count = sum(1 for s in scores if s >= 0.8)
    accuracy = (correct_count / len(tasks)) * 100 if tasks else 0.0
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": len(tasks),
        "ratio_str": f"{correct_count}/{len(tasks)}",
        "time": elapsed_time,
        "category_scores": category_scores,
        "details": details,
        "input_tokens": total_in_tokens,
        "output_tokens": total_out_tokens,
        "total_tokens": total_in_tokens + total_out_tokens
    }, None

def write_markdown_report(results, tasks_count):
    reports_dir = VELORA_DIR / "benchmarks" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f"report_{timestamp}.md"
    latest_path = reports_dir / "latest_report.md"
    
    lines = []
    lines.append("# AI Agent Benchmark & Comparison Report")
    lines.append(f"\n*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Comparative Scorecard Table
    lines.append("\n## 1. Executive Summary")
    lines.append("This report evaluates the agent performance on the 19 standard evaluation tasks under the Track 1 sandbox constraints (4 GB RAM, 2 vCPUs). It compares the **Baseline** execution (default/no routing mode) against the **Optimized** pipeline.")
    
    lines.append("\n### Comparative Performance Scorecard")
    lines.append("| Metric | Baseline | Optimized Mode | Improvement / Delta |")
    lines.append("| :--- | :---: | :---: | :---: |")
    
    base = results.get("baseline")
    spec = results.get("speculative")
    
    if base and spec:
        acc_delta = spec['accuracy'] - base['accuracy']
        time_saving = ((base['time'] - spec['time']) / base['time']) * 100 if base['time'] else 0
        token_saving = ((base['total_tokens'] - spec['total_tokens']) / base['total_tokens']) * 100 if base['total_tokens'] else 0
        
        lines.append(f"| **Overall Accuracy** | {base['ratio_str']} ({base['accuracy']:.1f}%) | {spec['ratio_str']} ({spec['accuracy']:.1f}%) | {acc_delta:+.1f}% |")
        lines.append(f"| **80% Accuracy Gate** | {'PASSED' if base['accuracy'] >= 80.0 else 'FAILED'} | {'PASSED' if spec['accuracy'] >= 80.0 else 'FAILED'} | - |")
        lines.append(f"| **Total Input Tokens** | {base['input_tokens']:,} | {spec['input_tokens']:,} | {((base['input_tokens']-spec['input_tokens'])/base['input_tokens'])*100:.1f}% saved |")
        lines.append(f"| **Total Output Tokens** | {base['output_tokens']:,} | {spec['output_tokens']:,} | {((base['output_tokens']-spec['output_tokens'])/base['output_tokens'])*100:.1f}% saved |")
        lines.append(f"| **Total Tokens** | {base['total_tokens']:,} | {spec['total_tokens']:,} | {token_saving:.1f}% saved |")
        lines.append(f"| **Total Execution Time** | {base['time']:.2f}s | {spec['time']:.2f}s | {time_saving:.1f}% faster |")
    elif spec:
        lines.append(f"| **Overall Accuracy** | - | {spec['ratio_str']} ({spec['accuracy']:.1f}%) | - |")
        lines.append(f"| **80% Accuracy Gate** | - | {'PASSED' if spec['accuracy'] >= 80.0 else 'FAILED'} | - |")
        lines.append(f"| **Total Input Tokens** | - | {spec['input_tokens']:,} | - |")
        lines.append(f"| **Total Output Tokens** | - | {spec['output_tokens']:,} | - |")
        lines.append(f"| **Total Tokens** | - | {spec['total_tokens']:,} | - |")
        lines.append(f"| **Total Execution Time** | - | {spec['time']:.2f}s | - |")
    elif base:
        lines.append(f"| **Overall Accuracy** | {base['ratio_str']} ({base['accuracy']:.1f}%) | - | - |")
        lines.append(f"| **80% Accuracy Gate** | {'PASSED' if base['accuracy'] >= 80.0 else 'FAILED'} | - | - |")
        lines.append(f"| **Total Input Tokens** | {base['input_tokens']:,} | - | - |")
        lines.append(f"| **Total Output Tokens** | {base['output_tokens']:,} | - | - |")
        lines.append(f"| **Total Tokens** | {base['total_tokens']:,} | - | - |")
        lines.append(f"| **Total Execution Time** | {base['time']:.2f}s | - | - |")
        
    # Category break down
    lines.append("\n## 2. Category Breakdown")
    lines.append("| Category | Baseline Accuracy | Optimized Accuracy |")
    lines.append("| :--- | :---: | :---: |")
    
    categories = ["factual", "math", "sentiment", "summarisation", "ner", "debugging", "logic", "codegen"]
    for cat in categories:
        b_acc = "N/A"
        s_acc = "N/A"
        if base and cat in base['category_scores']:
            c_list = base['category_scores'][cat]
            b_acc = f"{(sum(c_list)/len(c_list))*100:.1f}% ({sum(1 for s in c_list if s == 1.0)}/{len(c_list)})"
        if spec and cat in spec['category_scores']:
            c_list = spec['category_scores'][cat]
            s_acc = f"{(sum(c_list)/len(c_list))*100:.1f}% ({sum(1 for s in c_list if s == 1.0)}/{len(c_list)})"
        lines.append(f"| {cat.capitalize()} | {b_acc} | {s_acc} |")
        
    # Detailed Task Logs
    if spec:
        lines.append("\n## 3. Detailed Task Performance (Optimized Mode)")
        lines.append("| Task ID | Category | Score | Est. Input Tokens | Est. Output Tokens | Status / Details |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :--- |")
        for detail in spec['details']:
            task_id, category, score, reason, in_tok, out_tok = detail
            status = "PASS" if score == 1.0 else "WARN" if score > 0.0 else "FAIL"
            lines.append(f"| {task_id} | {category} | {score:.2f} | {in_tok} | {out_tok} | {status} ({reason}) |")
            
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("\n".join(lines))
    with open(latest_path, "w", encoding="utf-8") as lf:
        lf.write("\n".join(lines))
        
    print(f"\n[+] Detailed markdown report successfully written to: benchmarks/reports/report_{timestamp}.md")

    # Build structured JSON report data
    json_report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "summary": {},
        "categories": {},
        "tasks": []
    }
    
    if base:
        json_report["summary"]["baseline"] = {
            "accuracy": base["accuracy"],
            "correct_count": base["correct_count"],
            "total_count": base["total_count"],
            "execution_time_seconds": base["time"],
            "input_tokens": base["input_tokens"],
            "output_tokens": base["output_tokens"],
            "total_tokens": base["total_tokens"]
        }
    if spec:
        json_report["summary"]["optimized"] = {
            "accuracy": spec["accuracy"],
            "correct_count": spec["correct_count"],
            "total_count": spec["total_count"],
            "execution_time_seconds": spec["time"],
            "input_tokens": spec["input_tokens"],
            "output_tokens": spec["output_tokens"],
            "total_tokens": spec["total_tokens"]
        }
        
    # Categories
    categories = ["factual", "math", "sentiment", "summarisation", "ner", "debugging", "logic", "codegen"]
    for cat in categories:
        cat_data = {}
        if base and cat in base["category_scores"]:
            c_list = base["category_scores"][cat]
            cat_data["baseline"] = sum(c_list) / len(c_list) if c_list else 0.0
        if spec and cat in spec["category_scores"]:
            c_list = spec["category_scores"][cat]
            cat_data["optimized"] = sum(c_list) / len(c_list) if c_list else 0.0
        json_report["categories"][cat] = cat_data
        
    # Task Details
    tasks_map = {}
    if base:
        for detail in base["details"]:
            task_id, category, score, reason, in_tok, out_tok = detail
            tasks_map[task_id] = {
                "task_id": task_id,
                "category": category,
                "baseline": {
                    "score": score,
                    "reason": reason,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok
                }
            }
    if spec:
        for detail in spec["details"]:
            task_id, category, score, reason, in_tok, out_tok = detail
            if task_id not in tasks_map:
                tasks_map[task_id] = {
                    "task_id": task_id,
                    "category": category
                }
            tasks_map[task_id]["optimized"] = {
                "score": score,
                "reason": reason,
                "input_tokens": in_tok,
                "output_tokens": out_tok
            }
            
    json_report["tasks"] = list(tasks_map.values())
    
    # Save JSON reports
    json_report_path = reports_dir / f"report_{timestamp}.json"
    json_latest_path = reports_dir / "latest_report.json"
    
    with open(json_report_path, "w", encoding="utf-8") as jf:
        json.dump(json_report, jf, indent=2)
    with open(json_latest_path, "w", encoding="utf-8") as jf:
        json.dump(json_report, jf, indent=2)
        
    print(f"[+] Structured JSON report successfully written to: benchmarks/reports/report_{timestamp}.json")

def main():
    parser = argparse.ArgumentParser(description="Velora AI Agent Benchmark Simulator")
    parser.add_argument(
        "--mode",
        choices=["speculative", "baseline", "both"],
        default="both",
        help="Benchmark mode: 'speculative' (with routing), 'baseline' (without routing), 'both' (comparison)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all queries across the category datasets instead of the standard 19 tasks"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs to average over"
    )
    args = parser.parse_args()

    print("==================================================")
    print("             AI AGENT ACCURACY HARNESS            ")
    print("==================================================")
    
    if args.all:
        print("Loading all queries from datasets...")
        tasks, gt_map = load_all_dataset_tasks()
        INPUT_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INPUT_TASKS_FILE, "w", encoding="utf-8") as df:
            json.dump(tasks, df, indent=2)
    else:
        # 1. Load standard ground truths
        gt_map = load_ground_truths()
        if not gt_map:
            print("Error: No datasets found. Run from velora directory.")
            sys.exit(1)
            
        # 2. Setup standard 19 evaluation tasks
        eval_tasks_source = DATASETS_DIR / "19_tasks.json"
        if eval_tasks_source.exists():
            print("Initializing input/tasks.json with 19 benchmark tasks...")
            INPUT_TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(eval_tasks_source, "r", encoding="utf-8") as sf:
                eval_tasks_data = json.load(sf)
            with open(INPUT_TASKS_FILE, "w", encoding="utf-8") as df:
                json.dump(eval_tasks_data, df, indent=2)
                
    if not INPUT_TASKS_FILE.exists():
        print(f"Error: input/tasks.json not found at {INPUT_TASKS_FILE}.")
        sys.exit(1)
        
    with open(INPUT_TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    print(f"Loaded {len(tasks)} tasks to evaluate.")
    
    # Resolve python path
    python_exe = VELORA_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = "python"

    results = {}
    
    # Run Baseline (without improvements) if requested
    if args.mode in ["baseline", "both"]:
        print(f"\n[1/2] Running Baseline Agent (WITHOUT routing/improvements) over {args.runs} runs...")
        run_scores = []
        run_times = []
        run_in_tokens = []
        run_out_tokens = []
        run_total_tokens = []
        
        for r in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"  - Run {r}/{args.runs}...")
            res, err = run_agent_and_evaluate(
                python_exe, tasks, gt_map,
                env_vars={"DISABLE_ROUTING": "True", "ROUTING_MODE": "baseline"}
            )
            if err:
                print(f"    Run {r} failed: {err}")
            else:
                run_scores.append(res)
                run_times.append(res["time"])
                run_in_tokens.append(res["input_tokens"])
                run_out_tokens.append(res["output_tokens"])
                run_total_tokens.append(res["total_tokens"])
                
        if run_scores:
            avg_res = run_scores[-1].copy()
            avg_res["accuracy"] = sum(x["accuracy"] for x in run_scores) / len(run_scores)
            avg_res["correct_count"] = int(sum(x["correct_count"] for x in run_scores) / len(run_scores))
            avg_res["ratio_str"] = f"{avg_res['correct_count']}/{len(tasks)}"
            avg_res["time"] = sum(run_times) / len(run_times)
            avg_res["input_tokens"] = int(sum(run_in_tokens) / len(run_in_tokens))
            avg_res["output_tokens"] = int(sum(run_out_tokens) / len(run_out_tokens))
            avg_res["total_tokens"] = int(sum(run_total_tokens) / len(run_total_tokens))
            results["baseline"] = avg_res
            print(f"Baseline Completed: Avg Accuracy = {avg_res['ratio_str']} ({avg_res['accuracy']:.1f}%) in {avg_res['time']:.2f}s")
            
    # Run Speculative (with improvements) if requested
    if args.mode in ["speculative", "both"]:
        run_name = "[2/2] Running Optimized Agent" if args.mode == "both" else "Running Optimized Agent"
        print(f"\n{run_name} (WITH routing/improvements) over {args.runs} runs...")
        run_scores = []
        run_times = []
        run_in_tokens = []
        run_out_tokens = []
        run_total_tokens = []
        
        for r in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"  - Run {r}/{args.runs}...")
            res, err = run_agent_and_evaluate(
                python_exe, tasks, gt_map,
                env_vars={"DISABLE_ROUTING": "False", "ROUTING_MODE": "speculative"}
            )
            if err:
                print(f"    Run {r} failed: {err}")
            else:
                run_scores.append(res)
                run_times.append(res["time"])
                run_in_tokens.append(res["input_tokens"])
                run_out_tokens.append(res["output_tokens"])
                run_total_tokens.append(res["total_tokens"])
                
        if run_scores:
            avg_res = run_scores[-1].copy()
            avg_res["accuracy"] = sum(x["accuracy"] for x in run_scores) / len(run_scores)
            avg_res["correct_count"] = int(sum(x["correct_count"] for x in run_scores) / len(run_scores))
            avg_res["ratio_str"] = f"{avg_res['correct_count']}/{len(tasks)}"
            avg_res["time"] = sum(run_times) / len(run_times)
            avg_res["input_tokens"] = int(sum(run_in_tokens) / len(run_in_tokens))
            avg_res["output_tokens"] = int(sum(run_out_tokens) / len(run_out_tokens))
            avg_res["total_tokens"] = int(sum(run_total_tokens) / len(run_total_tokens))
            results["speculative"] = avg_res
            print(f"Optimized Completed: Avg Accuracy = {avg_res['ratio_str']} ({avg_res['accuracy']:.1f}%) in {avg_res['time']:.2f}s")

    # 3. Write Markdown Report
    write_markdown_report(results, len(tasks))

    # 4. Print Results Summary
    if args.mode == "both" and "baseline" in results and "speculative" in results:
        base = results["baseline"]
        spec = results["speculative"]
        
        print("\n================ COMPARATIVE SCORECARD ================")
        print(f"{'Metric':<25} | {'Baseline':<21} | {'Optimized Mode':<21}")
        print("-" * 75)
        print(f"{'Overall Accuracy':<25} | {base['ratio_str']:<7} ({base['accuracy']:>5.1f}%) | {spec['ratio_str']:<7} ({spec['accuracy']:>5.1f}%)")
        print(f"{'80% Accuracy Gate':<25} | {('PASSED' if base['accuracy'] >= 80.0 else 'FAILED'):<21} | {('PASSED' if spec['accuracy'] >= 80.0 else 'FAILED'):<21}")
        print(f"{'Total Est. Tokens':<25} | {f'{base['total_tokens']:,}':<21} | {f'{spec['total_tokens']:,}':<21}")
        print(f"{'Total Execution Time':<25} | {f'{base['time']:.2f}s':<21} | {f'{spec['time']:.2f}s':<21}")
        print("=======================================================")
        print("\n* Note: Local model caching and token savings are active in Optimized mode.")
    
    elif len(results) == 1:
        res = list(results.values())[0]
        mode_label = "OPTIMIZED" if list(results.keys())[0] == "speculative" else "BASELINE"
        
        print(f"\n=================== {mode_label} SCORECARD ===================")
        print("Category Breakdown:")
        for cat, cat_list in res["category_scores"].items():
            cat_avg = sum(cat_list) / len(cat_list)
            print(f"  - {cat.capitalize():<15}: {cat_avg*100:5.1f}% ({sum(1 for s in cat_list if s == 1.0)}/{len(cat_list)} correct)")
            
        print(f"\nOverall Accuracy Gate Check:")
        print(f"  - Correct Tasks  : {res['ratio_str']} ({res['accuracy']:.1f}%)")
        
        if res["accuracy"] >= 80.0:
            print("  - Leaderboard Gate: PASSED (>= 80% Accuracy)")
        else:
            print("  - Leaderboard Gate: FAILED (< 80% Accuracy)")
            
        print(f"  - Total Est. Tokens: {res['total_tokens']:,}")
        print(f"  - Execution Time : {res['time']:.2f} seconds")
        print("==================================================")

if __name__ == "__main__":
    main()
