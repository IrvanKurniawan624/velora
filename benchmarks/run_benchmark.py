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
                
                # Codegen uses ground_truth_test, others use ground_truth
                gt = data.get("ground_truth") or data.get("ground_truth_test")
                gt_map[prompt] = {
                    "category": category,
                    "ground_truth": gt
                }
    return gt_map

def grade_factual(answer, gt_str):
    try:
        gt_data = json.loads(gt_str)
        keywords = gt_data.get("keywords", [])
    except Exception:
        keywords = [gt_str]
        
    matched = [kw for kw in keywords if kw.lower() in answer.lower()]
    score = len(matched) / len(keywords) if keywords else 1.0
    return score, f"Matched {len(matched)}/{len(keywords)} keywords: {matched}"

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
        sentiment_match = data.get("sentiment", "").lower() == expected.get("sentiment", "").lower()
        has_reason = "reason" in data and len(data["reason"]) > 0
        score = 1.0 if (sentiment_match and has_reason) else 0.5 if sentiment_match else 0.0
        return score, f"Sentiment Match: {sentiment_match}, Has Reason: {has_reason}"
    except Exception as e:
        # Fallback if model answered as plain text rather than JSON
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

def main():
    print("==================================================")
    print("      VELORA HYBRID AGENT ACCURACY HARNESS       ")
    print("==================================================")
    
    # 1. Load ground truths
    gt_map = load_ground_truths()
    if not gt_map:
        print("Error: No datasets found. Run from velora directory.")
        sys.exit(1)
        
    # 2. Setup 19 evaluation tasks if present, otherwise fallback
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
        print("Please create tasks.json with your target evaluation prompts.")
        sys.exit(1)
        
    with open(INPUT_TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = json.load(f)
        
    total_tasks = len(tasks)
    print(f"Loaded {total_tasks} tasks to evaluate.")
    
    # 3. Execute the agent pipeline
    print("\nRunning agent pipeline...")
    start_time = time.time()
    
    # Run the main agent entrypoint
    python_exe = VELORA_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = "python"
        
    result = subprocess.run(
        [str(python_exe), "-m", "app.main"],
        cwd=str(VELORA_DIR),
        capture_output=True,
        text=True
    )
    
    elapsed_time = time.time() - start_time
    print(f"Agent finished in {elapsed_time:.2f} seconds.")
    
    if result.returncode != 0:
        print("Error: Agent exited with non-zero status code.")
        print(result.stderr)
        sys.exit(1)
        
    # 4. Read outputs and grade
    if not OUTPUT_RESULTS_FILE.exists():
        print(f"Error: output/results.json not found at {OUTPUT_RESULTS_FILE}.")
        sys.exit(1)
        
    with open(OUTPUT_RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    results_map = {r["task_id"]: r["answer"] for r in results}
    
    scores = []
    category_scores = {}
    
    print("\n---------------- Grading Results ----------------")
    for idx, task in enumerate(tasks, 1):
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        answer = results_map.get(task_id, "")
        gt_data = gt_map.get(prompt)
        
        if not gt_data:
            # Try fuzzy match on prompt in case of compression changes
            # We match by finding a key in gt_map that is a substring of the prompt or vice versa
            fuzzy_match = None
            for p, val in gt_map.items():
                # Strip spaces and punctuation for comparison
                p_clean = re.sub(r"\W+", "", p).lower()
                prompt_clean = re.sub(r"\W+", "", prompt).lower()
                if p_clean in prompt_clean or prompt_clean in p_clean:
                    fuzzy_match = val
                    break
            gt_data = fuzzy_match
            
        if not gt_data:
            print(f"[-] Task {task_id}: Skip (No ground truth found for prompt)")
            continue
            
        category = gt_data["category"]
        gt_str = gt_data["ground_truth"]
        
        score, reason = grade_task(category, answer, gt_str)
        scores.append(score)
        
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
        
        status_char = "PASS" if score == 1.0 else "WARN" if score > 0.0 else "FAIL"
        print(f"[{status_char:<4}] Task {task_id} ({category}): Score={score:.2f} | {reason}")
        
    # 5. Output Summary Scorecard
    print("\n=================== SCORECARD ===================")
    avg_overall_score = sum(scores) / len(scores) if scores else 0.0
    
    print("Category Breakdown:")
    for cat, cat_list in category_scores.items():
        cat_avg = sum(cat_list) / len(cat_list)
        print(f"  - {cat.capitalize():<15}: {cat_avg*100:5.1f}% ({sum(1 for s in cat_list if s == 1.0)}/{len(cat_list)} correct)")
        
    print(f"\nOverall Accuracy Gate Check:")
    correct_count = sum(1 for s in scores if s >= 0.8) # We count a task as correct if score is >= 0.8
    accuracy_percentage = (correct_count / total_tasks) * 100 if total_tasks else 0.0
    
    # We display as n/19 style if total tasks is 19
    formatted_ratio = f"{correct_count}/{total_tasks}"
    
    print(f"  - Correct Tasks  : {formatted_ratio} ({accuracy_percentage:.1f}%)")
    
    # Check 80% accuracy gate
    if accuracy_percentage >= 80.0:
        print("  - Leaderboard Gate: PASSED (>= 80% Accuracy)")
    else:
        print("  - Leaderboard Gate: FAILED (< 80% Accuracy)")
        
    print(f"  - Execution Time : {elapsed_time:.2f} seconds")
    print("==================================================")

if __name__ == "__main__":
    main()
