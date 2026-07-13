"""Local benchmark runner for Velora's new app/ engine.

Reads the 8 category jsonl datasets from benchmarks/datasets, runs the
container on the combined 19 tasks in either MODE=zero or MODE=hybrid, scores
answers against the jsonl ground truth, and prints a markdown report.

Requires Docker. The image is built separately (see README). For hybrid mode,
provide a .env file with FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

BENCHMARKS_DIR = pathlib.Path(__file__).parent
DATASETS_DIR = BENCHMARKS_DIR / "datasets"
REPORTS_DIR = BENCHMARKS_DIR / "reports"

DATASET_FILES = {
    "factual": "01_factual.jsonl",
    "math": "02_math.jsonl",
    "sentiment": "03_sentiment.jsonl",
    "summarisation": "04_summarisation.jsonl",
    "ner": "05_ner.jsonl",
    "debugging": "06_debugging.jsonl",
    "logic": "07_logic.jsonl",
    "codegen": "08_codegen.jsonl",
}


def load_tasks():
    tasks = []
    gt_by_id = {}
    for category, filename in DATASET_FILES.items():
        path = DATASETS_DIR / filename
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                messages = data.get("messages", [])
                if not messages:
                    continue
                prompt = messages[-1].get("content", "")
                gt = data.get("ground_truth") or data.get("ground_truth_test")
                if not prompt or not gt:
                    continue
                task_id = f"{category}-{idx:02d}"
                tasks.append({"task_id": task_id, "prompt": prompt})
                gt_by_id[task_id] = {"category": category, "ground_truth": gt}
    return tasks, gt_by_id


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
        clean = re.sub(r"```json|```", "", answer).strip()
        data = json.loads(clean)
        model_sentiment = data.get("sentiment", "").lower()
        expected_sentiment = expected.get("sentiment", "").lower()
        sentiment_match = model_sentiment == expected_sentiment
        if not sentiment_match:
            if model_sentiment in ("neutral", "mixed") and expected_sentiment in ("neutral", "mixed"):
                sentiment_match = True
            elif model_sentiment == "negative" and expected_sentiment in ("neutral", "mixed"):
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
        clean = re.sub(r"```json|```", "", answer).strip()
        data = json.loads(clean)
        bullets = data.get("bullets", [])
        score = 1.0 if len(bullets) == 2 else 0.5 if len(bullets) > 0 else 0.0
        return score, f"Found {len(bullets)} bullets (expected 2)"
    except Exception as e:
        return 0.0, f"Failed to parse JSON: {e}"


def grade_ner(answer, gt_str):
    try:
        expected = json.loads(gt_str)
        clean = re.sub(r"```json|```", "", answer).strip()
        data = json.loads(clean)
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


def grade_code(answer, gt_str):
    code = re.sub(r"```python|```", "", answer).strip()
    namespace = {}
    try:
        exec(code, namespace)
        for fn_name in ("sum_evens", "get_max", "factorial", "is_palindrome", "second_largest", "is_prime"):
            if fn_name in namespace:
                fn = namespace[fn_name]
                if fn_name == "sum_evens":
                    assert fn([1, 2, 3, 4]) == 6 and fn([1, 3, 5]) == 0 and fn([]) == 0
                elif fn_name == "get_max":
                    assert fn([1, 5, 3, 2]) == 5 and fn([-10, -5, -20]) == -5 and fn([42]) == 42
                elif fn_name == "factorial":
                    assert fn(0) == 1 and fn(1) == 1 and fn(5) == 120
                elif fn_name == "is_palindrome":
                    assert fn("A man, a plan, a canal: Panama") is True and fn("hello") is False and fn("racecar") is True
                elif fn_name == "second_largest":
                    assert fn([10, 20, 4, 45, 99, 99]) == 45 and fn([3, 3, 3, 1, 2, 2]) == 2
                elif fn_name == "is_prime":
                    assert fn(5) is True and fn(4) is False and fn(1) is False
                return 1.0, f"Passes all {fn_name} assertions"
        return 0.0, "No recognized function defined in response"
    except Exception as e:
        return 0.0, f"Assertion failed: {e}"


def grade_logic(answer, gt_str):
    expected = str(gt_str).strip().lower()
    score = 1.0 if expected in answer.lower() else 0.0
    return score, f"Expected '{expected}' in answer"


GRADERS = {
    "factual": grade_factual,
    "math": grade_math,
    "sentiment": grade_sentiment,
    "summarisation": grade_summarisation,
    "ner": grade_ner,
    "debugging": grade_code,
    "logic": grade_logic,
    "codegen": grade_code,
}


def grade_task(category, answer, gt_str):
    grader = GRADERS.get(category)
    if not grader:
        return 0.0, f"Unknown category: {category}"
    return grader(answer, gt_str)


def estimate_tokens(prompt, answer):
    return max(1, len(prompt) // 4), max(1, len(answer) // 4)


def parse_remote_tokens(stderr: str) -> int:
    total = 0
    for line in stderr.splitlines():
        m = re.search(r"running_total=(\d+)", line)
        if m:
            total = max(total, int(m.group(1)))
    return total


def run_docker(tasks, mode, image, env_file):
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="velora_benchmark_"))
    input_file = tmpdir / "tasks.json"
    output_file = tmpdir / "results.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)

    cmd = [
        "docker", "run", "--rm",
        "--cpus=2", "--memory=4g",
        "-v", f"{tmpdir}:/io",
        "-e", f"MODE={mode}",
        "-e", "INPUT_PATH=/io/tasks.json",
        "-e", "OUTPUT_PATH=/io/results.json",
    ]
    if env_file:
        cmd.extend(["--env-file", str(pathlib.Path(env_file).resolve())])
    cmd.append(image)

    print(f"\nRunning Docker benchmark: MODE={mode} image={image}")
    start = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - start
    remote_tokens = parse_remote_tokens(proc.stdout)
    print(f"Docker exit code: {proc.returncode}, elapsed: {elapsed:.1f}s")
    if not output_file.exists():
        print(proc.stdout)
        raise RuntimeError("No results.json produced")
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results, remote_tokens, proc.stdout, elapsed


def score_results(results, gt_by_id, prompt_by_id):
    by_cat = {cat: {"scores": [], "input_tokens": 0, "output_tokens": 0} for cat in DATASET_FILES}
    details = []
    for r in results:
        tid = r.get("task_id", "")
        answer = r.get("answer", "")
        meta = gt_by_id.get(tid)
        if not meta:
            continue
        cat = meta["category"]
        gt = meta["ground_truth"]
        prompt = prompt_by_id.get(tid, "")
        score, reason = grade_task(cat, answer, gt)
        in_tok, out_tok = estimate_tokens(prompt, answer)
        by_cat[cat]["scores"].append(score)
        by_cat[cat]["input_tokens"] += in_tok
        by_cat[cat]["output_tokens"] += out_tok
        details.append({
            "task_id": tid,
            "category": cat,
            "score": score,
            "reason": reason,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })

    total_scores = []
    total_in = 0
    total_out = 0
    for cat in by_cat:
        total_scores.extend(by_cat[cat]["scores"])
        total_in += by_cat[cat]["input_tokens"]
        total_out += by_cat[cat]["output_tokens"]
    accuracy = sum(total_scores) / len(total_scores) if total_scores else 0.0
    return accuracy, by_cat, details, total_in, total_out


def write_report(mode, accuracy, by_cat, details, total_in, total_out, remote_tokens, elapsed):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS_DIR / f"report_{ts}.md"
    json_path = REPORTS_DIR / f"report_{ts}.json"

    lines = [
        f"# Velora Benchmark Report — MODE={mode}",
        "",
        f"*Generated on: {datetime.now(timezone.utc).isoformat(timespec='seconds')}*",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        "| :--- | :--- |",
        f"| Accuracy | {accuracy*100:.1f}% ({sum(1 for d in details if d['score'] == 1.0)}/{len(details)}) |",
        f"| Total input tokens (est.) | {total_in} |",
        f"| Total output tokens (est.) | {total_out} |",
        f"| Remote tokens (hybrid) | {remote_tokens} |",
        f"| Execution time | {elapsed:.1f}s |",
        "",
        "## Category Breakdown",
        "",
        "| Category | Accuracy | Tasks | Input tokens | Output tokens |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for cat, data in by_cat.items():
        scores = data["scores"]
        cat_acc = sum(scores) / len(scores) if scores else 0.0
        lines.append(f"| {cat} | {cat_acc*100:.1f}% | {len(scores)} | {data['input_tokens']} | {data['output_tokens']} |")

    lines.extend([
        "",
        "## Per-Task Details",
        "",
        "| Task ID | Category | Score | Input | Output | Reason |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ])
    for d in details:
        lines.append(
            f"| {d['task_id']} | {d['category']} | {d['score']:.2f} | "
            f"{d['input_tokens']} | {d['output_tokens']} | {d['reason']} |"
        )
    lines.append("")
    md = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    report = {
        "mode": mode,
        "accuracy": accuracy,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "remote_tokens": remote_tokens,
        "elapsed_seconds": elapsed,
        "category_breakdown": {
            cat: {
                "accuracy": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0,
                "tasks": len(data["scores"]),
                "input_tokens": data["input_tokens"],
                "output_tokens": data["output_tokens"],
            }
            for cat, data in by_cat.items()
        },
        "details": details,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # also write latest_report.md/json for convenience
    (REPORTS_DIR / "latest_report.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "latest_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero", "hybrid"], required=True)
    parser.add_argument("--image", default="velora-agent:zero")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    tasks, gt_by_id = load_tasks()
    prompt_by_id = {t["task_id"]: t["prompt"] for t in tasks}
    print(f"Loaded {len(tasks)} tasks across {len(DATASET_FILES)} categories")

    results, remote_tokens, stdout, elapsed = run_docker(tasks, args.mode, args.image, args.env_file if args.mode == "hybrid" else None)
    accuracy, by_cat, details, total_in, total_out = score_results(results, gt_by_id, prompt_by_id)
    md_path = write_report(args.mode, accuracy, by_cat, details, total_in, total_out, remote_tokens, elapsed)

    print(f"\nMODE={args.mode}: accuracy={accuracy*100:.1f}%, tokens={total_in + total_out} (est. in+out), remote={remote_tokens}")
    print(f"Report written to: {md_path}")


if __name__ == "__main__":
    main()
