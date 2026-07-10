import os
import json
import re
import asyncio
from app.services.agent import AgentService
from app.clients.local_client import LocalClient
from app.clients.fireworks_client import FireworksClient
from app.config import Settings

# Helper for color printing in terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clean_code(text: str) -> str:
    # Strip markdown block indicators
    return re.sub(r"```python|```", "", text).strip()

def evaluate_factual(reply: str, ground_truth: str) -> tuple[float, str]:
    try:
        truth_data = json.loads(ground_truth)
        keywords = truth_data.get("keywords", [])
        matched = [kw for kw in keywords if kw.lower() in reply.lower()]
        score = len(matched) / len(keywords) if keywords else 1.0
        return score, f"Matched {len(matched)}/{len(keywords)} keywords: {matched}"
    except Exception as e:
        return 0.0, f"Error: {e}"

def evaluate_math(reply: str, ground_truth: str) -> tuple[float, str]:
    expected = str(ground_truth).strip()
    nums = re.findall(r"\d+", reply)
    model_answer = nums[-1] if nums else ""
    score = 1.0 if model_answer == expected else 0.0
    return score, f"Model answer: {model_answer} | Expected: {expected}"

def evaluate_sentiment(reply: str, ground_truth: str) -> tuple[float, str]:
    try:
        expected = json.loads(ground_truth)
        clean_reply = re.sub(r"```json|```", "", reply).strip()
        data = json.loads(clean_reply)
        sentiment_match = data.get("sentiment", "").lower() == expected.get("sentiment", "").lower()
        has_reason = "reason" in data and len(data["reason"]) > 0
        score = 1.0 if (sentiment_match and has_reason) else 0.5 if sentiment_match else 0.0
        return score, f"Sentiment match: {sentiment_match}, Has reason: {has_reason}"
    except Exception as e:
        return 0.0, f"Failed to parse JSON: {e}"

def evaluate_summarisation(reply: str, ground_truth: str) -> tuple[float, str]:
    try:
        clean_reply = re.sub(r"```json|```", "", reply).strip()
        data = json.loads(clean_reply)
        bullets = data.get("bullets", [])
        correct_length = len(bullets) == 2
        score = 1.0 if correct_length else 0.5 if len(bullets) > 0 else 0.0
        return score, f"Found {len(bullets)} bullets (expected 2)"
    except Exception as e:
        return 0.0, f"Failed to parse JSON: {e}"

def evaluate_ner(reply: str, ground_truth: str) -> tuple[float, str]:
    try:
        expected = json.loads(ground_truth)
        clean_reply = re.sub(r"```json|```", "", reply).strip()
        data = json.loads(clean_reply)
        score = 1.0
        reasons = []
        for key in ["PERSON", "ORG", "LOC"]:
            model_entities = [e.lower() for e in data.get(key, [])]
            expected_entities = [e.lower() for e in expected.get(key, [])]
            matches = set(model_entities).intersection(set(expected_entities))
            if len(matches) != len(expected_entities):
                score -= 0.33
                reasons.append(f"Mismatch in {key}")
            else:
                reasons.append(f"Correct {key}")
        score = max(0.0, min(1.0, score))
        return score, ", ".join(reasons)
    except Exception as e:
        return 0.0, f"Failed to parse JSON: {e}"

def evaluate_debugging(reply: str, ground_truth: str) -> tuple[float, str]:
    code = clean_code(reply)
    namespace = {}
    try:
        exec(code, namespace)
        if "sum_evens" not in namespace:
            return 0.0, "Function 'sum_evens' not defined in response"
        sum_evens = namespace["sum_evens"]
        assert sum_evens([1, 2, 3, 4]) == 6
        assert sum_evens([1, 3, 5]) == 0
        assert sum_evens([]) == 0
        return 1.0, "Passes all debugger unit assertions"
    except Exception as e:
        return 0.0, f"Execution/assertion failed: {e}"

def evaluate_logic(reply: str, ground_truth: str) -> tuple[float, str]:
    expected = str(ground_truth).strip().lower()
    score = 1.0 if expected in reply.lower() else 0.0
    return score, f"Found '{expected}' in response: {expected in reply.lower()}"

def evaluate_codegen(reply: str, ground_truth: str) -> tuple[float, str]:
    code = clean_code(reply)
    namespace = {}
    try:
        exec(code, namespace)
        if "is_palindrome" not in namespace:
            return 0.0, "Function 'is_palindrome' not defined in response"
        is_palindrome = namespace["is_palindrome"]
        assert is_palindrome("A man, a plan, a canal: Panama") is True
        assert is_palindrome("hello") is False
        assert is_palindrome("racecar") is True
        return 1.0, "Passes all palindrome assertions"
    except Exception as e:
        return 0.0, f"Execution/assertion failed: {e}"

EVALUATORS = {
    "01_factual.jsonl": evaluate_factual,
    "02_math.jsonl": evaluate_math,
    "03_sentiment.jsonl": evaluate_sentiment,
    "04_summarisation.jsonl": evaluate_summarisation,
    "05_ner.jsonl": evaluate_ner,
    "06_debugging.jsonl": evaluate_debugging,
    "07_logic.jsonl": evaluate_logic,
    "08_codegen.jsonl": evaluate_codegen,
}

async def run_evaluation():
    settings = Settings()
    print(f"{Colors.HEADER}=== VELORA AGENT BENCHMARK RUNNER ==={Colors.ENDC}")
    print("Initializing Agent and Clients...")
    local = LocalClient()
    fireworks = FireworksClient()
    agent = AgentService(local_client=local, fireworks_client=fireworks)
    
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "datasets")
    
    results = {}
    overall_score = 0.0
    total_categories = 0
    
    files = sorted([f for f in os.listdir(datasets_dir) if f.endswith(".jsonl") and not f.startswith("fine_tuning")])
    
    for filename in files:
        filepath = os.path.join(datasets_dir, filename)
        category_name = filename.split(".")[0].replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").strip("_")
        
        print(f"\n{Colors.OKBLUE}Evaluating Category: {category_name.upper()} ({filename}){Colors.ENDC}")
        evaluator = EVALUATORS.get(filename)
        if not evaluator:
            print(f"{Colors.WARNING}No evaluator registered for {filename}, skipping.{Colors.ENDC}")
            continue
            
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        
        category_scores = []
        for idx, row in enumerate(rows):
            prompt = row["messages"][-1]["content"]
            ground_truth = row.get("ground_truth", "")
            
            print(f"  - Running query {idx+1}/{len(rows)}... ", end="", flush=True)
            
            try:
                # Call agent
                reply = await agent.run(prompt)
                score, reason = evaluator(reply, ground_truth)
                category_scores.append(score)
                
                status_color = Colors.OKGREEN if score == 1.0 else Colors.WARNING if score > 0.0 else Colors.FAIL
                print(f"{status_color}Score: {score:.2f} ({reason}){Colors.ENDC}")
            except Exception as e:
                category_scores.append(0.0)
                print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
                
        category_avg = sum(category_scores) / len(category_scores) if category_scores else 0.0
        results[category_name] = {
            "score": category_avg,
            "details": category_scores
        }
        overall_score += category_avg
        total_categories += 1
        print(f"{Colors.OKCYAN}Category Average: {category_avg*100:.1f}%{Colors.ENDC}")

    final_avg = (overall_score / total_categories) if total_categories else 0.0
    print(f"\n{Colors.HEADER}====================================={Colors.ENDC}")
    print(f"{Colors.HEADER}=== FINAL BENCHMARK SCORE: {final_avg*100:.1f}% ==={Colors.ENDC}")
    print(f"{Colors.HEADER}====================================={Colors.ENDC}")
    
    # Save a report
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "benchmark_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark report to {report_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
