"""Live JIT Benchmark — runs against REAL models via LM Studio or mock fallback.

This script evaluates the Adaptive JIT Router against 5 baseline strategies
using the AMD Track 1 golden test set (8 categories x ~3 questions each).

Usage
-----
    # Real mode (LM Studio must be running with model loaded):
    python -m benchmarks.jit_live_benchmark

    # Mock/offline mode (no LM Studio needed):
    python -m benchmarks.jit_live_benchmark --mock

    # Only test specific category:
    python -m benchmarks.jit_live_benchmark --category math

    # Specify custom model name as shown in LM Studio:
    python -m benchmarks.jit_live_benchmark --model-3b "qwen3-3b-mlx"

Output
------
  - Live table printed to terminal
  - LIVE_RESULTS.md saved to project root
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path regardless of how script is invoked
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.router.decision_engine import AdaptiveRouter
from app.router.routing_policy import AVAILABLE_MODELS, MODEL_COST_WEIGHTS

logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", level=logging.WARNING
)
logger = logging.getLogger("jit_benchmark")
logger.setLevel(logging.INFO)

GOLDEN_SET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "golden_set.jsonl"
)
RESULTS_PATH = os.path.join(PROJECT_ROOT, "LIVE_RESULTS.md")

# ── AMD Track 1 categories (8) ────────────────────────────────────────────────
AMD_CATEGORIES = [
    "factual",
    "math",
    "sentiment",
    "summarisation",
    "ner",
    "debugging",
    "logic",
    "codegen",
]

# ── Mock model response simulator (used when --mock flag is set) ──────────────
MODEL_LATENCIES_MOCK = {"Qwen3-3B": 120, "Qwen3-7B": 380, "Fireworks": 950}


def _mock_response(model: str, prompt: str, gt: Any) -> Tuple[str, float, int]:
    """Deterministic mock based on model capability and category difficulty."""
    time.sleep(MODEL_LATENCIES_MOCK[model] / 1000)
    latency = float(MODEL_LATENCIES_MOCK[model])
    keywords = gt.get("keywords", []) if isinstance(gt, dict) else []

    success = False
    if model == "Fireworks":
        success = True
    elif model == "Qwen3-7B":
        success = random.random() > 0.15   # 85% success
    elif model == "Qwen3-3B":
        # 3B struggles on code, ner, complex prompts
        if any(kw in prompt.lower() for kw in ["def ", "function", "json", "extract", "entities"]):
            success = random.random() > 0.6
        else:
            success = random.random() > 0.25   # 75% success

    if success and keywords:
        return "The answer involves " + " ".join(keywords) + ".", latency, 60
    elif success:
        return f"[{model}] Processed successfully.", latency, 60
    else:
        return f"[{model}] I'm not certain about this.", latency, 20


# ── Evaluator ─────────────────────────────────────────────────────────────────
def evaluate(response: str, gt: Any, category: str) -> Tuple[bool, float]:
    """
    Score a response against ground truth.
    Returns (is_correct: bool, score: float in [0,1])
    """
    if not isinstance(gt, dict):
        return True, 1.0

    response_lower = response.lower()
    keywords = gt.get("keywords", [])

    if not keywords:
        return True, 1.0

    matched = [kw for kw in keywords if kw.lower() in response_lower]
    score = len(matched) / len(keywords)
    # Consider correct if >= 60% keywords matched
    return score >= 0.6, score


# ── Routing Strategies ────────────────────────────────────────────────────────
class BaseStrategy:
    name: str

    def __init__(self, invoke_fn):
        self.invoke = invoke_fn
        self.reset()

    def reset(self):
        self.results: List[Dict] = []

    def run(self, prompt: str, gt: Any, category: str) -> Dict:
        raise NotImplementedError


class StaticStrategy(BaseStrategy):
    def __init__(self, model: str, invoke_fn):
        self.model = model
        self.name = f"Always {model}"
        super().__init__(invoke_fn)

    def run(self, prompt: str, gt: Any, category: str) -> Dict:
        t0 = time.perf_counter()
        answer, latency, tokens = self.invoke(self.model, prompt)
        correct, score = evaluate(answer, gt, category)
        return {
            "model": self.model,
            "answer": answer,
            "latency_ms": latency,
            "tokens": tokens,
            "cost": MODEL_COST_WEIGHTS.get(self.model, 0) * tokens,
            "correct": correct,
            "score": score,
            "escalations": 0,
            "fireworks_used": self.model == "Fireworks",
        }


class RandomStrategy(BaseStrategy):
    name = "Random Router"

    def run(self, prompt: str, gt: Any, category: str) -> Dict:
        model = random.choice(AVAILABLE_MODELS)
        answer, latency, tokens = self.invoke(model, prompt)
        correct, score = evaluate(answer, gt, category)
        return {
            "model": model,
            "answer": answer,
            "latency_ms": latency,
            "tokens": tokens,
            "cost": MODEL_COST_WEIGHTS.get(model, 0) * tokens,
            "correct": correct,
            "score": score,
            "escalations": 0,
            "fireworks_used": model == "Fireworks",
        }


class RuleBasedStrategy(BaseStrategy):
    name = "Rule-Based Router"

    def run(self, prompt: str, gt: Any, category: str) -> Dict:
        p = prompt.lower()
        if category in ("codegen", "debugging") or "```" in p or "def " in p:
            model = "Qwen3-7B"
        elif category == "math" or any(w in p for w in ["calculate", "integral", "solve"]):
            model = "Fireworks"
        else:
            model = "Qwen3-3B"

        answer, latency, tokens = self.invoke(model, prompt)
        correct, score = evaluate(answer, gt, category)
        return {
            "model": model,
            "answer": answer,
            "latency_ms": latency,
            "tokens": tokens,
            "cost": MODEL_COST_WEIGHTS.get(model, 0) * tokens,
            "correct": correct,
            "score": score,
            "escalations": 0,
            "fireworks_used": model == "Fireworks",
        }


class AdaptiveStrategy(BaseStrategy):
    name = "Adaptive JIT Router *"

    def __init__(self, invoke_fn):
        super().__init__(invoke_fn)
        self.router = AdaptiveRouter(model_invoker=invoke_fn)

    def run(self, prompt: str, gt: Any, category: str) -> Dict:
        # Track what the router actually did via a wrapper invoker
        calls: List[Dict] = []
        original_invoke = self.router._invoke_model

        def tracking_invoke(model: str, p: str) -> Tuple[str, float, int]:
            answer, latency, tokens = original_invoke(model, p)
            calls.append({"model": model, "latency": latency, "tokens": tokens})
            return answer, latency, tokens

        self.router._invoke_model = tracking_invoke
        answer = self.router.route_query(prompt)
        self.router._invoke_model = original_invoke  # restore

        correct, score = evaluate(answer, gt, category)
        total_latency = sum(c["latency"] for c in calls)
        total_tokens = sum(c["tokens"] for c in calls)
        final_model = calls[-1]["model"] if calls else "unknown"
        fw_used = any(c["model"] == "Fireworks" for c in calls)
        total_cost = sum(
            MODEL_COST_WEIGHTS.get(c["model"], 0) * c["tokens"] for c in calls
        )

        return {
            "model": final_model,
            "answer": answer,
            "latency_ms": total_latency,
            "tokens": total_tokens,
            "cost": total_cost,
            "correct": correct,
            "score": score,
            "escalations": max(0, len(calls) - 1),
            "fireworks_used": fw_used,
        }


# ── Dataset Loader ────────────────────────────────────────────────────────────
def load_golden_set(category_filter: Optional[str] = None) -> List[Dict]:
    items = []
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if category_filter and rec.get("category") != category_filter:
                    continue
                items.append(rec)
            except Exception:
                pass
    return items


# ── Benchmark Runner ──────────────────────────────────────────────────────────
def run_benchmark(strategies: List[BaseStrategy], items: List[Dict], verbose: bool = False):
    """Run all strategies over all items and collect per-category metrics."""
    results: Dict[str, List[Dict]] = {s.name: [] for s in strategies}

    total = len(items)
    for i, item in enumerate(items, 1):
        prompt = item["messages"][-1]["content"]
        gt = item.get("ground_truth", {})
        category = item.get("category", "unknown")
        difficulty = item.get("difficulty", "?")

        logger.info(
            "[%d/%d] category=%-15s difficulty=%-6s prompt=%.50s...",
            i, total, category, difficulty, prompt
        )

        for strategy in strategies:
            res = strategy.run(prompt, gt, category)
            res["category"] = category
            res["difficulty"] = difficulty
            res["prompt_preview"] = prompt[:60]
            results[strategy.name].append(res)

            if verbose:
                status = "OK" if res["correct"] else "FAIL"
                logger.info(
                    "  %-30s %s model=%-12s lat=%5.0fms tokens=%3d fw=%s",
                    strategy.name, status, res["model"],
                    res["latency_ms"], res["tokens"],
                    "YES" if res["fireworks_used"] else "no",
                )

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    if total == 0:
        return {}
    return {
        "total": total,
        "accuracy": sum(r["correct"] for r in results) / total,
        "avg_score": sum(r["score"] for r in results) / total,
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / total,
        "avg_tokens": sum(r["tokens"] for r in results) / total,
        "avg_cost": sum(r["cost"] for r in results) / total,
        "fw_rate": sum(r["fireworks_used"] for r in results) / total,
        "escalation_rate": sum(r["escalations"] for r in results) / total,
    }


def compute_category_breakdown(results: List[Dict]) -> Dict[str, Dict]:
    by_cat: Dict[str, List] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    breakdown = {}
    for cat in AMD_CATEGORIES:
        cat_results = by_cat.get(cat, [])
        if cat_results:
            total = len(cat_results)
            breakdown[cat] = {
                "total": total,
                "accuracy": sum(r["correct"] for r in cat_results) / total,
                "avg_latency_ms": sum(r["latency_ms"] for r in cat_results) / total,
                "fw_rate": sum(r["fireworks_used"] for r in cat_results) / total,
            }
        else:
            breakdown[cat] = {"total": 0, "accuracy": 0.0, "avg_latency_ms": 0, "fw_rate": 0}
    return breakdown


def print_report(all_results: Dict[str, List[Dict]]):
    """Print a rich terminal report and save LIVE_RESULTS.md."""
    separator = "-" * 108
    print(f"\n{'='*108}")
    print("  AMD Track 1 -- JIT Router Live Benchmark Results")
    print(f"{'='*108}\n")

    # -- Overall Summary Table --
    header = f"{'Strategy':<30} {'Accuracy':>10} {'Avg Latency':>13} {'FW Call %':>11} {'Avg Cost':>10} {'Escalation%':>13} {'Avg Tokens':>12}"
    print(header)
    print(separator)

    strategy_metrics: Dict[str, Dict] = {}
    for name, results in all_results.items():
        m = compute_metrics(results)
        strategy_metrics[name] = m
        fw_indicator = "[HIGH]" if m["fw_rate"] > 0.3 else ("[LOW]" if m["fw_rate"] < 0.1 else "[MED]")
        print(
            f"{name:<30} {m['accuracy']:>9.1%} {m['avg_latency_ms']:>11.0f}ms "
            f"{m['fw_rate']:>10.1%} {m['avg_cost']:>10.4f} "
            f"{m['escalation_rate']:>12.2f}  {m['avg_tokens']:>11.1f}  {fw_indicator}"
        )

    print()

    # -- Category Breakdown for JIT Router --
    jit_name = "Adaptive JIT Router *"
    if jit_name in all_results:
        jit_results = all_results[jit_name]
        cat_data = compute_category_breakdown(jit_results)

        print(f"\n{'-'*60}")
        print("  Category Breakdown -- Adaptive JIT Router *")
        print(f"{'-'*60}")
        print(f"  {'Category':<18} {'Count':>5} {'Accuracy':>10} {'FW Rate':>9} {'Avg Latency':>13}")
        print(f"  {'-'*58}")
        for cat in AMD_CATEGORIES:
            d = cat_data[cat]
            if d["total"] > 0:
                print(
                    f"  {cat:<18} {d['total']:>5} {d['accuracy']:>9.1%} "
                    f"{d['fw_rate']:>8.1%} {d['avg_latency_ms']:>11.0f}ms"
                )

    print(f"\n{'='*108}\n")

    # -- Save Markdown Report --
    lines = [
        "# AMD Track 1 -- JIT Router Live Benchmark Report\n",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Golden Set:** `benchmarks/datasets/golden_set.jsonl`\n",
        f"**Categories:** {', '.join(AMD_CATEGORIES)}\n\n",
        "## Overall Strategy Comparison\n",
        "| Strategy | Accuracy | Avg Latency | Fireworks % | Avg Cost | Escalation Rate | Avg Tokens |",
        "|----------|----------|-------------|-------------|----------|-----------------|------------|",
    ]
    for name, m in strategy_metrics.items():
        lines.append(
            f"| {name} | {m['accuracy']:.1%} | {m['avg_latency_ms']:.0f}ms "
            f"| {m['fw_rate']:.1%} | {m['avg_cost']:.4f} "
            f"| {m['escalation_rate']:.2f} | {m['avg_tokens']:.1f} |"
        )

    if jit_name in all_results:
        cat_data = compute_category_breakdown(all_results[jit_name])
        lines += [
            "\n## Category Breakdown -- Adaptive JIT Router\n",
            "| Category | Count | Accuracy | Fireworks % | Avg Latency |",
            "|----------|-------|----------|-------------|-------------|",
        ]
        for cat in AMD_CATEGORIES:
            d = cat_data[cat]
            if d["total"] > 0:
                lines.append(
                    f"| {cat} | {d['total']} | {d['accuracy']:.1%} "
                    f"| {d['fw_rate']:.1%} | {d['avg_latency_ms']:.0f}ms |"
                )

    lines.append(
        "\n\n> **Note:** Lower Fireworks % = more queries handled locally = lower cost.\n"
        "> Adaptive JIT Router should achieve near-baseline accuracy with significantly lower Fireworks usage.\n"
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report saved -> %s", RESULTS_PATH)
    print(f"  [SAVED] Full report saved to: {RESULTS_PATH}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="JIT Router Live Benchmark — AMD Track 1"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use simulated model responses (no LM Studio required)"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=AMD_CATEGORIES,
        help="Run only this category"
    )
    parser.add_argument(
        "--model-3b", type=str, default=None,
        help="Override Qwen 3B model name as shown in LM Studio"
    )
    parser.add_argument(
        "--model-7b", type=str, default=None,
        help="Override Qwen 7B model name as shown in LM Studio"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-query results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    random.seed(args.seed)

    # ── Select invoker ────────────────────────────────────────────────────────
    if args.mock:
        logger.info("Running in MOCK mode (no LM Studio needed).")

        def invoke_fn(model: str, prompt: str) -> Tuple[str, float, int]:
            items = load_golden_set()
            # Get GT for this prompt
            gt: Any = {}
            for item in items:
                if item["messages"][-1]["content"] == prompt:
                    gt = item.get("ground_truth", {})
                    break
            return _mock_response(model, prompt, gt)

    else:
        # Real mode: check LM Studio is up first
        from app.router.lmstudio_invoker import check_lmstudio_connection, create_real_invoker
        from app.config import settings as cfg

        # Apply CLI overrides
        if args.model_3b:
            cfg.lmstudio_model_3b = args.model_3b
        if args.model_7b:
            cfg.lmstudio_model_7b = args.model_7b

        print(f"\n  [*] Connecting to LM Studio at {cfg.lmstudio_base_url} ...")
        if not check_lmstudio_connection():
            print(
                "\n  [FAIL] LM Studio is not reachable!\n"
                "  -> Make sure LM Studio is open and a model is loaded.\n"
                "  -> Then try again, or use --mock for offline testing.\n"
            )
            sys.exit(1)

        print(
            f"  [OK] LM Studio connected.\n"
            f"  3B model: {cfg.lmstudio_model_3b}\n"
            f"  7B model: {cfg.lmstudio_model_7b}\n"
        )
        invoke_fn = create_real_invoker()

    # ── Load dataset ──────────────────────────────────────────────────────────
    items = load_golden_set(category_filter=args.category)
    if not items:
        print(f"  [WARN] No items found for filter: {args.category}")
        sys.exit(1)
    print(f"  [DATA] Loaded {len(items)} test queries across {len(AMD_CATEGORIES)} categories.\n")

    # ── Build strategies ──────────────────────────────────────────────────────
    strategies = [
        StaticStrategy("Qwen3-3B", invoke_fn),
        StaticStrategy("Qwen3-7B", invoke_fn),
        StaticStrategy("Fireworks", invoke_fn),
        RandomStrategy(invoke_fn),
        RuleBasedStrategy(invoke_fn),
        AdaptiveStrategy(invoke_fn),
    ]

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results = run_benchmark(strategies, items, verbose=args.verbose)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(all_results)


if __name__ == "__main__":
    main()
