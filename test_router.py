"""Smoke test for the upgraded Velora Adaptive Routing Engine."""

import json
import logging

from app.router.decision_engine import AdaptiveRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


def run_test():
    router = AdaptiveRouter()

    prompts = [
        # Simple, should stay on 3B
        "Hello, how are you?",
        # Requires JSON → 3B will fail confidence, escalate to 7B
        'Please provide the output in JSON format: {"key": "value"}',
        # Simple question
        "What is 2+2?",
        # Long + JSON → 3B will fail (length > 100), escalate
        "Please provide the output in JSON format with a detailed explanation of quantum mechanics and all the equations involved in wave-particle duality.",
        # Similar to first → policy should learn
        "Hello, how are you doing today?",
        # Similar to complex JSON → policy should skip 3B proactively
        "Please provide the output in JSON format with a detailed explanation of string theory and the key mathematical frameworks.",
        # Code question
        "Write a Python function to sort a list using merge sort. ```python\ndef merge_sort(arr):\n    pass\n```",
        # Reasoning question
        "Explain step by step how gradient descent works in neural networks.",
        # Second run of similar queries to show learning
        "Hello there, how is everything going?",
        "Give me the output in JSON format describing the standard model of particle physics.",
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n{'=' * 60}")
        print(f"Query {i + 1}: {prompt[:70]}...")
        answer = router.route_query(prompt)
        print(f"Answer: {answer[:100]}")

    print(f"\n{'=' * 60}")
    print("FINAL METRICS")
    print("=" * 60)
    print(json.dumps(router.metrics.get_dashboard_metrics(), indent=2))


if __name__ == "__main__":
    run_test()
