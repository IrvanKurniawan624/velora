"""Router-level metrics dashboard.

Collects and exposes aggregate statistics for evaluation and tuning:
  • Per-model call counts and usage percentages
  • Escalation rate
  • Estimated cost savings
  • Average latency and confidence
"""

from typing import Any, Dict

from app.router.performance_memory import PerformanceMemory
from app.router.routing_policy import AVAILABLE_MODELS, MODEL_COST_WEIGHTS


class RouterMetrics:
    """Instruments runtime metrics for the routing engine."""

    def __init__(self, performance_memory: PerformanceMemory):
        self.perf_memory = performance_memory
        self.total_requests: int = 0
        self.escalations: int = 0
        self.total_reward: float = 0.0

    def record_request(self, escalated: bool, reward: float = 0.0) -> None:
        self.total_requests += 1
        if escalated:
            self.escalations += 1
        self.total_reward += reward

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        total = self.total_requests or 1  # avoid division by zero

        metrics: Dict[str, Any] = {
            "total_requests": self.total_requests,
            "escalations": self.escalations,
            "escalation_rate": round(self.escalations / total, 4),
            "avg_reward": round(self.total_reward / total, 4),
            "model_stats": {},
        }

        fireworks_calls = self.perf_memory.get_stats("Fireworks").total_calls
        metrics["fireworks_usage_pct"] = round(fireworks_calls / total, 4)
        metrics["local_inference_pct"] = round(1.0 - fireworks_calls / total, 4)

        for model in AVAILABLE_MODELS:
            stats = self.perf_memory.get_stats(model)
            metrics["model_stats"][model] = {
                "calls": stats.total_calls,
                "usage_pct": round(stats.total_calls / total, 4),
                "success_rate": round(stats.success_rate, 4),
                "avg_latency_ms": round(stats.avg_latency, 2),
                "avg_confidence": round(stats.avg_confidence, 4),
                "avg_tokens": round(stats.avg_tokens, 1),
                "cost_weight": MODEL_COST_WEIGHTS.get(model, 0.0),
            }

        return metrics
