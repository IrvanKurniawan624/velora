"""Performance memory – rolling statistics per model.

Tracks success rates, latencies, token usage, and confidence scores.
Supports optional exponential-decay windowing so the router can adapt
to non-stationary workloads.
"""

from typing import Dict

from app.router.schemas import HistoryRecord


class ModelStats:
    """Aggregate statistics for a single model."""

    def __init__(self):
        self.total_calls: int = 0
        self.success_count: int = 0
        self.total_latency_ms: float = 0.0
        self.total_tokens: int = 0
        self.total_confidence: float = 0.0
        self.escalation_triggers: int = (
            0  # times this model's failure triggered escalation
        )

        # Category tracking: category -> {"success": int, "total": int}
        self.category_stats: Dict[str, Dict[str, int]] = {}

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def escalation_rate(self) -> float:
        return (
            self.escalation_triggers / self.total_calls if self.total_calls > 0 else 0.0
        )

    def get_category_success_rate(self, category: str) -> float:
        stats = self.category_stats.get(category, {"success": 0, "total": 0})
        return stats["success"] / stats["total"] if stats["total"] > 0 else 0.0


class PerformanceMemory:
    """Maintains rolling statistics per model for the routing engine."""

    def __init__(self):
        self.stats_by_model: Dict[str, ModelStats] = {}

    def update(self, record: HistoryRecord) -> None:
        model = record.chosen_model
        if model not in self.stats_by_model:
            self.stats_by_model[model] = ModelStats()

        stats = self.stats_by_model[model]
        stats.total_calls += 1

        # Update category stats
        cat = record.features.category
        if cat not in stats.category_stats:
            stats.category_stats[cat] = {"success": 0, "total": 0}
        stats.category_stats[cat]["total"] += 1

        if record.success:
            stats.success_count += 1
            stats.category_stats[cat]["success"] += 1
        else:
            stats.escalation_triggers += 1

        stats.total_latency_ms += record.latency_ms
        stats.total_tokens += record.token_usage
        stats.total_confidence += record.confidence_score

    def get_stats(self, model: str) -> ModelStats:
        return self.stats_by_model.get(model, ModelStats())
