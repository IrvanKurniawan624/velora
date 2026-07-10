"""History store with cosine-similarity lookup and aggregate statistics.

Provides the in-memory history database for the Velora Adaptive Routing
Engine.  Each record is indexed by its prompt embedding for fast k-NN
retrieval.  For the hackathon we keep a simple list; this can be swapped
for HNSW / FAISS / ChromaDB later.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from app.router.schemas import HistoryAggregateStats, HistoryRecord


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


class HistoryStore:
    """In-memory store of past inference episodes with similarity search."""

    def __init__(self, max_records: int = 10_000):
        self.records: List[HistoryRecord] = []
        self.max_records = max_records

    def add_record(self, record: HistoryRecord) -> None:
        """Append a new record, evicting the oldest if at capacity."""
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]

    def find_similar(
        self, embedding: List[float], k: int = 5
    ) -> List[Tuple[HistoryRecord, float]]:
        """Return the *k* most similar past records by cosine similarity."""
        if not embedding or not self.records:
            return []

        scored: List[Tuple[HistoryRecord, float]] = []
        for record in self.records:
            if record.features.embedding:
                sim = cosine_similarity(embedding, record.features.embedding)
                scored.append((record, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def aggregate_stats(
        self,
        neighbors: List[Tuple[HistoryRecord, float]],
        sim_threshold: float = 0.5,
    ) -> HistoryAggregateStats:
        """
        Compute per-model success rates and confidences from similar
        history records.  Only records with similarity ≥ *sim_threshold*
        are included.
        """
        relevant = [rec for rec, sim in neighbors if sim >= sim_threshold]
        if not relevant:
            return HistoryAggregateStats()

        model_successes: Dict[str, List[bool]] = defaultdict(list)
        model_confidences: Dict[str, List[float]] = defaultdict(list)
        model_escalations: Dict[str, List[bool]] = defaultdict(list)
        failure_reasons = Counter()

        for rec in relevant:
            model_successes[rec.chosen_model].append(rec.success)
            model_confidences[rec.chosen_model].append(rec.confidence_score)

            # If the record triggered escalation, we log it
            # The record is considered to trigger escalation if it wasn't successful
            # and it wasn't the final model (which would just be a failure)
            model_escalations[rec.chosen_model].append(rec.escalated or not rec.success)

            if rec.failure_reason:
                failure_reasons[rec.failure_reason] += 1

        success_rates = {m: sum(v) / len(v) for m, v in model_successes.items()}
        avg_confidences = {m: sum(v) / len(v) for m, v in model_confidences.items()}
        escalation_rates = {m: sum(v) / len(v) for m, v in model_escalations.items()}

        common_failures = [reason for reason, count in failure_reasons.most_common(3)]

        return HistoryAggregateStats(
            model_success_rates=success_rates,
            model_avg_confidence=avg_confidences,
            model_escalation_rates=escalation_rates,
            common_failure_reasons=common_failures,
            total_similar=len(relevant),
        )
