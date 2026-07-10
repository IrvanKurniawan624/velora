"""Adaptive JIT Decision Engine – orchestrates the full routing pipeline.

Sequence per query:
  1. Feature extraction → context vector
  2. History lookup → aggregate priors
  3. LinUCB model selection
  4. Inference (mock or real)
  5. Confidence estimation
  6. Escalation loop (3B → 7B → Fireworks)
  7. Reward computation + policy update
  8. History & metrics logging
"""

import logging
import time
import uuid
from typing import Callable, Optional, Tuple

from app.router.confidence_estimator import ConfidenceEstimator
from app.router.feature_extractor import FeatureExtractor
from app.router.history_store import HistoryStore
from app.router.performance_memory import PerformanceMemory
from app.router.router_metrics import RouterMetrics
from app.router.routing_policy import AVAILABLE_MODELS, RoutingPolicy
from app.router.schemas import HistoryRecord

logger = logging.getLogger(__name__)

# Type alias for a model invocation function:
#   (model_name, prompt) -> (answer_text, latency_ms, token_count)
ModelInvoker = Callable[[str, str], Tuple[str, float, int]]


def _default_model_invoker(model: str, prompt: str) -> Tuple[str, float, int]:
    """
    Mock model invocation used during development and testing.

    In production this would dispatch to the actual Qwen local server or
    the Fireworks HTTP API.
    """
    delay = 0.05 if "3B" in model else (0.15 if "7B" in model else 0.4)
    time.sleep(delay)
    latency = delay * 1000

    # Simulate smaller models failing on complex / long prompts
    if "3B" in model and len(prompt) > 100:
        return "I don't know.", latency, 10

    if "json" in prompt.lower() and "3B" in model:
        return "Here is your JSON: not a json format at all", latency, 15

    answer = f"[{model}] Processed: {prompt[:30]}..."
    if "json" in prompt.lower():
        answer += ' ```json\n{"status": "success"}\n```'

    return answer, latency, 50


class AdaptiveRouter:
    """
    Entry point for the Velora Adaptive Routing Engine (VARE).

    Usage::

        router = AdaptiveRouter()
        answer = router.route_query("What is 2+2?")
        print(router.metrics.get_dashboard_metrics())
    """

    def __init__(
        self,
        model_invoker: Optional[ModelInvoker] = None,
        confidence_threshold: float = 0.7,
        linucb_alpha: float = 1.5,
        linucb_gamma: float = 0.995,
        cost_lambda: float = 0.3,
    ):
        self.history_store = HistoryStore()
        self.performance_memory = PerformanceMemory()
        self.policy = RoutingPolicy(
            alpha=linucb_alpha,
            gamma=linucb_gamma,
            cost_lambda=cost_lambda,
        )
        self.metrics = RouterMetrics(self.performance_memory)
        self.feature_extractor = FeatureExtractor()
        self.confidence_estimator = ConfidenceEstimator()
        self.confidence_threshold = confidence_threshold
        self._invoke_model: ModelInvoker = model_invoker or _default_model_invoker

    def route_query(self, prompt: str) -> str:
        """
        Main JIT routing pipeline.

        Parameters
        ----------
        prompt : str
            The user query.

        Returns
        -------
        str
            The best answer obtained (possibly after escalation).
        """
        query_id = str(uuid.uuid4())
        logger.info("── Query %s ──", query_id[:8])

        # ── 1. Feature Extraction ──────────────────────────────────────
        features = self.feature_extractor.extract(prompt)

        # ── 2. History Lookup ──────────────────────────────────────────
        neighbors = self.history_store.find_similar(features.embedding, k=5)
        history_stats = self.history_store.aggregate_stats(neighbors)

        # ── 3. Build context vector with history priors ────────────────
        context_vector = self.feature_extractor.build_context_vector(
            features,
            hist_success_3b=history_stats.model_success_rates.get("Qwen3-3B", 0.5),
            hist_success_7b=history_stats.model_success_rates.get("Qwen3-7B", 0.5),
            hist_success_fw=history_stats.model_success_rates.get("Fireworks", 0.5),
        )

        # ── 4. Routing Decision (LinUCB) ───────────────────────────────
        initial_model = self.policy.select_model(context_vector, history_stats)

        # ── 5. Inference + Escalation Loop ─────────────────────────────
        current_idx = AVAILABLE_MODELS.index(initial_model)
        escalated = False
        escalation_count = 0
        final_answer = ""
        final_model = initial_model
        final_confidence = 0.0
        final_success = False
        total_latency = 0.0
        total_tokens = 0

        while current_idx < len(AVAILABLE_MODELS):
            model_to_use = AVAILABLE_MODELS[current_idx]
            logger.info("  Trying model: %s", model_to_use)

            # Inference
            answer, latency, tokens = self._invoke_model(model_to_use, prompt)
            total_latency += latency
            total_tokens += tokens

            # Confidence check
            breakdown = self.confidence_estimator.estimate(prompt, answer)
            confidence = breakdown.overall
            success = confidence >= self.confidence_threshold
            failure_reason = (
                breakdown.failure_reasons[0] if breakdown.failure_reasons else None
            )

            logger.info(
                "    confidence=%.3f  success=%s  latency=%.1fms  tokens=%d",
                confidence,
                success,
                latency,
                tokens,
            )

            # Log each attempt to history
            attempt_record = HistoryRecord(
                query_id=query_id,
                prompt_hash=HistoryRecord.compute_prompt_hash(prompt),
                prompt=prompt,
                features=features,
                context_vector=context_vector,
                chosen_model=model_to_use,
                latency_ms=latency,
                token_usage=tokens,
                confidence_score=confidence,
                confidence_breakdown=breakdown,
                success=success,
                escalated=escalated,
                escalation_count=escalation_count,
                failure_reason=failure_reason,
            )
            self.history_store.add_record(attempt_record)
            self.performance_memory.update(attempt_record)

            if success:
                final_answer = answer
                final_model = model_to_use
                final_confidence = confidence
                final_success = True
                break

            # Escalate
            escalated = True
            escalation_count += 1
            current_idx += 1

        # If all models failed, take the last answer anyway
        if not final_success:
            final_answer = answer  # type: ignore[possibly-undefined]
            final_model = AVAILABLE_MODELS[-1]
            final_confidence = confidence  # type: ignore[possibly-undefined]

        # ── 6. Reward computation ──────────────────────────────────────
        reward = self.policy.compute_reward(final_success, final_model)

        # ── 7. Policy update (update the arm that was *initially* chosen)
        self.policy.update_policy(initial_model, context_vector, reward)

        # ── 8. Metrics logging ─────────────────────────────────────────
        self.metrics.record_request(escalated, reward)

        logger.info(
            "  Result: model=%s  confidence=%.3f  reward=%.3f  escalations=%d",
            final_model,
            final_confidence,
            reward,
            escalation_count,
        )

        return final_answer
