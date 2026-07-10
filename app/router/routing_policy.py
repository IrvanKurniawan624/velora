"""LinUCB contextual-bandit routing policy for model selection.

Implements the disjoint LinUCB algorithm (Li et al., 2010) with optional
discount factor γ for non-stationary environments (inspired by
ParetoBandit / PILOT).  Each "arm" is a candidate model (Qwen3-3B,
Qwen3-7B, Fireworks).  The context vector is built from prompt features
and historical success priors.

Algorithm per arm *k*:
    θ_k = A_k⁻¹ · b_k
    p_k = θ_k · x + α · √(xᵀ A_k⁻¹ x)
    Choose arm with highest p_k.
    After observing reward r:
        A_k ← γ · A_k + x · xᵀ
        b_k ← γ · b_k + r · x
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

from app.router.schemas import (
    HistoryAggregateStats,
    PromptFeatures,
)

logger = logging.getLogger(__name__)

AVAILABLE_MODELS: List[str] = ["Qwen3-3B", "Qwen3-7B", "Fireworks"]

# Cost weights used in reward computation: r = success − λ · cost_weight
MODEL_COST_WEIGHTS: Dict[str, float] = {
    "Qwen3-3B": 0.0,  # local, free
    "Qwen3-7B": 0.05,  # local, slightly more expensive (compute)
    "Fireworks": 0.4,  # remote, premium
}


class LinUCBArm:
    """State for one LinUCB arm (one model)."""

    def __init__(self, d: int, gamma: float = 1.0):
        """
        Parameters
        ----------
        d : int
            Dimensionality of the context vector.
        gamma : float
            Discount factor ∈ (0, 1].  1.0 = no discounting.
        """
        self.d = d
        self.gamma = gamma
        # A_k = identity matrix  (d × d)
        self.A: List[List[float]] = [
            [1.0 if i == j else 0.0 for j in range(d)] for i in range(d)
        ]
        # b_k = zero vector  (d × 1)
        self.b: List[float] = [0.0] * d

    def predict(self, x: List[float], alpha: float) -> float:
        """
        Compute the UCB score:  p = θ · x + α · √(xᵀ A⁻¹ x).
        """
        A_inv = _matrix_inverse(self.A)
        theta = _mat_vec(A_inv, self.b)

        exploitation = _dot(theta, x)
        A_inv_x = _mat_vec(A_inv, x)
        exploration = alpha * math.sqrt(max(_dot(x, A_inv_x), 0.0))

        return exploitation + exploration

    def update(self, x: List[float], reward: float) -> None:
        """
        LinUCB parameter update with optional discounting:
            A ← γ·A + x·xᵀ
            b ← γ·b + r·x
        """
        d = self.d
        for i in range(d):
            for j in range(d):
                self.A[i][j] = self.gamma * self.A[i][j] + x[i] * x[j]
            self.b[i] = self.gamma * self.b[i] + reward * x[i]


class RoutingPolicy:
    """
    LinUCB-based routing policy.

    Falls back to cheapest-model-first when context dimension is not yet
    established (first call before ``initialize`` is invoked).
    """

    def __init__(
        self,
        alpha: float = 1.5,
        gamma: float = 0.995,
        cost_lambda: float = 0.3,
    ):
        """
        Parameters
        ----------
        alpha : float
            Exploration bonus coefficient for UCB.
        gamma : float
            Discount factor for non-stationary adaptation.
        cost_lambda : float
            Trade-off weight between success and cost in reward.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.cost_lambda = cost_lambda
        self.arms: Dict[str, LinUCBArm] = {}
        self._initialized = False

    def _ensure_initialized(self, d: int) -> None:
        """Lazily create arms when the context dimension is first known."""
        if self._initialized:
            return
        for model in AVAILABLE_MODELS:
            self.arms[model] = LinUCBArm(d=d, gamma=self.gamma)
        self._initialized = True
        logger.info(
            "LinUCB policy initialized with d=%d, α=%.3f, γ=%.4f",
            d,
            self.alpha,
            self.gamma,
        )

    def select_model(
        self,
        context_vector: List[float],
        history_stats: Optional[HistoryAggregateStats] = None,
    ) -> str:
        """
        Choose the best model given a context vector.

        Parameters
        ----------
        context_vector : list[float]
            Normalized feature vector from FeatureExtractor.build_context_vector.
        history_stats : HistoryAggregateStats, optional
            Aggregated priors from similar past queries.

        Returns
        -------
        str
            Name of the chosen model.
        """
        d = len(context_vector)
        self._ensure_initialized(d)

        # --- Proactive skip from strong history signals ------------------
        if history_stats and history_stats.total_similar >= 3:
            sr = history_stats.model_success_rates
            # If 3B almost always fails on similar queries, skip it
            if sr.get("Qwen3-3B", 1.0) < 0.3:
                if sr.get("Qwen3-7B", 1.0) < 0.3:
                    logger.info("History skip: both local models poor → Fireworks")
                    return "Fireworks"
                logger.info("History skip: 3B poor → Qwen3-7B")
                return "Qwen3-7B"

        # --- LinUCB arm selection ----------------------------------------
        best_model = AVAILABLE_MODELS[0]
        best_score = -float("inf")

        for model in AVAILABLE_MODELS:
            arm = self.arms[model]
            ucb_score = arm.predict(context_vector, self.alpha)
            # Subtract cost penalty so cheaper models are preferred at equal UCB
            cost_penalty = MODEL_COST_WEIGHTS.get(model, 0.0)
            adjusted_score = ucb_score - cost_penalty

            logger.debug(
                "  %s: UCB=%.4f  cost_pen=%.2f  adj=%.4f",
                model,
                ucb_score,
                cost_penalty,
                adjusted_score,
            )

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_model = model

        logger.info("LinUCB selected: %s (score=%.4f)", best_model, best_score)
        return best_model

    def update_policy(
        self,
        model: str,
        context_vector: List[float],
        reward: float,
    ) -> None:
        """
        Update the chosen arm after observing the reward.

        Parameters
        ----------
        model : str
            The model that was used.
        context_vector : list[float]
            The context vector that was used to make the decision.
        reward : float
            Observed reward signal.
        """
        d = len(context_vector)
        self._ensure_initialized(d)

        if model in self.arms:
            self.arms[model].update(context_vector, reward)
            logger.debug("Updated arm %s with reward=%.4f", model, reward)

    @staticmethod
    def compute_reward(success: bool, model: str) -> float:
        """
        Reward = success − λ · cost.

        Simple reward: success ∈ {0, 1}, cost from MODEL_COST_WEIGHTS.
        """
        cost = MODEL_COST_WEIGHTS.get(model, 0.0)
        return (1.0 if success else 0.0) - cost


# ─── Pure-Python linear algebra helpers (no numpy dependency) ────────


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    return [_dot(row, v) for row in M]


def _matrix_inverse(M: List[List[float]]) -> List[List[float]]:
    """
    Gauss-Jordan inverse of a square matrix.
    For the small dimensions we use (d ≈ 10) this is efficient enough.
    """
    n = len(M)
    # Augmented matrix [M | I]
    aug = [
        row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)
    ]

    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            # Near-singular: add small regularization
            aug[col][col] += 1e-6
            pivot = aug[col][col]

        for j in range(2 * n):
            aug[col][j] /= pivot

        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    return [row[n:] for row in aug]
