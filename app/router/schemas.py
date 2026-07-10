"""Pydantic schemas for the Velora Adaptive Routing Engine."""

import hashlib
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PromptFeatures(BaseModel):
    """Features extracted from a prompt for routing decisions."""

    prompt_length: int = Field(..., description="Length of the prompt in characters.")
    word_count: int = Field(
        default=0, description="Number of whitespace-delimited words."
    )
    is_question: bool = Field(
        ..., description="Whether the prompt is formulated as a question."
    )
    has_code_indicators: bool = Field(
        ...,
        description="Whether the prompt contains code indicators (e.g. backticks, def, class).",
    )
    code_token_fraction: float = Field(
        default=0.0,
        description="Fraction of tokens that look like code keywords.",
    )
    requires_json: bool = Field(
        default=False,
        description="Whether the prompt explicitly asks for JSON output.",
    )
    complexity_score: float = Field(
        default=0.0,
        description="Heuristic complexity score in [0, 1].",
    )
    category: str = Field(
        default="general",
        description="Inferred prompt category (e.g. 'math', 'code', 'factual_qa', 'creative', 'general').",
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Semantic embedding vector of the prompt."
    )


class ConfidenceBreakdown(BaseModel):
    """Per-signal breakdown of the confidence estimation."""

    overall: float = Field(
        ..., description="Final aggregated confidence score in [0, 1]."
    )
    length_ok: bool = Field(
        default=True, description="Whether response length is adequate."
    )
    json_valid: Optional[bool] = Field(
        default=None,
        description="Whether JSON output is valid (None if not applicable).",
    )
    code_syntax_ok: Optional[bool] = Field(
        default=None,
        description="Whether code output parses syntactically (None if not applicable).",
    )
    completeness_ok: bool = Field(
        default=True,
        description="Whether the response meets completeness heuristics.",
    )
    not_repetitive: bool = Field(
        default=True, description="Whether the response is not degenerate/repetitive."
    )
    not_refusal: bool = Field(
        default=True, description="Whether the response is not a refusal/hedge."
    )
    failure_reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable list of why confidence was lowered.",
    )


class HistoryRecord(BaseModel):
    """A record of a single inference episode stored in the history database."""

    query_id: str = Field(..., description="Unique ID for the query.")
    prompt_hash: str = Field(
        default="",
        description="SHA-256 hash of the prompt for quick deduplication.",
    )
    prompt: str = Field(..., description="The original prompt string.")
    features: PromptFeatures = Field(
        ..., description="Extracted features of the prompt."
    )
    context_vector: Optional[List[float]] = Field(
        default=None,
        description="Normalized context vector fed to the bandit policy.",
    )
    chosen_model: str = Field(
        ...,
        description="The model selected (and evaluated) for this attempt.",
    )
    latency_ms: float = Field(
        ..., description="Latency of the model inference in milliseconds."
    )
    token_usage: int = Field(default=0, description="Number of tokens generated/used.")
    confidence_score: float = Field(
        ..., description="Estimated confidence score of the answer."
    )
    confidence_breakdown: Optional[ConfidenceBreakdown] = Field(
        default=None,
        description="Detailed per-signal confidence breakdown.",
    )
    success: bool = Field(..., description="Whether the answer was deemed successful.")
    escalated: bool = Field(
        default=False,
        description="Whether this request was escalated from a cheaper model.",
    )
    escalation_count: int = Field(
        default=0,
        description="Number of escalation steps taken before final answer.",
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Primary reason for failure (e.g. 'json_parse_error', 'too_short').",
    )
    reward: float = Field(
        default=0.0,
        description="Computed reward signal (success − λ·cost).",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the record was created.",
    )

    @staticmethod
    def compute_prompt_hash(prompt: str) -> str:
        """Compute a SHA-256 hash of the prompt for quick lookups."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class HistoryAggregateStats(BaseModel):
    """Aggregated statistics from similar past queries, per model."""

    model_success_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Average success rate per model on similar queries.",
    )
    model_avg_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Average confidence per model on similar queries.",
    )
    model_escalation_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of similar queries where this model triggered escalation.",
    )
    common_failure_reasons: List[str] = Field(
        default_factory=list,
        description="Most common failure reasons among similar queries.",
    )
    total_similar: int = Field(
        default=0,
        description="Total number of similar history records found.",
    )
