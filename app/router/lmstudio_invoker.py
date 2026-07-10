"""Real model invoker for LM Studio and Fireworks API.

LM Studio exposes an OpenAI-compatible HTTP API at http://localhost:1234/v1
so we can use the `openai` Python library directly — zero extra dependencies.

Usage in AdaptiveRouter::

    from app.router.lmstudio_invoker import create_real_invoker
    invoker = create_real_invoker()
    router = AdaptiveRouter(model_invoker=invoker)
    answer = router.route_query("What is photosynthesis?")
"""

import logging
import time
from typing import Tuple

from openai import OpenAI, OpenAIError

from app.config import settings

logger = logging.getLogger(__name__)

# ── Model name mappings ───────────────────────────────────────────────────────
# The JIT router uses logical names (Qwen3-3B, Qwen3-7B, Fireworks).
# These are mapped to the actual model identifiers each server expects.
LMSTUDIO_MODEL_MAP: dict[str, str] = {
    "Qwen3-3B": settings.lmstudio_model_3b,
    "Qwen3-7B": settings.lmstudio_model_7b,
}


# ── LM Studio client (singleton) ─────────────────────────────────────────────
def _get_lmstudio_client() -> OpenAI:
    return OpenAI(
        base_url=settings.lmstudio_base_url,
        api_key="lm-studio",  # LM Studio ignores the key but openai lib needs one
    )


def _get_fireworks_client() -> OpenAI | None:
    key = settings.fireworks_api_key
    if key is None:
        return None
    return OpenAI(
        base_url=settings.fireworks_base_url,
        api_key=key.get_secret_value(),
    )


# ── Core invoke helpers ───────────────────────────────────────────────────────
def invoke_lmstudio(model_logical_name: str, prompt: str) -> Tuple[str, float, int]:
    """
    Call LM Studio local server for Qwen3-3B or Qwen3-7B.

    Returns
    -------
    (answer_text, latency_ms, token_count)
    """
    model_id = LMSTUDIO_MODEL_MAP.get(model_logical_name, model_logical_name)
    client = _get_lmstudio_client()

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else len(answer.split())
        logger.debug(
            "LMStudio [%s] latency=%.0fms tokens=%d", model_logical_name, latency_ms, tokens
        )
        return answer, latency_ms, tokens
    except OpenAIError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning("LMStudio call failed for %s: %s", model_logical_name, exc)
        return f"[ERROR: LM Studio unavailable – {exc}]", latency_ms, 0


def invoke_fireworks(model_logical_name: str, prompt: str) -> Tuple[str, float, int]:
    """
    Call the Fireworks remote API as the final escalation tier.

    Returns
    -------
    (answer_text, latency_ms, token_count)
    """
    client = _get_fireworks_client()
    if client is None:
        logger.warning("FIREWORKS_API_KEY not set – returning fallback answer.")
        return (
            "[Fireworks unavailable – FIREWORKS_API_KEY not configured]",
            0.0,
            0,
        )

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.fireworks_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else len(answer.split())
        logger.debug("Fireworks latency=%.0fms tokens=%d", latency_ms, tokens)
        return answer, latency_ms, tokens
    except OpenAIError as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning("Fireworks call failed: %s", exc)
        return f"[ERROR: Fireworks – {exc}]", latency_ms, 0


def create_real_invoker():
    """
    Factory that returns a ModelInvoker function wired to real backends:
    - Qwen3-3B / Qwen3-7B  → LM Studio (http://localhost:1234/v1)
    - Fireworks             → Fireworks remote API

    Usage::

        from app.router.lmstudio_invoker import create_real_invoker
        from app.router.decision_engine import AdaptiveRouter

        invoker = create_real_invoker()
        router = AdaptiveRouter(model_invoker=invoker)
        answer = router.route_query("Explain recursion.")
    """

    def _invoke(model_name: str, prompt: str) -> Tuple[str, float, int]:
        if model_name in ("Qwen3-3B", "Qwen3-7B"):
            return invoke_lmstudio(model_name, prompt)
        elif model_name == "Fireworks":
            return invoke_fireworks(model_name, prompt)
        else:
            raise ValueError(f"Unknown model: {model_name!r}")

    return _invoke


def check_lmstudio_connection() -> bool:
    """Quick health-check: returns True if LM Studio is reachable."""
    try:
        client = _get_lmstudio_client()
        models = client.models.list()
        available = [m.id for m in models.data]
        logger.info("LM Studio connected. Available models: %s", available)
        return True
    except Exception as exc:
        logger.error("LM Studio not reachable at %s: %s", settings.lmstudio_base_url, exc)
        return False
