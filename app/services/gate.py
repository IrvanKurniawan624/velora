"""Answer sanity gate: cheap, deterministic checks that flag clearly-bad local
outputs so the orchestrator can re-verify or escalate them.

Only catches degenerate answers - empty, refusals, non-Latin drift, repetitive
loops. It never raises and only ever *lowers* confidence on bad outputs, so
good answers are untouched. Drawn from the retired app/'s gate/self-check
services, stripped of the non-compliant speculative-routing behaviour.
"""
from .self_check import _is_degenerate, _non_latin_ratio

_REFUSALS = (
    "i don't know", "i do not know", "i cannot", "i can't", "as an ai",
    "i'm not sure", "i am not sure", "unable to", "apologize", "i apologize",
    "cannot fulfill", "cannot comply", "against my guidelines",
)


def sanity_check(prompt: str, answer: str, category: str) -> bool:
    """Return True if the answer is acceptable (non-degenerate)."""
    if not answer or not answer.strip():
        return False
    low = answer.strip().lower()
    if _is_degenerate(low):
        return False
    if _non_latin_ratio(answer) > 0.15:
        return False
    if any(h in low for h in _REFUSALS):
        return False
    return True
