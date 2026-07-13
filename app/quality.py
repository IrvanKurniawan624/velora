"""Answer sanity gate: cheap, deterministic checks that flag clearly-bad local
outputs so the orchestrator can re-verify or escalate them.

Only catches degenerate answers - empty, refusals, non-Latin drift, repetitive
loops. It never raises and only ever *lowers* confidence on bad outputs, so
good answers are untouched. Drawn from the retired app/'s gate/self-check
services, stripped of the non-compliant speculative-routing behaviour (no
forced category escalation - the solvers' own verification handles that).
"""
import re

_REFUSALS = (
    "i don't know", "i do not know", "i cannot", "i can't", "as an ai",
    "i'm not sure", "i am not sure", "unable to", "apologize", "i apologize",
    "cannot fulfill", "cannot comply", "against my guidelines",
)


def _is_degenerate(answer: str) -> bool:
    """Repetitive 4-gram loop - a common small-model failure mode."""
    words = answer.lower().split()
    if len(words) < 12:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return any(grams.count(g) >= 4 for g in set(grams))


def _non_latin_ratio(text: str) -> float:
    """Small local models occasionally drift into non-Latin scripts."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 0x24F) / len(text)


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
