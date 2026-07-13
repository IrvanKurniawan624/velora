"""Self-check helpers: small-model failure pattern detection.

Only detects internal degenerate patterns (repetitive loops, non-Latin script
drift). The public gate lives in app.services.gate. These checks never raise.
"""


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
