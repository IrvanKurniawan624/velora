"""Zero-cost confidence gate: strict deterministic heuristics for local response evaluation.
"""
import ast
import re

_HEDGES = ("i don't know", "i do not know", "i cannot", "i can't", "as an ai",
           "i'm not sure", "i am not sure", "unable to", "apologize", "i apologize")


def _sentences(text):
    return [s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()]


def _degenerate(answer):
    words = answer.lower().split()
    if len(words) >= 12:
        grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
        if max(grams.count(g) for g in set(grams)) >= 4:
            return True
    return False


def _non_latin_ratio(text):
    """Answers must be in English; small local models occasionally drift into other scripts."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 0x24F) / len(text)


_SENT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _requested_sentences(prompt):
    m = re.search(r"\b(?:in|exactly)\s+(one|two|three|four|five|\d+)\s+sentences?\b",
                  prompt.lower())
    if not m:
        return None
    return _SENT_WORDS.get(m.group(1)) or int(m.group(1))


def score_local_response(prompt: str, answer: str, category: str) -> float:
    if not answer or not answer.strip():
        return 0.0
    a = answer.strip()
    low = a.lower()
    
    # 1. Repetitive loops check
    if _degenerate(a):
        return 0.0
        
    # 2. English language check
    if _non_latin_ratio(a) > 0.15:
        return 0.0
        
    # 3. Refusal check
    if any(h in low for h in _HEDGES):
        return 0.0

    # 4. Strict category-based validation
    if category in ("math", "logic", "code", "codegen", "debugging", "code_debugging", "code_generation"):
        # Local models are too weak for arithmetic, word math, logic puzzles, and coding.
        # Force escalation by returning 0.0.
        return 0.0

    # JSON structure check (for sentiment, ner, summaries expecting JSON)
    if "json" in prompt.lower():
        # Try loading as JSON
        try:
            # Strip markdown fences if any
            clean_json = re.sub(r"^```[a-zA-Z]*\n?", "", a)
            clean_json = re.sub(r"\n?```$", "", clean_json).strip()
            # Try to parse
            import json
            json.loads(clean_json)
        except Exception:
            # Not valid JSON, reject
            return 0.0

    if category == "sentiment":
        # Must contain one of the sentiment labels
        if not re.search(r"\b(positive|negative|neutral|mixed)\b", low):
            return 0.0
        return 0.95

    elif category == "ner":
        # Check if the output has standard separators like '-' or ':' or is a list
        if "-" not in a and ":" not in a and not (a.startswith("[") and a.endswith("]")):
            return 0.0
        return 0.95

    elif category in ("summarise", "summarisation", "summarize"):
        limit = _requested_sentences(prompt)
        if limit:
            sentences_count = len(_sentences(a))
            if sentences_count > limit:
                return 0.0
        if len(a) >= len(prompt):
            return 0.0
        return 0.90

    # Factual
    if len(a.split()) < 3:
        return 0.0
    # Refined check: escalate if multiple question marks exist OR if prompt is long (>150 chars) and contains joining/comparison keywords
    if prompt.count("?") > 1 or (len(prompt) > 150 and any(k in prompt.lower() for k in (" and ", " explain ", " compare ", " difference "))):
        # Long/multi-part factual questions are prone to local hallucinations, force escalation
        return 0.0
        
    return 0.85
