"""Feature extraction module for the Velora Adaptive Routing Engine.

Computes lightweight query features and a normalized context vector
used by the LinUCB routing policy.
"""

import re
from typing import List

from app.router.embedding_service import get_embedding
from app.router.schemas import PromptFeatures

# Common code keywords across Python, JS, Java, etc.
_CODE_KEYWORDS = frozenset(
    [
        "def",
        "class",
        "import",
        "return",
        "function",
        "const",
        "let",
        "var",
        "if",
        "else",
        "for",
        "while",
        "try",
        "except",
        "catch",
        "async",
        "await",
        "print",
        "console",
        "void",
        "int",
        "float",
        "string",
        "list",
        "dict",
        "self",
        "this",
        "public",
        "private",
    ]
)

_QUESTION_STARTERS = frozenset(
    [
        "what",
        "how",
        "why",
        "who",
        "when",
        "where",
        "is",
        "are",
        "can",
        "do",
        "does",
        "could",
        "would",
        "should",
        "explain",
        "describe",
        "compare",
        "list",
    ]
)


class FeatureExtractor:
    """Extracts prompt features and builds the context vector for routing."""

    @staticmethod
    def extract(prompt: str) -> PromptFeatures:
        """Extract lightweight features from a prompt."""
        prompt_length = len(prompt)
        words = prompt.split()
        word_count = len(words)
        lower_words = [w.lower().strip(".,;:!?()[]{}\"'") for w in words]

        # Question detection
        is_question = "?" in prompt or (
            len(lower_words) > 0 and lower_words[0] in _QUESTION_STARTERS
        )

        # Code detection
        has_code_indicators = (
            "```" in prompt
            or "def " in prompt
            or "class " in prompt
            or "function " in prompt
            or "import " in prompt
        )

        # Code token fraction
        code_tokens = sum(1 for w in lower_words if w in _CODE_KEYWORDS)
        code_token_fraction = code_tokens / max(word_count, 1)

        # JSON requirement detection
        requires_json = bool(re.search(r"\bjson\b", prompt, re.IGNORECASE))

        # Complexity heuristic: combines length, code presence, and structural cues
        complexity_score = FeatureExtractor._compute_complexity(
            prompt, word_count, code_token_fraction, has_code_indicators
        )

        # Prompt category (for per-category performance stats)
        category = FeatureExtractor._classify_category(
            prompt, has_code_indicators, code_token_fraction
        )

        # Semantic embedding
        embedding = get_embedding(prompt)

        return PromptFeatures(
            prompt_length=prompt_length,
            word_count=word_count,
            is_question=is_question,
            has_code_indicators=has_code_indicators,
            code_token_fraction=round(code_token_fraction, 4),
            requires_json=requires_json,
            complexity_score=round(complexity_score, 4),
            category=category,
            embedding=embedding,
        )

    @staticmethod
    def _compute_complexity(
        prompt: str,
        word_count: int,
        code_fraction: float,
        has_code: bool,
    ) -> float:
        """Return a heuristic complexity score in [0, 1]."""
        score = 0.0

        # Length contribution (longer prompts are often harder)
        if word_count > 200:
            score += 0.3
        elif word_count > 50:
            score += 0.15

        # Code contribution
        if has_code:
            score += 0.2
        score += min(code_fraction * 2.0, 0.2)

        # Multi-step / reasoning cues
        reasoning_cues = [
            "step by step",
            "explain",
            "analyze",
            "compare",
            "derive",
            "prove",
            "calculate",
            "implement",
            "algorithm",
            "optimize",
        ]
        for cue in reasoning_cues:
            if cue in prompt.lower():
                score += 0.1
                break

        # Nested structure cues (lists, bullets, multiple questions)
        if prompt.count("?") > 1:
            score += 0.1
        if prompt.count("\n") > 3:
            score += 0.05

        return min(score, 1.0)

    @staticmethod
    def _classify_category(
        prompt: str,
        has_code: bool,
        code_fraction: float,
    ) -> str:
        """Classify the prompt into a high-level category for per-category stats."""
        lower = prompt.lower()

        # Math / calculation
        math_cues = [
            "calculate",
            "compute",
            "solve",
            "equation",
            "integral",
            "derivative",
            "proof",
            "theorem",
            "algebra",
            "probability",
            "statistics",
            "sum of",
            "what is ",
            "2+2",
            "multiply",
            "divide",
        ]
        if any(c in lower for c in math_cues):
            return "math"

        # Code / programming
        if has_code or code_fraction > 0.15:
            return "code"

        # Creative writing
        creative_cues = [
            "write a story",
            "poem",
            "creative",
            "fiction",
            "imagine",
            "compose",
            "narrative",
        ]
        if any(c in lower for c in creative_cues):
            return "creative"

        # Factual QA
        if "?" in prompt:
            return "factual_qa"

        return "general"

    @staticmethod
    def build_context_vector(
        features: PromptFeatures,
        hist_success_3b: float = 0.0,
        hist_success_7b: float = 0.0,
        hist_success_fw: float = 0.0,
    ) -> List[float]:
        """
        Build a normalized context vector x ∈ [0,1]^d for the LinUCB policy.

        Dimensions:
          0: normalized prompt length (clipped to 2000 chars)
          1: is_question (0/1)
          2: has_code_indicators (0/1)
          3: code_token_fraction [0,1]
          4: requires_json (0/1)
          5: complexity_score [0,1]
          6: hist_success_3b [0,1]
          7: hist_success_7b [0,1]
          8: hist_success_fw [0,1]
          9: bias term (always 1.0)
        """
        return [
            min(features.prompt_length / 2000.0, 1.0),
            float(features.is_question),
            float(features.has_code_indicators),
            features.code_token_fraction,
            float(features.requires_json),
            features.complexity_score,
            hist_success_3b,
            hist_success_7b,
            hist_success_fw,
            1.0,  # bias
        ]
