from app.core.constants import Category
from app.core.logger import get_logger

logger = get_logger("self_check")


class SelfCheckService:
    def __init__(self) -> None:
        pass

    def check(self, prompt: str, response_content: str, category: Category) -> bool:
        """Check if local model response is acceptable, or if we need Fireworks escalation."""
        cleaned = response_content.strip()

        # 1. Null / too short responses
        if not cleaned or len(cleaned) < 2:
            logger.warning("Self-check failed: Empty or extremely short response.")
            return False

        refusals = [
            "i cannot answer",
            "i am unable to",
            "i don't know",
            "i do not know",
            "too complex",
            "i can't answer",
            "cannot solve",
            "unable to fulfill",
            "sorry, but i",
            "sorry, as an ai",
            "i do not have information",
            "i do not have access",
            "as a language model",
            "i am not able to",
        ]
        response_lower = cleaned.lower()
        if any(phrase in response_lower for phrase in refusals):
            logger.warning(
                "Self-check failed: Response contains refusal/inability phrases."
            )
            return False

        # 3. General repetition loop check (if word count is substantial)
        words_cleaned = cleaned.split()
        if len(words_cleaned) > 15:
            unique_words = set(words_cleaned)
            if len(unique_words) / len(words_cleaned) < 0.35:
                logger.warning("Self-check failed: Repetition loop detected in output.")
                return False

        # 4. Category-specific semantic structural checks
        if category == Category.SENTIMENT:
            # Check if sentiment keywords or JSON structure is present
            sentiment_labels = ["positive", "negative", "neutral"]
            has_label = any(lbl in response_lower for lbl in sentiment_labels)
            if not has_label:
                logger.warning(
                    "Self-check failed: Sentiment response has no clear sentiment label."
                )
                return False

        elif category == Category.MATH:
            import re

            # Check that there is at least one digit in the response
            if not re.search(r"\d", cleaned):
                logger.warning(
                    "Self-check failed: Math task output contains no digits."
                )
                return False

        elif category == Category.NER:
            # NER outputs must not say "no entities found" or "none" if there are entities in prompt
            if "no entities" in response_lower or response_lower in {
                "none",
                "n/a",
                "empty",
            }:
                capitalized = [w for w in prompt.split() if w and w[0].isupper()]
                if len(capitalized) > 1:
                    logger.warning(
                        "Self-check failed: NER response returned 'none' but prompt has capitalized words."
                    )
                    return False

        elif category == Category.SUMMARIZE:
            # Summary shouldn't be longer than the prompt (if prompt is substantial)
            p_words = len(prompt.split())
            c_words = len(cleaned.split())
            if p_words > 30 and c_words >= p_words:
                logger.warning(
                    "Self-check failed: Summary is longer than or equal to the original prompt."
                )
                return False

        elif category in {Category.CODE_GEN, Category.CODE_DEBUG}:
            # A code output should have code block indicators or keywords if code is requested
            code_indicators = [
                "def ",
                "class ",
                "import ",
                "function",
                "fn ",
                "let ",
                "const ",
                "var ",
                "void",
                "public",
                "private",
                "return",
                "select ",
                "from ",
                "where ",
            ]
            has_code = (
                any(ind in cleaned for ind in code_indicators) or "```" in cleaned
            )
            if not has_code:
                logger.warning(
                    "Self-check failed: Code task output contains no code structural markers."
                )
                return False

        return True
