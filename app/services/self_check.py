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

        # 2. Standard LLM refusal or fallback phrases
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
            "tidak tahu",
            "tidak bisa menjawab",
            "tidak dapat menjawab",
        ]
        response_lower = cleaned.lower()
        if any(phrase in response_lower for phrase in refusals):
            logger.warning("Self-check failed: Response contains refusal/inability phrases.")
            return False

        # 3. Category-specific semantic structural checks
        if category == Category.SENTIMENT:
            # Check if sentiment keywords or JSON structure is present
            sentiment_labels = ["positive", "negative", "neutral", "positif", "negatif", "netral"]
            has_label = any(lbl in response_lower for lbl in sentiment_labels)
            if not has_label:
                logger.warning("Self-check failed: Sentiment response has no clear sentiment label.")
                return False

        return True
