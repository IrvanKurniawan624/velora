from app.core.constants import Category
from app.core.logger import get_logger

logger = get_logger("router")


class RouterService:
    def __init__(self) -> None:
        pass

    def route(self, category: Category) -> str:
        """Determine if a category goes to local or fireworks client."""
        local_categories = {
            Category.SENTIMENT,
            Category.NER,
            Category.FACTUAL,
            Category.SUMMARIZE,
        }

        if category in local_categories:
            logger.info(f"Routing category '{category.value}' to LOCAL client.")
            return "local"
        else:
            logger.info(f"Routing category '{category.value}' to FIREWORKS client.")
            return "fireworks"
