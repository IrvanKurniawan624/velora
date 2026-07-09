from app.core.constants import Category
from app.core.logger import get_logger

logger = get_logger("router")


class RouterService:
    def __init__(self) -> None:
        pass

    def calculate_complexity_score(self, features: dict) -> float:
        """Calculate a composite complexity score for the prompt."""
        len_score = min(1.0, features.get("word_count", 0) / 220.0)
        inst_score = min(1.0, features.get("instruction_count", 0) / 3.0)
        constraint_score = min(1.0, features.get("constraint_count", 0) / 2.0)
        code_score = min(1.0, features.get("code_tokens", 0) / 4.0)
        math_score = min(1.0, features.get("math_score", 0) / 1.5)
        logical_score = min(1.0, features.get("conditionals", 0) / 2.0)
        ambiguity_score = min(1.0, features.get("ambiguities", 0) / 1.0)

        # Weighted composite score (normalized to sum to 1.0)
        score = (
            0.10 * len_score
            + 0.10 * inst_score
            + 0.15 * constraint_score
            + 0.20 * code_score
            + 0.20 * math_score
            + 0.15 * logical_score
            + 0.10 * ambiguity_score
        )
        return score

    def route_advanced(self, category: Category, features: dict) -> str:
        """Route query using extracted features and complexity scoring."""
        complexity = self.calculate_complexity_score(features)
        logger.info(
            f"Calculated complexity score: {complexity:.3f} for category {category.value}"
        )

        local_categories = {
            Category.SENTIMENT,
            Category.NER,
            Category.FACTUAL,
            Category.SUMMARIZE,
        }

        # 1. Heavy categories: local only if extremely trivial, otherwise Fireworks
        if category not in local_categories:
            if complexity < 0.18 and features.get("word_count", 0) < 15:
                logger.info(
                    f"Routing trivial heavy category '{category.value}' to LOCAL client."
                )
                return "local"
            logger.info(
                f"Routing heavy category '{category.value}' to FIREWORKS client."
            )
            return "fireworks"

        # 2. Light categories: escalate if any threshold is exceeded
        # Direct threshold checks
        if features.get("word_count", 0) > 220:
            logger.info("Prompt exceeds length threshold. Escalating to FIREWORKS.")
            return "fireworks"
        if features.get("constraint_count", 0) >= 2:
            logger.info("Constraint count threshold exceeded. Escalating to FIREWORKS.")
            return "fireworks"
        if features.get("instruction_count", 0) >= 3:
            logger.info(
                "Instruction count threshold exceeded. Escalating to FIREWORKS."
            )
            return "fireworks"
        if features.get("math_score", 0) > 1.5:
            logger.info("Math score threshold exceeded. Escalating to FIREWORKS.")
            return "fireworks"
        if features.get("conditionals", 0) >= 2:
            logger.info(
                "Conditional score threshold exceeded. Escalating to FIREWORKS."
            )
            return "fireworks"
        if features.get("code_tokens", 0) >= 4:
            logger.info("Code tokens threshold exceeded. Escalating to FIREWORKS.")
            return "fireworks"
        if (
            features.get("entity_density", 0.0) > 0.15
            and features.get("word_count", 0) > 100
        ):
            logger.info("High entity density in long prompt. Escalating to FIREWORKS.")
            return "fireworks"
        if features.get("ambiguities", 0) >= 1:
            logger.info("Ambiguity indicator detected. Escalating to FIREWORKS.")
            return "fireworks"

        # Composite score check
        if complexity >= 0.5:
            logger.info(
                f"Complexity score {complexity:.3f} exceeds threshold. Escalating to FIREWORKS."
            )
            return "fireworks"

        logger.info(f"Routing category '{category.value}' to LOCAL client.")
        return "local"

    def route(self, category: Category, features: dict | None = None) -> str:
        """Determine if a category goes to local or fireworks client, optionally using features."""
        if features is None:
            local_categories = {
                Category.SENTIMENT,
                Category.NER,
                Category.FACTUAL,
                Category.SUMMARIZE,
            }
            if category in local_categories:
                logger.info(
                    f"Routing category '{category.value}' to LOCAL client (fallback)."
                )
                return "local"
            else:
                logger.info(
                    f"Routing category '{category.value}' to FIREWORKS client (fallback)."
                )
                return "fireworks"

        return self.route_advanced(category, features)
