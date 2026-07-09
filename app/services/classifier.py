import re

from app.clients.base_client import BaseClient
from app.core.constants import Category
from app.core.logger import get_logger
from app.models import ChatRequest, Message

logger = get_logger("classifier")

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a query classifier. Classify the user's query into exactly one "
    "of the following categories:\n"
    "- factual: General knowledge or factual questions.\n"
    "- math: Mathematical problems, equations, or calculation reasoning.\n"
    "- sentiment: Requests to analyze sentiment or emotion of text.\n"
    "- logic: Logic puzzles, riddles, or deductive reasoning.\n"
    "- summarize: Requests to summarize, shorten, or condense text.\n"
    "- ner: Named Entity Recognition or extracting entities like names, locations, dates.\n"
    "- code_debug: Debugging existing code, fixing bugs, or explaining errors.\n"
    "- code_gen: Writing new code, generating scripts, or implementing functions.\n\n"
    "Respond with ONLY the category name (factual, math, sentiment, logic, summarize, ner, code_debug, code_gen). "
    "No punctuation, no explanation, no formatting."
)


class ClassifierService:
    def __init__(self, client: BaseClient, model: str = "local") -> None:
        self.client = client
        self.model = model

    def extract_features(self, query: str) -> dict:
        """Extract linguistic features and complexity markers from the query."""
        q_lower = query.lower()
        words = query.split()
        word_count = len(words)

        # 1. Math complexity (operators, variables, numbers)
        math_operators = len(re.findall(r"[\+\-\*\/\=\<\>\^]|\\times|\\div", query))
        digits = len(re.findall(r"\d+", query))
        math_score = math_operators + (0.5 * digits)

        # 2. Code complexity (programming structural tokens)
        code_tokens = len(
            re.findall(
                r"\b(def|class|import|from|function|fn|let|const|var|return|public|private|void|int|float|double|char|struct)\b|[{};]",
                query,
            )
        )

        # 3. Instruction & Constraint count
        instructions = len(
            re.findall(
                r"\b(write|create|implement|solve|calculate|evaluate|find|debug|fix|explain|summarize|condense|extract|identify|list|analyze|determine)\b",
                q_lower,
            )
        )
        constraints = len(
            re.findall(
                r"\b(must|only|not|without|except|exactly|limit|restrict|no explanation|in json|json format|words? limit|sentences? limit|bullet points?)\b",
                q_lower,
            )
        )

        # 4. Entity density (capitalized words excluding sentence starts, numbers, dates)
        capitalized = len(
            [
                w
                for idx, w in enumerate(words)
                if w
                and w[0].isupper()
                and idx > 0
                and not words[idx - 1].endswith((".", "?", "!"))
            ]
        )
        special_entities = len(
            re.findall(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b",
                q_lower,
            )
        )
        entity_density = (capitalized + special_entities) / max(1, word_count)

        # 5. Logical dependency & conditionals
        conditionals = len(
            re.findall(
                r"\b(if|else|then|when|unless|otherwise|because|therefore|consequently|implies|suppose|assume|assumed|given)\b",
                q_lower,
            )
        )

        # 6. Ambiguity Indicators
        ambiguities = len(
            re.findall(
                r"\b(maybe|perhaps|could|might|sometimes|unclear|ambiguous|depends|either|or|sarcastic|sarcasm|ironic|irony)\b",
                q_lower,
            )
        )

        return {
            "word_count": word_count,
            "math_score": math_score,
            "code_tokens": code_tokens,
            "instruction_count": instructions,
            "constraint_count": constraints,
            "entity_density": entity_density,
            "conditionals": conditionals,
            "ambiguities": ambiguities,
        }

    def _rule_based_classify(self, query: str) -> Category | None:
        q = query.lower()

        # Rule-based detection
        if any(
            w in q
            for w in [
                "sentiment",
                "emotion",
                "positive",
                "negative",
                "neutral",
            ]
        ):
            return Category.SENTIMENT

        if any(
            w in q
            for w in [
                "ner",
                "named entity",
                "entities",
                "extract",
                "entity",
                "find name",
                "organization",
                "location",
            ]
        ):
            return Category.NER

        if any(
            w in q
            for w in [
                "summarize",
                "summary",
                "condense",
                "shorten",
            ]
        ):
            return Category.SUMMARIZE

        if any(
            w in q
            for w in [
                "debug",
                "error in",
                "fix code",
                "bug in",
                "code error",
                "runtimeerror",
                "syntaxerror",
                "exception",
            ]
        ):
            return Category.CODE_DEBUG

        if any(
            w in q
            for w in [
                "write a python",
                "write a code",
                "write a script",
                "generate code",
                "implement a function",
                "write code",
                "create function",
                "make program",
            ]
        ):
            return Category.CODE_GEN

        if any(
            w in q
            for w in [
                "riddle",
                "logic puzzle",
                "logic",
                "deduction",
                "deductive reasoning",
            ]
        ):
            return Category.LOGIC

        if any(
            w in q
            for w in [
                "math",
                "calculate",
                "solve the equation",
                "solve",
                "integral",
                "derivative",
                "mathematics",
                "equation",
            ]
        ):
            return Category.MATH

        return None

    async def classify(self, user_message: str) -> Category:
        # 1. Try rule-based first
        category = self._rule_based_classify(user_message)
        if category:
            logger.info(f"Rule-based classification matched: {category.value}")
            return category

        # 2. Fallback to LLM
        logger.info("Rule-based classification did not match. Falling back to LLM...")
        request = ChatRequest(
            model=self.model,
            messages=[
                Message(role="system", content=CLASSIFIER_SYSTEM_PROMPT),
                Message(role="user", content=user_message),
            ],
            temperature=0.0,
            max_tokens=10,
        )
        try:
            response = await self.client.chat_complete(request)
            label = response.content.strip().lower()

            # Clean punctuation
            label = re.sub(r"[^\w_-]", "", label)

            for cat in Category:
                if cat.value in label:
                    logger.info(f"LLM classification result: {cat.value}")
                    return cat

            logger.warning(
                f"LLM returned unrecognized category label: '{label}'. Defaulting to factual."
            )
            return Category.FACTUAL
        except Exception as e:
            logger.error(
                f"Error during LLM classification: {e}. Defaulting to factual."
            )
            return Category.FACTUAL
