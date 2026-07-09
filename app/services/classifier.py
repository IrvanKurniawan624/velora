import re
from app.clients.base_client import BaseClient
from app.core.constants import Category
from app.models import ChatRequest, Message
from app.core.logger import get_logger

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

    def _rule_based_classify(self, query: str) -> Category | None:
        q = query.lower()

        # Rule-based detection
        if any(
            w in q
            for w in [
                "sentiment",
                "sentimen",
                "emotion",
                "emosi",
                "positive",
                "negative",
                "neutral",
                "positif",
                "negatif",
                "netral",
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
                "ekstrak",
                "entitas",
                "temukan nama",
                "organisasi",
                "lokasi",
            ]
        ):
            return Category.NER

        if any(
            w in q
            for w in [
                "summarize",
                "summary",
                "ringkas",
                "ringkasan",
                "rangkum",
                "rangkuman",
                "singkat",
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
                "perbaiki kode",
                "salah di kode",
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
                "tulis kode",
                "buat fungsi",
                "buat program",
            ]
        ):
            return Category.CODE_GEN

        if any(
            w in q
            for w in [
                "riddle",
                "logic puzzle",
                "teka-teki",
                "logika",
                "deduksi",
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
                "matematika",
                "hitunglah",
                "persamaan",
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
