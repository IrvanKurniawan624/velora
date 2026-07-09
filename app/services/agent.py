from app.clients.base_client import BaseClient
from app.core.constants import CATEGORY_PROMPT_FILES, PROMPTS_DIR, Category
from app.core.logger import get_logger
from app.models import ChatRequest, Message
from app.services.classifier import ClassifierService
from app.services.router import RouterService
from app.services.self_check import SelfCheckService

logger = get_logger("agent_service")


class AgentService:
    def __init__(self, local_client: BaseClient, fireworks_client: BaseClient) -> None:
        self.local_client = local_client
        self.fireworks_client = fireworks_client
        self.classifier = ClassifierService(client=self.local_client)
        self.router = RouterService()
        self.self_checker = SelfCheckService()

    def _load_prompt(self, category: Category) -> str:
        filename = CATEGORY_PROMPT_FILES.get(category, "factual.txt")
        path = PROMPTS_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        logger.warning(f"Prompt file not found at {path}. Using empty system prompt.")
        return ""

    async def run(self, user_message: str) -> str:
        # 1. Classify & Extract Features
        category = await self.classifier.classify(user_message)
        logger.info(f"Task classified as: {category.value}")

        features = self.classifier.extract_features(user_message)

        # 2. Route
        route = self.router.route(category, features)

        system_prompt = self._load_prompt(category)

        # 3. Process
        max_tokens_map = {
            Category.SENTIMENT: 30,
            Category.NER: 100,
            Category.FACTUAL: 200,
            Category.SUMMARIZE: 250,
            Category.MATH: 350,
            Category.LOGIC: 350,
            Category.CODE_DEBUG: 800,
            Category.CODE_GEN: 1024,
        }
        tokens_limit = max_tokens_map.get(category, 512)

        if route == "local":
            logger.info("Processing locally...")
            try:
                request = ChatRequest(
                    model="local-gguf",
                    messages=[
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=user_message),
                    ],
                    temperature=0.3,
                    max_tokens=tokens_limit,
                )
                response = await self.local_client.chat_complete(request)

                # Check response
                if self.self_checker.check(user_message, response.content, category):
                    logger.info("Local response passed validation.")
                    return response.content

                logger.warning(
                    "Local response failed validation. Escalating to Fireworks..."
                )
            except Exception as e:
                logger.error(f"Local inference failed: {e}. Escalating to Fireworks...")

            # Fallback to Fireworks
            route = "fireworks"

        if route == "fireworks":
            logger.info("Processing via Fireworks API...")
            request = ChatRequest(
                model="",  # Automatically resolved by FireworksClient
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_message),
                ],
                temperature=0.3,
                max_tokens=tokens_limit,
            )
            response = await self.fireworks_client.chat_complete(request)
            return response.content

        return ""
