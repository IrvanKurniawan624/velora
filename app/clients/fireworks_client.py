from openai import AsyncOpenAI

from app.clients.base_client import BaseClient
from app.config import Settings
from app.core.logger import get_logger
from app.models import ChatRequest, ChatResponse

logger = get_logger("fireworks_client")


class FireworksClient(BaseClient):
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.settings = Settings()
        resolved_key = api_key or (
            self.settings.fireworks_api_key.get_secret_value()
            if self.settings.fireworks_api_key
            else ""
        )
        resolved_base = base_url or self.settings.fireworks_base_url

        self.mock_mode = False
        if not resolved_key:
            logger.warning(
                "fireworks_api_key is empty or missing! FireworksClient will run in mock mode."
            )
            self.mock_mode = True
            self.client = None
        else:
            if not resolved_base:
                raise ValueError(
                    "fireworks_base_url is not set! It must be injected by the grading environment or defined in configuration."
                )
            self.client = AsyncOpenAI(api_key=resolved_key, base_url=resolved_base)

    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        if self.mock_mode or not self.client:
            logger.info("Fireworks client Mock response (no api_key).")
            return ChatResponse(
                content="[Fireworks API mock response] This is a mock response because the Fireworks API key was not available.",
                model=request.model or "mock-fireworks-model",
                finish_reason="stop",
            )

        # Determine the model to use based on ALLOWED_MODELS from runtime
        allowed = self.settings.allowed_models_list
        model_to_use = request.model

        if allowed:
            if request.model not in allowed:
                model_to_use = allowed[0]
                logger.info(
                    f"Requested model '{request.model}' not in allowed list. Falling back to allowed model: '{model_to_use}'"
                )
        else:
            logger.warning(
                "No ALLOWED_MODELS configured. Proceeding with requested model."
            )

        if not model_to_use:
            raise ValueError("No model specified and allowed_models list is empty.")

        logger.info(
            f"Sending chat completion to Fireworks using model: '{model_to_use}'"
        )
        response = await self.client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            finish_reason=choice.finish_reason,
            usage=response.usage.model_dump() if response.usage else None,
        )
