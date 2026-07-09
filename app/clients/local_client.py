import os
import asyncio
from app.clients.base_client import BaseClient
from app.models import ChatRequest, ChatResponse
from app.config import Settings
from app.core.logger import get_logger

logger = get_logger("local_client")

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    logger.warning("llama-cpp-python not installed. Local client will run in mock mode.")


class LocalClient(BaseClient):
    def __init__(self, model_path: str | None = None) -> None:
        settings = Settings()
        self.model_path = model_path or settings.local_model_path
        self.llm = None

        if HAS_LLAMA_CPP:
            if not os.path.exists(self.model_path):
                logger.warning(f"Local model GGUF file not found at {self.model_path}. Expect failures if used.")
            else:
                logger.info(f"Loading local model from {self.model_path}...")
                try:
                    # Load model once at startup to avoid overhead per task
                    self.llm = Llama(
                        model_path=self.model_path,
                        n_ctx=2048,
                        n_threads=os.cpu_count() or 4,
                        verbose=False,
                    )
                    logger.info("Local model successfully loaded.")
                except Exception as e:
                    logger.error(f"Failed to load local model: {e}")
        else:
            logger.warning("Running LocalClient in mock mode (llama-cpp-python missing).")

    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        if not HAS_LLAMA_CPP or not self.llm:
            # Fallback mock for local development/testing without GGUF compiled
            logger.info("Local client Mock response (no llama-cpp or model file).")
            return ChatResponse(
                content="[Local model mock response] This is a mock response because llama-cpp or the GGUF model file was not available.",
                model="mock-local-model",
                finish_reason="stop",
            )

        formatted_messages = []
        for msg in request.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        # Run llama-cpp chat completion in executor to prevent blocking the asyncio loop.
        loop = asyncio.get_running_loop()

        def _run_inference():
            return self.llm.create_chat_completion(
                messages=formatted_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

        try:
            response_data = await loop.run_in_executor(None, _run_inference)
            choice = response_data["choices"][0]
            content = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason", "stop")
            usage = response_data.get("usage", None)

            return ChatResponse(
                content=content,
                model=request.model or "local-gguf",
                finish_reason=finish_reason,
                usage=usage
            )
        except Exception as e:
            logger.error(f"Inference error in LocalClient: {e}")
            raise
