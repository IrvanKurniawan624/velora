import math
import logging
from app.schemas import ChatResponse

logger = logging.getLogger(__name__)

class LocalClient:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llm = None

    def load_model(self) -> None:
        """
        Lazily load the Gemma 2B model using llama-cpp-python.
        n_ctx=2048 matches expected context length.
        n_threads=2 matches container vCPU budget (2 vCPUs).
        logits_all=True is required to enable logprobs on all tokens.
        """
        if self.llm is not None:
            return

        try:
            from llama_cpp import Llama
            logger.info(f"Loading local model from {self.model_path}...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=1024,       # Reduced from 2048 — sufficient for all benchmark tasks
                n_threads=2,      # Match container vCPU budget
                n_batch=128,      # Smaller batch = lower memory pressure on 4 GB limit
                logits_all=True,  # Required for logprobs-based confidence scoring
                verbose=False
            )
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load local model (llama-cpp-python not available or model missing): {e}. Falling back to remote-only.")
            self.llm = "fallback"

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.1) -> ChatResponse:
        """
        Runs local inference using Gemma 2B and extracts token logprobs to calculate confidence.
        If llama-cpp-python is not installed, it returns a 0.0 confidence fallback response.
        """
        self.load_model()
        if self.llm == "fallback":
            return ChatResponse(
                content="[Fallback: Local model inference not available]",
                model="gemma-2-2b-fallback",
                confidence=0.0,
                remote_tokens_used=0
            )


        # Construct messages according to Chat Completion format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call create_chat_completion with logprobs on output tokens only
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                top_logprobs=1
            )
        except Exception as e:
            logger.error(f"Error during local model inference: {e}")
            raise

        content = response["choices"][0]["message"]["content"] or ""
        
        # Extract logprobs and calculate average confidence
        logprobs_data = response["choices"][0].get("logprobs", {})
        content_logprobs = logprobs_data.get("content", []) if logprobs_data else []
        
        avg_confidence = 1.0
        if content_logprobs:
            probs = []
            for item in content_logprobs:
                logprob = item.get("logprob")
                if logprob is not None:
                    # convert logprob (base e) to probability
                    probs.append(math.exp(logprob))
            if probs:
                avg_confidence = sum(probs) / len(probs)
            else:
                avg_confidence = 0.0
        else:
            avg_confidence = 0.0

        return ChatResponse(
            content=content,
            model=self.model_path,
            confidence=avg_confidence,
            remote_tokens_used=0  # Local inference is free
        )
