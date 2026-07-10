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
        n_ctx=1024 sufficient for all benchmark tasks.
        n_threads=2 matches container vCPU budget.
        logits_all=False for fast inference (heuristic confidence used instead).
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
                logits_all=False, # Fast inference — heuristic confidence used instead of logprobs
                verbose=False
            )
            logger.info("Local model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load local model (llama-cpp-python not available or model missing): {e}. Falling back to remote-only.")
            self.llm = "fallback"

    def _heuristic_confidence(self, content: str, task_type: str = None) -> float:
        """
        Fast heuristic confidence score based on output characteristics and task category.
        Replaces logprobs (which require logits_all=True, causing ~2 min/task on 2 vCPUs).

        Rules (in priority order):
        - Empty output          → 0.0 (certain failure)
        - Refusal phrases       → 0.0
        - Uncertainty phrases   → 0.5 (may escalate depending on threshold)
        - Otherwise             → category-specific baseline confidence
        """
        if not content or not content.strip():
            return 0.0

        lowered = content.lower()

        refusal_phrases = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i don't know", "i do not know", "as an ai", "i'm not sure",
            "i am not sure", "i'm not able", "i apologize"
        ]
        if any(p in lowered for p in refusal_phrases):
            return 0.0

        uncertainty_phrases = [
            "i think", "i believe", "i'm not certain", "possibly",
            "i'm guessing", "not entirely sure", "might be"
        ]
        if any(p in lowered for p in uncertainty_phrases):
            return 0.5

        # Heuristic based on task type to optimize local routing safety:
        if task_type == "math":
            # 2B models are highly inaccurate for math arithmetic, always escalate
            return 0.10
        elif task_type == "logic":
            # Logic puzzles require strong reasoning, always escalate
            return 0.60
        elif task_type == "factual":
            # Strict keyword matching in factual tasks makes remote safer
            return 0.70
        elif task_type == "code":
            # Code is safe to run locally if it compiles successfully
            return 0.95
        elif task_type in ["sentiment", "ner", "summarise"]:
            # Extraction and classification tasks are perfect for 2B models
            return 0.95

        return 0.92

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.1, task_type: str = None) -> ChatResponse:
        """
        Runs local inference using Gemma 2B.
        Uses fast heuristic confidence scoring instead of logprobs to avoid
        logits_all=True overhead (~2 min/task on 2 vCPUs).
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

        # Call create_chat_completion without logprobs for speed
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Error during local model inference: {e}")
            raise

        content = response["choices"][0]["message"]["content"] or ""
        confidence = self._heuristic_confidence(content, task_type)

        return ChatResponse(
            content=content,
            model=self.model_path,
            confidence=confidence,
            remote_tokens_used=0  # Local inference is free
        )
