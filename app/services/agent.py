import logging
import os
import re
from openai import OpenAI
from app.clients.local_client import LocalClient
from app.services.self_check import SelfCheckService
from app.schemas import ChatResponse
from app.config import Settings

logger = logging.getLogger(__name__)

class AgentService:
    def __init__(self, settings: Settings, self_check: SelfCheckService):
        self.settings = settings
        self.self_check = self_check
        
        # Initialize LocalClient with GGUF path
        self.local_client = LocalClient(model_path=settings.local_model_path)
        
        # Initialize OpenAI remote client
        api_key = settings.fireworks_api_key.get_secret_value() if settings.fireworks_api_key else None
        self.remote_client = OpenAI(
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY", "mock-key"),
            base_url=settings.fireworks_base_url
        )
        
        # In-memory cache to skip duplicate tasks
        self.cache = {}
        
        # Parse allowed remote models
        allowed_models_str = os.environ.get("ALLOWED_MODELS", settings.allowed_models)
        self.allowed_models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]

    def classify_task(self, prompt: str) -> str:
        """
        Classifies task based on prompt patterns to determine routing path and threshold.
        """
        prompt_lower = prompt.lower()
        
        # Code detection
        code_patterns = [
            r"def\s+\w+\(", r"import\s+", r"class\s+\w+", r"function\s*\(", 
            r"const\s+\w+\s*=", r"let\s+\w+\s*=", r"var\s+\w+\s*=", r"```python",
            r"code", r"debug", r"syntax", r"compile", r"python", r"function",
            r"implementation", r"algorithm", r"program", r"script"
        ]
        if any(re.search(pat, prompt_lower) for pat in code_patterns):
            return "code"
            
        # Math detection
        math_patterns = [
            r"\d+\s*[\+\-\*\/=]\s*\d+", r"percent", r"percentage", r"arithmetic",
            r"how many", r"solve for", r"ratio", r"probability"
        ]
        if any(re.search(pat, prompt_lower) for pat in math_patterns):
            return "math"
            
        # Logic detection
        logic_patterns = [
            r"logic puzzle", r"friends", r"owns the", r"sitting in a row", 
            r"if and only if", r"true or false", r"truth table", r"deduce"
        ]
        if any(re.search(pat, prompt_lower) for pat in logic_patterns):
            return "logic"
            
        # Sentiment detection
        if "sentiment" in prompt_lower or "classify the sentiment" in prompt_lower:
            return "sentiment"
            
        # NER detection
        if "extract all" in prompt_lower or "named entities" in prompt_lower or "ner" in prompt_lower:
            return "ner"
            
        # Summarisation detection
        if "summarize" in prompt_lower or "summarise" in prompt_lower or "summary" in prompt_lower:
            return "summarise"
            
        return "factual"

    def select_remote_model(self, task_type: str) -> str:
        """
        Dynamically selects the appropriate remote Fireworks model for the task.
        """
        is_complex = task_type in ["code", "logic"]
        target_substring = "kimi-k2p7-code" if is_complex else "minimax-m3"
        
        # Find exact model in allowed_models
        for model in self.allowed_models:
            if target_substring in model:
                return model
                
        # Alternate search
        alternate_substring = "minimax-m3" if is_complex else "kimi-k2p7-code"
        for model in self.allowed_models:
            if alternate_substring in model:
                return model
                
        # Default fallback
        if is_complex:
            return "accounts/fireworks/models/kimi-k2p7-code"
        else:
            return "accounts/fireworks/models/minimax-m3"

    def generate_remote(self, prompt: str, model: str, system_prompt: str = "") -> ChatResponse:
        """
        Query Fireworks API proxy.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.remote_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            tokens_used = usage.total_tokens if usage else 0
            
            return ChatResponse(
                content=content,
                model=model,
                confidence=1.0,
                remote_tokens_used=tokens_used
            )
        except Exception as e:
            logger.error(f"Fireworks API call failed: {e}")
            raise

    def clean_and_extract_content(self, text: str, task_type: str) -> str:
        """
        Extracts clean code or JSON block from markdown code fences and removes chat chatter.
        """
        text = text.strip()
        if not text:
            return text

        # If it's a code task
        if task_type == "code":
            match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text

        # If it's a JSON-producing task
        elif task_type in ["sentiment", "ner", "summarise"]:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            # Try finding the outer-most { ... }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text

        # For other tasks, just remove standard top/bottom backticks if any
        text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        return text.strip()

    def process_task(self, prompt: str) -> ChatResponse:
        """
        Orchestrates the dynamic Confidence-Cascaded Speculative Routing pipeline.
        """
        # Check for environment variable bypass (Baseline/No improvement mode)
        disable_routing = os.getenv("DISABLE_ROUTING", "False").lower() == "true"
        if disable_routing:
            logger.info("Routing disabled (Baseline Mode). Escalating directly to remote API.")
            task_type = self.classify_task(prompt)
            remote_model = self.select_remote_model(task_type)
            # Baseline does NOT compress the prompt or strip comments, send original prompt
            remote_response = self.generate_remote(prompt, model=remote_model)
            # Clean remote answer too
            remote_response.content = self.clean_and_extract_content(remote_response.content, task_type)
            return remote_response

        # 1. Local Cache Lookup
        if prompt in self.cache:
            logger.info("Cache hit! Returning cached answer.")
            return self.cache[prompt]
            
        # 2. Classify task
        task_type = self.classify_task(prompt)
        logger.info(f"Task classified as: {task_type}")
        
        # Compress prompt using our utility
        from app.utils import compress_prompt
        compressed_prompt = compress_prompt(prompt, task_type)
        logger.info(f"Prompt compressed. Original: {len(prompt)} chars -> Compressed: {len(compressed_prompt)} chars.")
        
        # 3. Define confidence thresholds
        # Code/logic tasks lowered to 0.90 to reduce unnecessary remote escalations
        confidence_threshold = 0.90 if task_type in ["code", "logic"] else 0.85
        
        # 4. Attempt local generation
        local_response = None
        try:
            # Code tasks need higher token budgets to avoid truncated function bodies
            max_tokens = 1024 if task_type == "code" else 512 if task_type == "logic" else 256
            local_response = self.local_client.generate(compressed_prompt, max_tokens=max_tokens, task_type=task_type)
            logger.info(f"Local client generated response. Confidence: {local_response.confidence:.2f}")
        except Exception as e:
            logger.warning(f"Local client generation failed: {e}")
            
        # 5. Local verification check
        if local_response and local_response.confidence > 0.0:
            # Extract and clean content first so validation runs on clean code/JSON
            local_response.content = self.clean_and_extract_content(local_response.content, task_type)
            local_ok = True
            
            # Refusal check
            if self.self_check.has_refusal(local_response.content):
                local_ok = False
                logger.warning("Local response contained refusal phrase.")
                
            # JSON format verification (for sentiment, ner, summarisation)
            elif task_type in ["sentiment", "ner", "summarise"] and "json" in prompt.lower():
                # Check JSON is parseable
                if not self.self_check.validate_json_structure(local_response.content):
                    local_ok = False
                    logger.warning("Local response failed JSON validation.")
                    
            # Code compiler verification
            elif task_type == "code":
                is_valid_syntax, error_msg = self.self_check.validate_python_syntax(local_response.content)
                if not is_valid_syntax:
                    logger.warning("Local code failed syntax compilation check. Retrying self-correction...")
                    # Local self-correction loop (1 free retry)
                    correction_prompt = self.self_check.build_self_correction_prompt(
                        original_prompt=compressed_prompt,
                        failed_code=local_response.content,
                        error_msg=error_msg
                    )
                    try:
                        corrected_response = self.local_client.generate(correction_prompt, max_tokens=max_tokens, task_type=task_type)
                        corrected_response.content = self.clean_and_extract_content(corrected_response.content, task_type)
                        is_valid_corrected, _ = self.self_check.validate_python_syntax(corrected_response.content)
                        if is_valid_corrected and not self.self_check.has_refusal(corrected_response.content):
                            logger.info("Local self-correction succeeded!")
                            local_response = corrected_response
                        else:
                            local_ok = False
                            logger.warning("Local self-correction failed syntax check.")
                    except Exception as e:
                        local_ok = False
                        logger.warning(f"Local self-correction failed: {e}")
                        
            # Confidence gating
            if local_ok and local_response.confidence >= confidence_threshold:
                logger.info("Local response passed all verification gates. Routing locally (0 remote tokens).")
                self.cache[prompt] = local_response
                return local_response
            else:
                logger.info("Local response did not clear verification gates or threshold.")
                
        # 6. Escalate to remote Fireworks API
        remote_model = self.select_remote_model(task_type)
        logger.info(f"Escalating task to remote model: {remote_model}")
        
        # Anti-yapping formatting rules added to prompt
        full_prompt = compressed_prompt
        if "strictly" not in compressed_prompt.lower():
            full_prompt = f"{compressed_prompt}\nOutput ONLY the direct answer. No intro, no explanation, no pleasantries, no yapping."
            
        remote_response = self.generate_remote(full_prompt, model=remote_model)
        remote_response.content = self.clean_and_extract_content(remote_response.content, task_type)
        
        # Save to cache
        self.cache[prompt] = remote_response
        return remote_response

