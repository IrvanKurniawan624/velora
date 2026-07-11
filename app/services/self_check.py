import json
import logging
import re
import traceback

logger = logging.getLogger(__name__)

class SelfCheckService:
    def __init__(self):
        # Common refusal patterns in LLM outputs
        self.refusal_patterns = [
            r"i am sorry",
            r"i apologize",
            r"as an ai language model",
            r"i cannot fulfill this request",
            r"i am unable to",
            r"i don't know",
            r"i do not know",
            r"cannot answer",
            r"cannot comply",
            r"unable to answer",
            r"against my guidelines"
        ]

    def has_refusal(self, text: str) -> bool:
        """
        Returns True if the text contains a standard model refusal phrase.
        """
        text_lower = text.lower()
        for pattern in self.refusal_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"Refusal pattern matched: '{pattern}'")
                return True
        return False

    def validate_json_structure(self, text: str, required_keys: list[str] = None) -> bool:
        """
        Verifies if the response text is valid JSON (after cleaning markdown code block backticks).
        Optionally checks if it contains all required keys.
        """
        # Clean markdown code blocks
        clean_text = re.sub(r"```json|```", "", text).strip()
        try:
            data = json.loads(clean_text)
            if required_keys:
                for key in required_keys:
                    if key not in data:
                        logger.warning(f"JSON validation failed: missing key '{key}'")
                        return False
            return True
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
            return False

    def validate_python_syntax(self, code_text: str) -> tuple[bool, str]:
        """
        Extracts python code blocks and tries to compile them to verify syntax.
        Returns a tuple of (is_valid, error_message).
        """
        # Extract code from markdown blocks if present
        code = code_text
        if "```" in code_text:
            match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                # Fallback to general code block stripping
                match = re.search(r"```(.*?)```", code_text, re.DOTALL)
                if match:
                    code = match.group(1).strip()

        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            # Format compiler error message for self-correction
            tb = traceback.format_exception_only(type(e), e)
            error_msg = "".join(tb).strip()
            logger.warning(f"Python compilation failed: {error_msg}")
            return False, error_msg
        except Exception as e:
            logger.warning(f"Compilation checker error: {e}")
            return False, str(e)

    def build_self_correction_prompt(self, original_prompt: str, failed_code: str, error_msg: str) -> str:
        """
        Constructs the self-correction prompt to request the local model to patch the compilation error.
        """
        return f"""The code you generated contains a syntax error and fails to compile.

Original task:
{original_prompt}

Your previous implementation:
```python
{failed_code}
```

Compilation Error:
{error_msg}

Please fix the syntax error. Return ONLY the corrected python function implementation. No explanation, no commentary."""
