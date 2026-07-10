import ast
import json
import logging
import re
from typing import List

from app.router.schemas import ConfidenceBreakdown

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    @staticmethod
    def estimate(prompt: str, response: str) -> ConfidenceBreakdown:

        score = 1.0
        failure_reasons: List[str] = []

        length_ok = True
        json_valid = None  # None = not applicable
        code_syntax_ok = None
        completeness_ok = True
        not_repetitive = True
        not_refusal = True

        prompt_words = len(prompt.split())
        resp_words = len(response.split())
        prompt_lower = prompt.lower()
        resp_lower = response.lower()

        # ── 1. Length adequacy ──────────────────────────────────────────
        if prompt_words > 40 and resp_words < 5:
            score -= 0.5
            length_ok = False
            failure_reasons.append("response_too_short")

        # Reasoning prompts deserve longer answers
        reasoning_cues = [
            "explain",
            "step by step",
            "analyze",
            "compare",
            "derive",
            "describe in detail",
        ]
        if any(c in prompt_lower for c in reasoning_cues) and resp_words < 15:
            score -= 0.3
            length_ok = False
            if "response_too_short" not in failure_reasons:
                failure_reasons.append("reasoning_answer_too_brief")

        # ── 2. JSON validity ───────────────────────────────────────────
        if re.search(r"\bjson\b", prompt, re.IGNORECASE):
            json_valid = False  # assume invalid until proven
            json_block = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            text_to_check = json_block.group(1) if json_block else response

            try:
                start = text_to_check.find("{")
                if start == -1:
                    start = text_to_check.find("[")
                if start != -1:
                    parsed = json.loads(text_to_check[start:])
                    json_valid = True

                    # Optional: check for required keys if prompt mentions them
                    key_match = re.findall(
                        r'"(\w+)"', prompt
                    )  # keys mentioned in prompt
                    if key_match and isinstance(parsed, dict):
                        missing = [k for k in key_match if k not in parsed]
                        if missing:
                            score -= 0.2
                            failure_reasons.append(
                                f"json_missing_keys:{','.join(missing[:3])}"
                            )
                else:
                    score -= 0.8
                    failure_reasons.append("json_structure_missing")
            except json.JSONDecodeError:
                score -= 0.6
                failure_reasons.append("json_parse_error")

        # ── 3. Code syntax check ───────────────────────────────────────
        code_block = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
        if code_block or ("def " in response and "return" in response):
            code_syntax_ok = False
            code_text = code_block.group(1) if code_block else response

            # Extract just the Python code portion
            lines = []
            in_code = False
            for line in code_text.split("\n"):
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "import ", "from ")):
                    in_code = True
                if in_code:
                    lines.append(line)

            if lines:
                try:
                    ast.parse("\n".join(lines))
                    code_syntax_ok = True
                except SyntaxError:
                    score -= 0.4
                    failure_reasons.append("code_syntax_error")
            else:
                # No clear code block found to parse, neutral
                code_syntax_ok = None

        # ── 4. Completeness heuristic ──────────────────────────────────
        list_match = re.search(r"list\s+(\d+)", prompt, re.IGNORECASE)
        if list_match:
            expected = int(list_match.group(1))
            bullet_count = len(re.findall(r"(?:^|\n)\s*[-*•]|\d+\.", response))
            if bullet_count < expected * 0.5:
                score -= 0.4
                completeness_ok = False
                failure_reasons.append(
                    f"incomplete_list:expected_{expected}_got_{bullet_count}"
                )

        # ── 5. Repetition / degenerate output ──────────────────────────
        words = response.split()
        if len(words) > 20:
            vocab = set(words)
            ratio = len(vocab) / len(words)
            if ratio < 0.2:
                score -= 0.7
                not_repetitive = False
                failure_reasons.append("degenerate_repetition")
            elif ratio < 0.3:
                score -= 0.3
                failure_reasons.append("high_repetition")

        # ── 6. Refusal / hedging detection ─────────────────────────────
        refusal_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot",
            "as an ai",
            "i'm unable",
            "i can't help",
            "i apologize, but i",
        ]
        for phrase in refusal_phrases:
            if phrase in resp_lower and resp_words < 30:
                score -= 0.5
                not_refusal = False
                failure_reasons.append("refusal_or_hedge")
                break

        overall = max(0.0, min(score, 1.0))

        breakdown = ConfidenceBreakdown(
            overall=round(overall, 4),
            length_ok=length_ok,
            json_valid=json_valid,
            code_syntax_ok=code_syntax_ok,
            completeness_ok=completeness_ok,
            not_repetitive=not_repetitive,
            not_refusal=not_refusal,
            failure_reasons=failure_reasons,
        )

        if failure_reasons:
            logger.info(
                "  Confidence %.3f — failures: %s",
                overall,
                ", ".join(failure_reasons),
            )

        return breakdown
