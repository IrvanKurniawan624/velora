import os
import json
import re
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "04_summarisation.jsonl")],
    rollout_processor=SingleTurnRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_04_summarisation(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    
    try:
        clean_reply = re.sub(r"```json|```", "", assistant_reply).strip()
        data = json.loads(clean_reply)
        bullets = data.get("bullets", [])
        
        # Verify it has exactly 2 bullet points
        correct_length = len(bullets) == 2
        score = 1.0 if correct_length else 0.5 if len(bullets) > 0 else 0.0
        reason = f"Found {len(bullets)} bullets (expected 2)"
    except Exception as e:
        score = 0.0
        reason = f"Failed to parse JSON: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
