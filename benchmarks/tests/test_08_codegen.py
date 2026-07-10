import os
import re
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test
from benchmarks.agent_rollout import AgentPipelineRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "08_codegen.jsonl")],
    rollout_processor=AgentPipelineRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_08_codegen(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    
    code = re.sub(r"```python|```", "", assistant_reply).strip()
    
    namespace = {}
    try:
        exec(code, namespace)
        if "is_palindrome" not in namespace:
            raise ValueError("Function 'is_palindrome' not defined in response")
            
        is_palindrome = namespace["is_palindrome"]
        
        # Verify palindrome behavior
        assert is_palindrome("A man, a plan, a canal: Panama") is True
        assert is_palindrome("hello") is False
        assert is_palindrome("racecar") is True
        
        score = 1.0
        reason = "Passes all palindrome assertions"
    except Exception as e:
        score = 0.0
        reason = f"Execution/assertion failed: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
