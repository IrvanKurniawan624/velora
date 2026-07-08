import os
import re
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "06_debugging.jsonl")],
    rollout_processor=SingleTurnRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_06_debugging(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    
    # Strip markdown block if present
    code = re.sub(r"```python|```", "", assistant_reply).strip()
    
    namespace = {}
    try:
        # Execute the code to define the sum_evens function
        exec(code, namespace)
        if "sum_evens" not in namespace:
            raise ValueError("Function 'sum_evens' not defined in response")
            
        sum_evens = namespace["sum_evens"]
        
        # Test assertions
        assert sum_evens([1, 2, 3, 4]) == 6, "Expected 6 for [1,2,3,4]"
        assert sum_evens([1, 3, 5]) == 0, "Expected 0 for odds"
        assert sum_evens([]) == 0, "Expected 0 for empty list"
        
        score = 1.0
        reason = "Passes all debugger unit assertions"
    except Exception as e:
        score = 0.0
        reason = f"Execution/assertion failed: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
