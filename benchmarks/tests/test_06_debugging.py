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
        # Execute the code to define the function
        exec(code, namespace)
        
        if "sum_evens" in namespace:
            sum_evens = namespace["sum_evens"]
            assert sum_evens([1, 2, 3, 4]) == 6, "Expected 6 for [1,2,3,4]"
            assert sum_evens([1, 3, 5]) == 0, "Expected 0 for odds"
            assert sum_evens([]) == 0, "Expected 0 for empty list"
            score = 1.0
            reason = "Passes all sum_evens unit assertions"
        elif "get_max" in namespace:
            get_max = namespace["get_max"]
            assert get_max([1, 5, 3, 2]) == 5, "Expected 5 for [1,5,3,2]"
            assert get_max([-10, -5, -20]) == -5, "Expected -5 for negative list"
            assert get_max([42]) == 42, "Expected 42 for single item list"
            score = 1.0
            reason = "Passes all get_max unit assertions"
        elif "factorial" in namespace:
            factorial = namespace["factorial"]
            assert factorial(0) == 1, "Expected 1 for 0!"
            assert factorial(1) == 1, "Expected 1 for 1!"
            assert factorial(5) == 120, "Expected 120 for 5!"
            score = 1.0
            reason = "Passes all factorial unit assertions"
        else:
            raise ValueError("No recognized function (sum_evens, get_max, factorial) defined in response")
    except Exception as e:
        score = 0.0
        reason = f"Execution/assertion failed: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
