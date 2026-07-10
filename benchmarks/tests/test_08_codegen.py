import os
import re
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test, SingleTurnRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "08_codegen.jsonl")],
    rollout_processor=SingleTurnRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_08_codegen(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    
    code = re.sub(r"```python|```", "", assistant_reply).strip()
    
    namespace = {}
    try:
        exec(code, namespace)
        
        if "is_palindrome" in namespace:
            is_palindrome = namespace["is_palindrome"]
            assert is_palindrome("A man, a plan, a canal: Panama") is True, "Failed Panama palindrome check"
            assert is_palindrome("hello") is False, "Failed hello non-palindrome check"
            assert is_palindrome("racecar") is True, "Failed racecar palindrome check"
            score = 1.0
            reason = "Passes all is_palindrome assertions"
        elif "second_largest" in namespace:
            second_largest = namespace["second_largest"]
            assert second_largest([10, 20, 4, 45, 99, 99]) == 45, "Expected 45 for [10,20,4,45,99,99]"
            assert second_largest([3, 3, 3, 1, 2, 2]) == 2, "Expected 2 for [3,3,3,1,2,2]"
            score = 1.0
            reason = "Passes all second_largest assertions"
        elif "is_prime" in namespace:
            is_prime = namespace["is_prime"]
            assert is_prime(5) is True, "Expected True for 5"
            assert is_prime(4) is False, "Expected False for 4"
            assert is_prime(1) is False, "Expected False for 1"
            score = 1.0
            reason = "Passes all is_prime assertions"
        else:
            raise ValueError("No recognized function (is_palindrome, second_largest, is_prime) defined in response")
    except Exception as e:
        score = 0.0
        reason = f"Execution/assertion failed: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
