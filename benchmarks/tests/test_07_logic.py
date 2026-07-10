import os
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test
from benchmarks.agent_rollout import AgentPipelineRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "07_logic.jsonl")],
    rollout_processor=AgentPipelineRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_07_logic(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    expected = str(row.ground_truth).strip().lower()
    
    score = 1.0 if expected in assistant_reply.lower() else 0.0
    row.evaluation_result = EvaluateResult(
        score=score,
        reason=f"Found '{expected}' in response: {expected in assistant_reply.lower()}"
    )
    return row
