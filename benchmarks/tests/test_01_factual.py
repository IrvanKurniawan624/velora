import os
import json
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test
from benchmarks.agent_rollout import AgentPipelineRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "01_factual.jsonl")],
    rollout_processor=AgentPipelineRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_01_factual(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    truth_data = json.loads(row.ground_truth)
    keywords = truth_data.get("keywords", [])
    
    matched = [kw for kw in keywords if kw.lower() in assistant_reply.lower()]
    score = len(matched) / len(keywords) if keywords else 1.0
    
    row.evaluation_result = EvaluateResult(
        score=score,
        reason=f"Matched {len(matched)}/{len(keywords)} keywords: {matched}"
    )
    return row
