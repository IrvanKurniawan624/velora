import os
import json
import re
from eval_protocol.models import EvaluationRow, EvaluateResult
from eval_protocol.pytest import evaluation_test
from benchmarks.agent_rollout import AgentPipelineRolloutProcessor

# Resolve dataset paths relative to this file
DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
TEST_MODEL = os.getenv("AGENT_TEST_MODEL", "openai/mock-model")

@evaluation_test(
    input_dataset=[os.path.join(DATASETS_DIR, "03_sentiment.jsonl")],
    rollout_processor=AgentPipelineRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_03_sentiment(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    expected = json.loads(row.ground_truth)
    
    try:
        # Strip markdown code blocks if model wrapped JSON
        clean_reply = re.sub(r"```json|```", "", assistant_reply).strip()
        data = json.loads(clean_reply)
        
        sentiment_match = data.get("sentiment", "").lower() == expected.get("sentiment", "").lower()
        has_reason = "reason" in data and len(data["reason"]) > 0
        
        score = 1.0 if (sentiment_match and has_reason) else 0.5 if sentiment_match else 0.0
        reason = f"Sentiment match: {sentiment_match}, Has reason: {has_reason}"
    except Exception as e:
        score = 0.0
        reason = f"Failed to parse JSON: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
