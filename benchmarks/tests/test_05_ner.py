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
    input_dataset=[os.path.join(DATASETS_DIR, "05_ner.jsonl")],
    rollout_processor=AgentPipelineRolloutProcessor(),
    completion_params=[{"model": TEST_MODEL}],
    mode="pointwise"
)
def test_05_ner(row: EvaluationRow) -> EvaluationRow:
    assistant_reply = row.messages[-1].content or ""
    expected = json.loads(row.ground_truth)
    
    try:
        clean_reply = re.sub(r"```json|```", "", assistant_reply).strip()
        data = json.loads(clean_reply)
        
        # Calculate matching entities
        score = 1.0
        reasons = []
        for key in ["PERSON", "ORG", "LOC"]:
            model_entities = [e.lower() for e in data.get(key, [])]
            expected_entities = [e.lower() for e in expected.get(key, [])]
            
            matches = set(model_entities).intersection(set(expected_entities))
            if len(matches) != len(expected_entities):
                score -= 0.33
                reasons.append(f"Mismatch in {key}")
            else:
                reasons.append(f"Correct {key}")
                
        score = max(0.0, min(1.0, score))
        reason = ", ".join(reasons)
    except Exception as e:
        score = 0.0
        reason = f"Failed to parse JSON: {str(e)}"
        
    row.evaluation_result = EvaluateResult(score=score, reason=reason)
    return row
