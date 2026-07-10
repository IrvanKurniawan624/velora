# JIT Router -- Adaptive History-Aware Model Routing

This module implements a dynamic model routing system designed to optimize cost and latency while maintaining output accuracy.

---

## How It Works

The Just-In-Time (JIT) Router selects the most suitable model for a given query dynamically at runtime:

1. **Feature Extraction:** Analyzes prompt characteristics (length, word count, code indicators, JSON requirement, complexity).
2. **History Retrieval:** Performs a k-NN cosine similarity search against past queries in the History Database.
3. **Model Selection:** Employs a LinUCB Contextual Bandit policy to pick the initial model based on prompt features and historical success priors.
4. **Quality Estimation:** Evaluates the generated output via the `ConfidenceEstimator` (verifies JSON formats, code syntax parsing, completeness, repetition).
5. **Cascading Escalation:** If the output fails the confidence check, it escalates to the next model in the hierarchy (`Qwen3-3B -> Qwen3-7B -> Fireworks`).

---

## File Directory Structure

* `schemas.py`: Pydantic definitions for features, breakdown scores, records, and statistics.
* `feature_extractor.py`: Extracts syntactic and semantic features from incoming prompts.
* `confidence_estimator.py`: Heuristics and syntax validators to score output confidence.
* `routing_policy.py`: Core LinUCB disjoint contextual bandit implementation.
* `decision_engine.py`: Integrates and orchestrates the routing and escalation lifecycle.
* `history_store.py`: Vector search database and statistical aggregator of previous runs.
* `performance_memory.py`: In-memory tracking of rolling performance stats per category.
* `router_metrics.py`: Metrics manager logging latency, costs, and escalation frequency.
* `embedding_service.py`: Computes semantic representation vectors for k-NN queries.
* `lmstudio_invoker.py`: Concrete API handler connecting to LM Studio and Fireworks APIs.
* `README.md`: Setup guidelines and module information.

---

## Configuration Settings

Define the routing settings in your root `.env` file. These options are additive and will not override or conflict with other components:

```env
# Fireworks API Setup (escalation fallback tier)
FIREWORKS_API_KEY=your_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
FIREWORKS_MODEL=accounts/fireworks/models/qwen3-30b-a3b

# Local Server Settings (LM Studio)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL_3B=qwen3-3b
LMSTUDIO_MODEL_7B=qwen3-7b

# JIT Tuning Thresholds
CONFIDENCE_THRESHOLD=0.7
JIT_COST_LAMBDA=0.3
```

---

## Running Benchmarks

### 1. Setup Environment

Install dependencies and set up your local configuration:
```bash
uv sync
cp .env.example .env
# Edit the .env file with your local model names and keys
```

### 2. Simulated Benchmark Run (Mock Mode)

Run the benchmark suite using local simulations without making real API calls:
```bash
python -m benchmarks.jit_live_benchmark --mock
```

### 3. Real Benchmark Run (LM Studio)

Make sure LM Studio is open, your target model is loaded, and the local server is running at port `1234`. Run:
```bash
python -m benchmarks.jit_live_benchmark --model-3b "qwen2.5-coder-3b-instruct"
```

To run a benchmark on a single category (e.g., `math`):
```bash
python -m benchmarks.jit_live_benchmark --model-3b "qwen2.5-coder-3b-instruct" --category math
```

To inspect full detailed logs per query:
```bash
python -m benchmarks.jit_live_benchmark --model-3b "qwen2.5-coder-3b-instruct" --verbose
```

### 4. Router Tests

Execute unit tests verifying router logic:
```bash
python test_router.py
```
