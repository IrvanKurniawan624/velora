# Agent Evaluation and Benchmarking Suite

This suite provides a standardized, offline-ready framework for evaluating, profiling, and exporting training datasets from AI agent executions.

---

## 1. Setup

Ensure the virtual environment is activated and install all project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Running Benchmarks

Evaluations are written as pytest test cases located in the `benchmarks/tests/` directory.

### Running the Entire Test Suite
By default, the benchmark runs in an offline mock sandbox using simulated model completions. To execute the entire suite:

```bash
OPENAI_API_KEY=mock-key pytest benchmarks/tests/
```

### Running a Specific Category
To execute only a single test file (e.g., Code Generation):

```bash
OPENAI_API_KEY=mock-key pytest benchmarks/tests/test_08_codegen.py
```

### Dynamically Overriding the Model (this doesn't work idk why)
You can target a specific remote model at runtime by setting the `AGENT_TEST_MODEL` environment variable. Ensure the appropriate provider prefix and API keys are exported:

```bash
AGENT_TEST_MODEL="fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct" \
FIREWORKS_API_KEY="your-api-key" \
pytest benchmarks/tests/
```

---

## 3. Viewing Results in the Dashboard

Every evaluation run automatically records trajectories and score metrics to a local SQLite database.

To view these results interactively in the web dashboard:

1. Start the persistent logs viewer server (use a mock key to satisfy internal parser client load checks if running entirely offline):
   ```bash
   FIREWORKS_API_KEY=mock-key ep logs
   ```
2. Open your web browser and navigate to:
   * Main dashboard: http://localhost:8000
   * Pivot tables comparison view: http://localhost:8000/pivot

To stop the server, press `Ctrl+C` in the terminal.

---

## 4. Exporting Datasets for Fine-Tuning

Once evaluations are completed and graded, you can parse the log database to extract successfully executed agent runs for model fine-tuning.

### Supervised Fine-Tuning (SFT) Export
Parses successful runs (score = 1.0) and generates standard conversational target files:

```bash
python export_trajectories.py
```
Output location: `benchmarks/datasets/fine_tuning_train.jsonl`

### Direct Preference Optimization (DPO) Export
Pairs successful runs (score = 1.0) with failed or synthetic suboptimal runs to create training pairs in the Fireworks DPO schema:

```bash
python export_dpo_trajectories.py
```
Output location: `benchmarks/datasets/fine_tuning_dpo.jsonl`
