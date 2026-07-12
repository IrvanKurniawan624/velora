# Velora — Zero-Token Local-First Agent (AMD Hackathon ACT II, Track 1)

A general-purpose AI agent for the AMD Developer Hackathon ACT II **Track 1
(Hybrid Token-Efficient Routing Agent)**. It answers all eight task categories
**entirely inside the container** with a local Qwen2.5-3B-Instruct model
(llama.cpp) plus deterministic verification — spending **zero Fireworks
tokens** in its default mode, the best possible token score, while clearing
the accuracy gate.

```
/input/tasks.json
      │
      ▼
┌──────────────┐   regex, 0 cost
│  classifier   │──────────────► category (factual / math / sentiment /
└──────────────┘                summarize / NER / code-debug / code-gen / logic)
      │
      ▼
┌───────────────────────────────────────────────────────────┐
│ Pass 1 — bank a best-shot answer for every task           │
│   • math  : LLM writes a tiny Python program → executed   │
│             twice independently → results must agree      │
│   • code  : generated/fixed code is compiled AND run,     │
│             then differential-fuzz-checked vs a reference │
│   • logic : executable brute-force enumeration + truth-   │
│             table decomposition + cross-checked CoT       │
│   • facts : offline capitals gazetteer + 2 independent    │
│             answers cross-checked for agreement           │
│   • text  : sentiment/NER/summarize with format           │
│             constraints enforced programmatically         │
├───────────────────────────────────────────────────────────┤
│ Pass 2 — remaining time re-verifies low-confidence tasks  │
│   (MODE=hybrid only: escalate still-low-conf to Fireworks)│
├───────────────────────────────────────────────────────────┤
│ Watchdog — guarantees valid, complete results.json and    │
│            exit 0 well before the 10-minute limit         │
└───────────────────────────────────────────────────────────┘
      │
      ▼
/output/results.json
```

## Why local-only (this branch: `feat/zero-api-local`)

Track 1 scores by **remote tokens only** — local model inference inside the
container counts as **zero** toward the token score. The optimal strategy is
therefore to do everything locally and escalate to Fireworks only when a task
cannot be independently verified. This branch makes local-only the default
(`MODE=zero`) and keeps `MODE=hybrid` as an escape hatch.

It retires the earlier speculative-routing + persistent fuzzy-cache pipeline
(`app/`), which (a) committed a pre-hydrated answer cache that violated the
**"No Hardcoding/Caching"** rule and (b) defaulted most tasks to remote,
producing ~6k tokens on the real (unseen-variant) evaluation. The engine is
now the proven ZeroFire design: **verification-based local solving, no answer
cache** — arithmetic/logic correctness comes from executing and cross-checking
model-written code, not from the 3B's mental reasoning.

## Local-context optimization

The agent is tuned to squeeze accuracy out of a 3B model under the 4 GB / 2 vCPU
judge budget:

- **Context window** — `LLAMA_CTX=4096` (env-tunable), enough for long-passage
  summarisation/NER. Served by `llama-server` as a subprocess, not in-process.
- **KV-cache reuse** — `--cache-reuse 256`. The 2-pass design and the
  math/logic voting ballots re-issue the same task prompt repeatedly; cache
  reuse makes repeats near-instant — large wall-clock savings under the
  10-minute cap.
- **Verification over guessing** — math is solved by executing Python twice
  and majority-voting (programs weight 2, CoT weight 1); code is compiled, run,
  and differential-fuzz-checked against an independent reference; logic uses
  executable enumeration + truth-table decomposition. The model writes code;
  the interpreter guarantees correctness.
- **Per-category prompting** — tight, anti-yapping system prompts with firm
  format anchors (NER `Entity - Type` one-per-line, code-gen single
  ```` ```python ```` fence + asserts, math `ANSWER: <n>`).
- **2-pass pacing** — Pass 1 banks a best-shot answer for every task (cheap
  categories first); Pass 2 spends remaining wall-clock re-verifying
  low-confidence tasks with the executable solvers. A watchdog flushes and
  exits 0 at `HARD_DEADLINE`.
- **Offline general-knowledge RAG** — a capitals gazetteer (country → capital +
  nearby body of water) answers capital questions deterministically, beating a
  3B's spotty geography. General reference data only — no task-specific answers.
- **JSON output on demand** — when a task explicitly requests JSON
  (sentiment/summarise/NER), the handler's plain-text answer is repackaged into
  the requested JSON shape (`{"sentiment", "reason"}`, `{"bullets": [...]}`,
  `{"PERSON": [...], ...}` grouped by the requested keys); the proven
  plain-text path is preserved for every other prompt.

## Container contract (Track 1)

- Reads `/input/tasks.json`, writes `/output/results.json`
  (`[{"task_id": ..., "answer": ...}]`), exits 0.
- `linux/amd64`, starts in seconds, finishes well inside the 10-minute cap; a
  watchdog flushes results and exits 0 even in worst-case stalls.
- Env consumed at runtime: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`,
  `ALLOWED_MODELS` (hybrid mode only; zero mode ignores them) plus
  `LLAMA_BIN`, `MODEL_PATH`, `LLAMA_THREADS`, `LLAMA_CTX`, `MODE`,
  `SOFT_DEADLINE`, `HARD_DEADLINE`.

## Modes

- **`MODE=zero` (default, image tag `:zero`)** — never calls Fireworks. Local
  models are explicitly a valid strategy; local tokens score 0. This is the
  submission.
- **`MODE=hybrid` (image tag `:latest`)** — identical pipeline, but tasks whose
  answers could not be independently verified escalate — one terse call each —
  to the cheapest suitable model from `ALLOWED_MODELS` via `FIREWORKS_BASE_URL`
  (env-injected, never hardcoded), with `reasoning_effort` disabled and
  `max_tokens` capped.

## Run it (Docker)

```bash
docker buildx build --platform linux/amd64 --tag velora-agent:zero --build-arg MODE=zero .
docker run --rm --cpus=2 --memory=4g \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  velora-agent:zero
```

`/path/to/input/tasks.json`:

```json
[ { "task_id": "t1", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?" } ]
```

## Develop locally (no Docker)

```bash
# 1) download a llama.cpp release for your OS into tools/llama/
# 2) download the model:
#    https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF (Q4_K_M)
#    into models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
python eval/run_local.py                      # runs the variant local eval
python eval/run_local.py eval/practice_tasks.json
python eval/run_local.py input/tasks.json     # velora's 19-task set
```

`eval/run_local.py` spawns the server itself, or reuses one you started when
`LLAMA_URL` is set. Auto-checks math/logic/fact variants against expected
values; prints the rest for eyeballing. Runs with `MODE=zero` by default and
emits **zero** Fireworks calls.

## CI

`.github/workflows/build.yml` builds the `linux/amd64` image on push to `main`
(and on manual dispatch), pushes `velora-agent:{zero,latest,<sha>}` to GHCR,
then runs the practice task set inside the freshly built `:zero` image under
judge-like limits (`--cpus=2 --memory=4g`) and validates the output schema.

## Repository layout

```
agent/            the agent (stdlib-only Python)
  main.py         orchestrator: 2-pass pacing, watchdog, atomic flushes
  classify.py     zero-cost category router
  solvers.py      per-category handlers + verification
  capitals.py     offline country → capital + body-of-water gazetteer
  local_llm.py    llama-server lifecycle + OpenAI-compatible client
  pyexec.py       sandboxed execution of model-written Python
  fireworks.py    hybrid-mode escalation (token-accounted; zero mode never imports it)
eval/             practice tasks, checkable variants, local harness (run_local.py)
Dockerfile        3-stage build: llama.cpp release + GGUF weights + slim runtime
.github/          CI: build linux/amd64 + smoke test under judge limits
```

## Tech stack

- **Language:** Python 3.12, **stdlib only** at runtime (urllib, subprocess,
  ast, re, difflib, threading) — no `llama-cpp-python`, `openai`, or `pydantic`
  in the image.
- **Local model:** Qwen2.5-3B-Instruct Q4_K_M (~1.9 GB) served by `llama-server`
  with 2 threads — sized for the 4 GB RAM / 2 vCPU judging environment.
- **Remote (hybrid only):** Fireworks AI via `FIREWORKS_BASE_URL`, models read
  from `ALLOWED_MODELS` at runtime.
- **Dev tooling:** `pytest`, `ruff` (the vendored `agent/` + `eval/` ports are
  excluded from ruff as proven code).
