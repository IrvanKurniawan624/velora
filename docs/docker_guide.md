# Docker Build & Run Guide

This guide explains how to build, run, and locally test the agent container under the exact resource limits of the Track 1 grading environment (4 GB RAM, 2 vCPUs).

---

## 1. Prerequisites

Before running, create a `.env` file at the project root with your API credentials (see `.env.example`):

```
FIREWORKS_API_KEY=your_api_key_here
```

> **Note:** The `.env` file is listed in `.dockerignore` and is never copied into the image. It is injected at runtime only via `--env-file`.

---

## 2. Build the Docker Image

To compile dependencies (`llama-cpp-python` wheels), install the OpenMP runtime (`libgomp1`) required by the local GGUF model, and bundle the quantized model weights (`gemma-2-2b-it-Q4_K_M.gguf`), build the image from the project root:

```bash
docker build -t velora-agent .
```

> **First build:** The GGUF model (~1.6 GB) is downloaded during build. Subsequent rebuilds of code-only changes reuse the cached layer and complete in ~10 seconds.

---

## 3. Run the Agent (Grading Mode)

Run the container with CPU, Memory, and volume constraints matching the sandbox grading host. Use `--env-file .env` to securely inject credentials:

### Linux / macOS (bash):
```bash
docker run --env-file .env --cpus="2.0" --memory="4g" \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  velora-agent
```

### Windows PowerShell:
```powershell
docker run --env-file .env --cpus="2.0" --memory="4g" `
  -v "${pwd}/input:/input" `
  -v "${pwd}/output:/output" `
  velora-agent
```

---

## 4. Command Line Flag Breakdown

| Flag | Purpose | Explanation |
| :--- | :--- | :--- |
| `--env-file .env` | Credentials | Reads all `KEY=VALUE` pairs from `.env` and injects them as environment variables. Preferred over `-e` flags for security. |
| `--cpus="2.0"` | CPU Limit | Restricts the container to a maximum of 2 vCPUs — matches the hackathon sandbox. |
| `--memory="4g"` | RAM Limit | Restricts the container to 4 GB of RAM. The container will be OOM-killed if it exceeds this. |
| `-v ...:/input` | Input Mount | Mounts the local host directory containing `tasks.json` to `/input/tasks.json` inside the container. |
| `-v ...:/output` | Output Mount | Mounts the local host directory to `/output` where the container writes `results.json`. |

---

## 5. Verification Checklist

When the container starts, verify the following in the logs:

1. ✅ `llama-cpp-python` loads without errors (requires `libgomp1` — already installed in the image).
2. ✅ Local model initializes: look for `Loaded model from /app/models/gemma-2-2b-it-q4_k_m.gguf`.
3. ✅ Agent reads tasks from `/input/tasks.json`.
4. ✅ Tasks routed locally show `Source model: local` in stdout.
5. ✅ Results written to `/output/results.json`.

> **Troubleshooting:** If you see `libgomp.so.1: cannot open shared object file`, the image was built before the `libgomp1` fix was added. Rebuild with `docker build --no-cache -t velora-agent .`

---

## 6. Running the Benchmark Suite Inside Docker

The `benchmarks/` directory is excluded from the production image (via `.dockerignore`) to keep the image lean. Mount it as a volume at runtime to run the full accuracy benchmark inside the resource-constrained container:

### Linux / macOS (bash):
```bash
docker run --env-file .env --cpus="2.0" --memory="4g" \
  -v "$(pwd)/benchmarks:/app/benchmarks" \
  velora-agent python benchmarks/run_benchmark.py
```

### Windows PowerShell:
```powershell
docker run --env-file .env --cpus="2.0" --memory="4g" `
  -v "${pwd}/benchmarks:/app/benchmarks" `
  velora-agent python benchmarks/run_benchmark.py
```

This runs the full comparative benchmark (Baseline vs Optimized) inside the sandbox-equivalent environment. Reports are written to `benchmarks/reports/` on your host machine via the volume mount.
