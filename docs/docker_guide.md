# Docker Build & Run Guide

This guide explains how to build, run, and locally test the Velora AI Agent container under the exact resource limits of the Track 1 grading environment (4 GB RAM, 2 vCPUs).

---

## 1. Build the Docker Image
To compile dependencies (`llama-cpp-python` wheels) and bundle the quantized local model weights (`gemma-2-2b-it-Q4_K_M.gguf`), build the image from the project root:

```bash
docker build -t velora-agent .
```

---

## 2. Run with Resource Limits
Run the container with CPU, Memory, and Mount constraints matching the sandbox grading host:

```bash
docker run -it --cpus="2.0" --memory="4g" \
  -v "$(pwd)/input:/input" \
  -v "$(pwd)/output:/output" \
  -e FIREWORKS_API_KEY="your_api_key_here" \
  velora-agent
```

### Windows PowerShell equivalent:
```powershell
docker run -it --cpus="2.0" --memory="4g" `
  -v "${pwd}/input:/input" `
  -v "${pwd}/output:/output" `
  -e FIREWORKS_API_KEY="your_api_key_here" `
  velora-agent
```

---

## 3. Command Line Flag Breakdown

| Flag | Purpose | Explanation |
| :--- | :--- | :--- |
| `--cpus="2.0"` | CPU Limit | Restricts the container to a maximum of 2 vCPUs of host processing power. |
| `--memory="4g"` | RAM Limit | Restricts the container to 4 GB of RAM. The container will face OOM termination if it exceeds this. |
| `-v ...:/input` | Input Mount | Mounts the local host directory containing `tasks.json` to `/input/tasks.json` in the container. |
| `-v ...:/output` | Output Mount | Mounts the local host directory to `/output` where the container writes `results.json`. |
| `-e FIREWORKS_API_KEY="..."` | API Key Pass | Injects your Fireworks AI API key into the container's environment. |

---

## 4. Verification Check
Inside the container:
1. `llama-cpp-python` will load the compiled C++ bindings.
2. The agent will initialize and read tasks from `/input/tasks.json`.
3. It will process all tasks using the hybrid routing orchestrator.
4. It will write final formatted output to `/output/results.json`.

---

## 5. Running the Benchmark Suite Inside Docker
Since the `benchmarks/` directory is ignored in `.dockerignore` to minimize the production image size, you can mount the benchmarks folder as a volume at runtime to run the accuracy scorecard check inside the resource-constrained container:

```bash
docker run -it --cpus="2.0" --memory="4g" \
  -v "$(pwd)/benchmarks:/app/benchmarks" \
  -e FIREWORKS_API_KEY="your_api_key_here" \
  velora-agent python benchmarks/run_benchmark.py
```

### Windows PowerShell equivalent:
```powershell
docker run -it --cpus="2.0" --memory="4g" `
  -v "${pwd}/benchmarks:/app/benchmarks" `
  -e FIREWORKS_API_KEY="your_api_key_here" `
  velora-agent python benchmarks/run_benchmark.py
```

This runs the exact `run_benchmark.py` grading simulation inside the limited sandbox to verify 100% correct setup and execution!

