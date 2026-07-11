import re
import json
import os
import pathlib
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.config import Settings
from app.services import SelfCheckService
from app.services.agent import AgentService


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json, ```python, ```) from model answers."""
    # Strip leading/trailing whitespace first
    text = text.strip()
    # Remove opening fence with optional language tag
    text = re.sub(r'^```[a-zA-Z]*\n?', '', text)
    # Remove closing fence
    text = re.sub(r'\n?```$', '', text)
    return text.strip()

def write_json_atomically(path: pathlib.Path, data) -> None:
    """Writes a JSON file atomically using a temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)

def main() -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr
    )
    print("Initializing Velora AI Agent with Speculative Routing...")
    
    # Load configuration
    settings = Settings()
    
    # 1. Resolve paths for input/output files
    input_paths = [
        pathlib.Path("/input/tasks.json"),
        pathlib.Path("input/tasks.json"),
        pathlib.Path("./input/tasks.json"),
        pathlib.Path("../input/tasks.json"),
    ]
    
    input_file = None
    for p in input_paths:
        if p.exists() and p.is_file():
            input_file = p
            break
            
    if not input_file:
        print("Error: Could not find tasks.json in any expected location.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found input tasks file: {input_file}")
    
    # Resolve output path
    output_dir = pathlib.Path("/output")
    if output_dir.exists() and os.access(output_dir, os.W_OK):
        output_file = output_dir / "results.json"
    else:
        local_output_dir = pathlib.Path("output")
        local_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = local_output_dir / "results.json"
        
    print(f"Target output file: {output_file}")
    
    # 2. Read tasks
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing input file: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not isinstance(tasks, list):
        print("Error: tasks.json must contain a JSON list of task objects.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loaded {len(tasks)} tasks.")
    
    # 3. Initialize Services
    self_check_service = SelfCheckService()
    agent_service = AgentService(settings, self_check_service)
    
    # 4. Process tasks
    results = []
    metrics = []
    for idx, task in enumerate(tasks, 1):
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        if not task_id or not prompt:
            print(f"Skipping task index {idx} due to missing task_id or prompt.")
            continue
            
        print(f"[{idx}/{len(tasks)}] Processing task {task_id}...")
        
        try:
            # Process via speculative routing service
            response = agent_service.process_task(prompt)
            answer = response.content
            model = response.model
            remote_tokens = response.remote_tokens_used
            print(f"Success. Source model: {response.model} | Answer length: {len(answer)} chars.")
        except Exception as e:
            print(f"Error processing task {task_id}: {e}", file=sys.stderr)
            answer = f"Error during model generation: {e}"
            model = "error"
            remote_tokens = 0
            
        results.append({
            "task_id": task_id,
            "answer": answer
        })
        metrics.append({
            "task_id": task_id,
            "model": model,
            "remote_tokens_used": remote_tokens
        })
        
        # Write results and metrics atomically after every task to protect against timeouts/kills
        try:
            write_json_atomically(output_file, results)
            write_json_atomically(output_file.parent / "metrics.json", metrics)
        except Exception as e:
            print(f"Warning: Failed to write intermediate results atomically: {e}", file=sys.stderr)
            
    print("Velora AI Agent execution complete.")

if __name__ == "__main__":
    main()
