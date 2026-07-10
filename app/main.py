import json
import os
import pathlib
import sys
from openai import OpenAI
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def resolve_model(prompt: str, allowed_models: list[str]) -> str:
    # Determine task type
    prompt_lower = prompt.lower()
    
    # We check for keywords indicating a coding task
    code_keywords = [
        "code", "debug", "function", "compile", "syntax", "python", "javascript", 
        "java", "c++", "def ", "class ", "return ", "import ", "fn ", "is_palindrome", "sum_evens"
    ]
    
    is_code_task = any(kw in prompt_lower for kw in code_keywords)
    
    target_substring = "kimi-k2p7-code" if is_code_task else "minimax-m3"
    
    # Find matching model in allowed_models
    for model in allowed_models:
        if target_substring in model:
            return model
            
    # Try alternate if target not found
    alternate_substring = "minimax-m3" if is_code_task else "kimi-k2p7-code"
    for model in allowed_models:
        if alternate_substring in model:
            return model
            
    # Fallback to defaults
    if is_code_task:
        return "accounts/fireworks/models/kimi-k2p7-code"
    else:
        return "accounts/fireworks/models/minimax-m3"

def main() -> None:
    print("Initializing Velora AI Agent...")
    
    # 1. Resolve paths for input/output files
    # Check absolute /input first, then look for relative paths
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
    # If the system /output dir exists, write there. Otherwise write locally.
    output_dir = pathlib.Path("/output")
    if output_dir.exists() and os.access(output_dir, os.W_OK):
        output_file = output_dir / "results.json"
    else:
        # Fall back to local output directory
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
    
    # 3. Initialize API client
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    
    # Parse allowed models
    allowed_models_str = os.environ.get("ALLOWED_MODELS", "")
    allowed_models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
    print(f"Allowed models: {allowed_models}")
    
    if not api_key:
        print("Warning: FIREWORKS_API_KEY environment variable is not set.", file=sys.stderr)
        
    client = OpenAI(
        api_key=api_key or "mock-key",
        base_url=base_url
    )
    
    # 4. Process tasks
    results = []
    for idx, task in enumerate(tasks, 1):
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        
        if not task_id or not prompt:
            print(f"Skipping task index {idx} due to missing task_id or prompt.")
            continue
            
        print(f"[{idx}/{len(tasks)}] Processing task {task_id}...")
        
        # Resolve target model
        model = resolve_model(prompt, allowed_models)
        print(f"Selected model: {model}")
        
        answer = ""
        try:
            # Call Fireworks API
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for deterministic/accurate results
                max_tokens=1024   # Appropriate limit for generation/debugging tasks
            )
            answer = response.choices[0].message.content or ""
            print(f"Success. Answer length: {len(answer)} chars.")
        except Exception as e:
            print(f"Error processing task {task_id}: {e}", file=sys.stderr)
            answer = f"Error during model generation: {e}"
            
        results.append({
            "task_id": task_id,
            "answer": answer
        })
        
    # 5. Write results
    try:
        # Ensure parent directories exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Successfully wrote results to {output_file}")
    except Exception as e:
        print(f"Error writing results to output file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("Velora AI Agent execution complete.")

if __name__ == "__main__":
    main()
