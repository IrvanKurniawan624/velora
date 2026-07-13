"""Container entry point: starts the agent orchestrator."""
import sys

from app.services.agent import run, write_results

if __name__ == "__main__":
    try:
        code = run()
    except Exception as e:
        sys.stderr.write(f"UNCAUGHT: {e}\n")
        write_results()
        code = 0
    sys.exit(code)
