import os
import subprocess
import sys

def run_python(code: str, timeout: float = 8.0):
    """Execute code with `python -I -c`. Returns (ok, stdout, stderr)."""
    cmd = [sys.executable, "-I", "-c", code]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={"PYTHONIOENCODING": "utf-8"}
        )
        return (r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip())
    except subprocess.TimeoutExpired:
        return (False, "", "timeout")
    except Exception as e:
        return (False, "", f"{type(e).__name__}: {e}")
