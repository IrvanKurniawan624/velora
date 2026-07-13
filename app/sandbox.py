"""Execute model-generated Python in a hard-limited subprocess sandbox.

The solvers never trust model-written code: every candidate program is run in a
fresh `python -I -c` child with a wall-clock cap and, on POSIX, an address + CPU
rlimit, so a broken program cannot hang the agent or exhaust memory.
"""
import os
import subprocess
import sys


def exec_sandboxed(code: str, timeout: float = 8.0):
    """Run `code` in an isolated child interpreter.

    Returns a (ok, stdout, stderr) triple. `ok` is True only when the child
    exited with status 0; timeouts and any other failure return ok=False with an
    explanatory stderr string and empty stdout.
    """
    argv = [sys.executable, "-I", "-c", code]
    extra = {}
    if os.name == "posix":
        import resource

        def _clamp():
            # cap resident set at ~1.5 GB and CPU just above the wall-clock budget
            resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 2, int(timeout) + 2))

        extra["preexec_fn"] = _clamp
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
            env={"PYTHONIOENCODING": "utf-8"}, **extra,
        )
        return (proc.returncode == 0,
                (proc.stdout or "").strip(),
                (proc.stderr or "").strip())
    except subprocess.TimeoutExpired:
        return (False, "", "timeout")
    except Exception as exc:  # pragma: no cover
        return (False, "", f"{type(exc).__name__}: {exc}")
