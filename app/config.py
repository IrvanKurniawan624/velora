"""Runtime configuration read from the environment.

Every value is resolved at import time so the engine can be configured without
changing code. Defaults are set for the judging container (MODE=zero, 10-min
deadline, 4 GB / 2 vCPU model sizing).
"""
import os

SOFT_DEADLINE = float(os.environ.get("SOFT_DEADLINE", "500"))  # stop optional work
HARD_DEADLINE = float(os.environ.get("HARD_DEADLINE", "560"))  # write_results + exit
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MODE = os.environ.get("MODE", "zero").lower()
ESC_MAX = int(os.environ.get("ESC_MAX", "6"))
ESC_CONF = float(os.environ.get("ESC_CONF", "0.55"))
