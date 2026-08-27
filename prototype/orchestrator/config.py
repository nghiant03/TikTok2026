"""Paths and constants. Run everything from the starter-kit root."""
from pathlib import Path
import json

KIT_DIR = Path(__file__).resolve().parent.parent          # kuairand-starter-kit/
DATA_DIR = KIT_DIR / "KuaiRand-Pure" / "data"
RUNS_DIR = KIT_DIR / "runs"
LOG_PATH = KIT_DIR / "run_log.jsonl"
KIT_FILES = ["baseline.py", "data.py", "evaluate.py", "submit.py"]

_scores = json.loads((KIT_DIR / "baseline_scores.json").read_text())
# baseline_scores.json layout may differ; fall back to README numbers
BASELINE_VALID_PRIMARY = 0.6016
BASELINE_TEST_PRIMARY = 0.5946
RANDOM_TEST_PRIMARY = 0.4753
EPS = 0.002
N_CONVERGE = 3
MAX_RETRY = 2
MAX_ITERS = 15
TIMEOUT_SEC = 900
