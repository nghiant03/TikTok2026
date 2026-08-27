"""python3 orchestrator/run.py   (from the starter-kit root)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import RUNS_DIR, LOG_PATH
from graph import build, INITIAL

if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    final = build().invoke(INITIAL, config={"recursion_limit": 500})
    print(f"\nDONE  best valid primary {final['best_primary']:.4f} @ iter {final['best_iteration']}")
    print(f"log: {LOG_PATH}")
