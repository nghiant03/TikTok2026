"""python3 orchestrator/view_log.py  -> iteration timeline for judges"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import LOG_PATH

for line in open(LOG_PATH):
    r = json.loads(line); e = r["event"]; it = r["iter"]
    if e == "baseline_reproduced":
        print(f"[{it:02d}] BASELINE valid {r['metrics']['valid']['primary']:.4f} ok={r['ok']}")
    elif e == "hypothesis":
        h = r["hypothesis"]; print(f"[{it:02d}] HYP {h['id']}: {h['text']}")
    elif e == "code_diff":
        print(f"[{it:02d}]   diff {len(r['diff'])} chars, args {r['train_args']}")
    elif e == "metrics":
        print(f"[{it:02d}]   valid primary {r['metrics']['valid']['primary']:.4f} ({r['sec']}s)")
    elif e == "decision":
        print(f"[{it:02d}]   {'IMPROVED' if r['improved'] else 'no gain'}  best {r['best']:.4f}  streak {r['streak']}  vs baseline {r['delta_vs_baseline']:+.4f}")
    elif e == "error":
        print(f"[{it:02d}]   ERROR ({r.get('stage')}): {(r.get('stderr') or r.get('msg',''))[-200:].strip()}")
    elif e == "recovery":
        print(f"[{it:02d}]   RECOVER retry {r['retry']}")
    elif e == "abandon":
        print(f"[{it:02d}]   ABANDON {r['hypothesis_id']}")
    elif e == "finalize":
        print(f"[{it:02d}] FINAL best iter {r['best_iteration']} valid {r['best_valid_primary']:.4f} check_ok={r['check_ok']} interventions={r['human_interventions']}")
