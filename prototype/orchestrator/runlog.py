import json, time
from config import LOG_PATH

def log(it: int, event: str, **kw) -> dict:
    rec = {"ts": round(time.time(), 1), "iter": it, "event": event, **kw}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
