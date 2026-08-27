"""Research / Implementation agent stubs.
Replace `propose()` and `implement()` bodies with LLM calls later; keep the contracts.

Hypothesis contract:
  {id, text, target_module, rationale, references,
   train_args: [...],              # CLI args passed to train.py
   patches: [{file, find, replace}]}   # optional literal edits applied in the sandbox
"""
from pathlib import Path

# Ordered queue based on EDA + README (dead ends excluded on purpose)
QUEUE = [   
    
    {"id": "h1", "text": "Replace pointwise logloss with within-user BPR pairwise loss",
     "target_module": "loss", "references": ["Rendle 2009 BPR"],
     "rationale": "Metric is within-user ranking; median 4 exposures/user; 17.5M train pairs; untested by organizers",
     "train_args": ["--loss", "bpr", "--lr", "0.001"], "patches": []},
    {"id": "h2", "text": "BPR with 2 negatives per positive and higher lr",
     "target_module": "loss", "references": ["BPR"],
     "rationale": "More pairs per epoch; check sensitivity", 
     "train_args": ["--loss", "bpr", "--lr", "0.002", "--neg_per_pos", "2"], "patches": []},
    {"id": "h3", "text": "BPR with patience 8 (longer training)",
     "target_module": "training", "references": [],
     "rationale": "Pairwise curve may be slower/noisier than logloss",
     "train_args": ["--loss", "bpr", "--lr", "0.001", "--patience", "8"], "patches": []},
    # TODO real research agent: multi-task (is_click aux), censored watch-time, history features
]

def propose(history: list[dict], last_error: str | None) -> dict | None:
    tried = {h.get("hypothesis", {}).get("id") for h in history if h.get("event") == "hypothesis"}
    for h in QUEUE:
        if h["id"] not in tried:
            return h
    return None

def implement(wd: Path, hyp: dict) -> list[str]:
    """Apply patches in sandbox; return train.py args. Raises if a patch doesn't match."""
    for p in hyp.get("patches", []):
        f = wd / p["file"]; src = f.read_text()
        if p["find"] not in src:
            raise RuntimeError(f"patch target not found in {p['file']}: {p['find'][:60]!r}")
        f.write_text(src.replace(p["find"], p["replace"], 1))
    return hyp.get("train_args", [])
