"""Execution helpers: sandbox per iteration, subprocess with timeout, scoring, submission."""
import os, re, shutil, subprocess, sys, time, json
from pathlib import Path
import numpy as np
from config import *

sys.path.insert(0, str(KIT_DIR))
from data import load                      # noqa: E402
from evaluate import evaluate              # noqa: E402

_SPLITS = None
def splits():
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = load(str(DATA_DIR))
    return _SPLITS

def labels(split):
    rws = splits()[split]
    return ([x[1] for x in rws], [x[2] for x in rws], np.array([x[6] for x in rws], dtype=np.float32))

# ---------- sandbox ----------
def make_sandbox(it: int, base_from: Path | None = None) -> Path:
    """Create runs/iter_N with kit files (+ train.py). If base_from given, copy its code instead."""
    wd = RUNS_DIR / f"iter_{it:03d}"
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    src = base_from or KIT_DIR
    for f in KIT_FILES:
        shutil.copy(src / f, wd / f)
    train_src = (base_from / "train.py") if base_from else (Path(__file__).parent / "train.py")
    shutil.copy(train_src, wd / "train.py")
    os.symlink(KIT_DIR / "KuaiRand-Pure", wd / "KuaiRand-Pure")
    return wd

def run_cmd(cmd: list[str], cwd: Path, timeout=TIMEOUT_SEC) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "code": p.returncode, "stdout": p.stdout,
                "stderr": p.stderr[-4000:], "sec": round(time.time() - t0, 1)}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "code": -1, "stdout": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
                "stderr": f"TIMEOUT after {timeout}s", "sec": timeout}

# ---------- training / scoring ----------
def train_in_sandbox(wd: Path, extra_args: list[str]) -> dict:
    r = run_cmd([sys.executable, "train.py", *extra_args], cwd=wd)
    (wd / "train_stdout.txt").write_text(r["stdout"])
    (wd / "train_stderr.txt").write_text(r["stderr"])
    return r

def score_sandbox(wd: Path) -> dict:
    """Score preds_valid.npy (and test if present) with the official evaluate()."""
    out = {}
    for sp in ("valid", "test"):
        f = wd / f"preds_{sp}.npy"
        if not f.exists():
            if sp == "valid":
                raise FileNotFoundError("preds_valid.npy missing")
            continue
        s = np.load(f)
        u, _, y = labels(sp)
        if len(s) != len(y) or not np.all(np.isfinite(s)):
            raise ValueError(f"bad preds_{sp}: len {len(s)} vs {len(y)} or non-finite")
        r = evaluate(u, y, s)
        out[sp] = {"gauc": float(r["GAUC"]), "ndcg5": float(r["nDCG@5"]), "primary": float(r["primary"])}
    return out

# ---------- submission ----------
def write_submission(wd: Path, out_csv: Path, split="test") -> dict:
    s = np.load(wd / f"preds_{split}.npy")
    u, v, _ = labels(split)
    with open(out_csv, "w") as f:
        f.write("row_id,user_id,video_id,score\n")
        for i, (uu, vv, ss) in enumerate(zip(u, v, s)):
            f.write(f"{i},{uu},{vv},{ss:.6f}\n")
    r = run_cmd([sys.executable, "submit.py", "--check", "--split", split, str(out_csv)], cwd=KIT_DIR)
    return r

def code_diff(wd: Path, prev: Path | None) -> str:
    base = prev if prev else KIT_DIR
    diffs = []
    for f in ["train.py", "baseline.py", "data.py"]:
        a = base / f if (base / f).exists() else Path(__file__).parent / f
        if not a.exists():
            continue
        r = subprocess.run(["diff", "-u", str(a), str(wd / f)], capture_output=True, text=True)
        if r.stdout:
            diffs.append(r.stdout)
    return "\n".join(diffs)[:20000]
