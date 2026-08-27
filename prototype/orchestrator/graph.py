"""LangGraph orchestration loop with real execution."""
import operator, shutil
from pathlib import Path
from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph import StateGraph, END
from config import *
import kit, agents
from runlog import log

class State(TypedDict):
    iteration: int
    baseline_ok: bool
    hypothesis: Optional[dict]
    train_args: list[str]
    workdir: Optional[str]
    metrics: Optional[dict]
    error: Optional[str]
    retry_count: int
    best_primary: float
    best_iteration: int
    best_workdir: Optional[str]
    no_improve_streak: int
    human_interventions: int
    history: Annotated[list[dict], operator.add]
    done: bool

# ---------- nodes ----------
def reproduce_baseline(s: State) -> dict:
    wd = kit.make_sandbox(0)
    r = kit.train_in_sandbox(wd, ["--loss", "logloss"])
    if not r["ok"]:
        rec = log(0, "error", stage="baseline", stderr=r["stderr"])
        return {"baseline_ok": False, "error": r["stderr"], "history": [rec]}
    m = kit.score_sandbox(wd)
    ok = abs(m["valid"]["primary"] - BASELINE_VALID_PRIMARY) <= 3 * EPS
    rec = log(0, "baseline_reproduced", metrics=m, ok=ok, sec=r["sec"])
    return {"baseline_ok": ok, "metrics": m, "workdir": str(wd),
            "best_primary": m["valid"]["primary"], "best_iteration": 0, "best_workdir": str(wd),
            "history": [rec]}

def research(s: State) -> dict:
    hyp = agents.propose(s["history"], s["error"])
    it = s["iteration"] + 1
    if hyp is None:
        rec = log(it, "research_exhausted")
        return {"iteration": it, "hypothesis": None, "history": [rec]}
    rec = log(it, "hypothesis", hypothesis=hyp)
    return {"iteration": it, "hypothesis": hyp, "error": None, "retry_count": 0, "history": [rec]}

def implement(s: State) -> dict:
    it = s["iteration"]
    # build on best-so-far code, not on the failed attempt
    wd = kit.make_sandbox(it, base_from=Path(s["best_workdir"]))
    try:
        args = agents.implement(wd, s["hypothesis"])
    except Exception as e:
        rec = log(it, "error", stage="implement", msg=str(e))
        return {"workdir": str(wd), "error": str(e), "history": [rec]}
    diff = kit.code_diff(wd, Path(s["best_workdir"]))
    rec = log(it, "code_diff", hypothesis_id=s["hypothesis"]["id"], train_args=args, diff=diff)
    return {"workdir": str(wd), "train_args": args, "error": None, "history": [rec]}

def validate(s: State) -> dict:
    it, wd = s["iteration"], Path(s["workdir"])
    if s["error"]:
        return {"error": s["error"]}
    r = kit.train_in_sandbox(wd, s["train_args"])
    if not r["ok"]:
        rec = log(it, "error", stage="train", stderr=r["stderr"], sec=r["sec"])
        return {"error": r["stderr"], "history": [rec]}
    try:
        m = kit.score_sandbox(wd)
    except Exception as e:
        rec = log(it, "error", stage="score", msg=str(e))
        return {"error": str(e), "history": [rec]}
    rec = log(it, "metrics", metrics=m, sec=r["sec"])
    return {"metrics": m, "error": None, "history": [rec]}

def recover(s: State) -> dict:
    it = s["iteration"]
    rec = log(it, "recovery", retry=s["retry_count"] + 1, error=s["error"][-500:])
    # TODO: feed s["error"] back into the implementation agent for a fix attempt
    return {"retry_count": s["retry_count"] + 1, "history": [rec]}

def abandon(s: State) -> dict:
    it = s["iteration"]
    rec = log(it, "abandon", hypothesis_id=s["hypothesis"]["id"], reason=s["error"][-500:])
     return {"error": None, "history": [rec]}

def update_best(s: State) -> dict:
    it, p = s["iteration"], s["metrics"]["valid"]["primary"]
    better = p > s["best_primary"]              # adopt any improvement
    significant = p > s["best_primary"] + EPS   # convergence counter uses ε
    streak = 0 if significant else s["no_improve_streak"] + 1
    rec = log(it, "decision", improved=better, significant=significant, primary=p,
              best=max(p, s["best_primary"]), streak=streak,
              delta_vs_baseline=round(p - BASELINE_VALID_PRIMARY, 4))
    out = {"no_improve_streak": streak, "history": [rec]}
    if better:
        out.update({"best_primary": p, "best_iteration": it, "best_workdir": s["workdir"]})
    return out

def finalize(s: State) -> dict:
    it = s["iteration"]
    if not s["best_workdir"]:
        rec = log(it, "finalize_failed", reason="no successful run")
        return {"done": True, "history": [rec]}
    out_csv = KIT_DIR / "submission_final.csv"
    r = kit.write_submission(Path(s["best_workdir"]), out_csv, "test")
    shutil.copytree(s["best_workdir"], KIT_DIR / "best_run", dirs_exist_ok=True)
    rec = log(it, "finalize", best_iteration=s["best_iteration"], best_valid_primary=s["best_primary"],
              submission=str(out_csv), check_ok=r["ok"], check_out=r["stdout"][-500:],
              human_interventions=s["human_interventions"])
    return {"done": True, "history": [rec]}

# ---------- routing ----------
def after_baseline(s) -> Literal["research", "finalize"]:
    return "research" if s["baseline_ok"] else "finalize"

def after_research(s) -> Literal["implement", "finalize"]:
    return "implement" if s["hypothesis"] else "finalize"

def after_validate(s) -> Literal["update_best", "recover", "abandon"]:
    if not s["error"]:
        return "update_best"
    return "recover" if s["retry_count"] < MAX_RETRY else "abandon"

def after_update(s) -> Literal["research", "finalize"]:
    if s["no_improve_streak"] >= N_CONVERGE or s["iteration"] >= MAX_ITERS:
        return "finalize"
    return "research"

def build():
    g = StateGraph(State)
    for n, f in [("reproduce_baseline", reproduce_baseline), ("research", research), ("implement", implement),
                 ("validate", validate), ("recover", recover), ("abandon", abandon),
                 ("update_best", update_best), ("finalize", finalize)]:
        g.add_node(n, f)
    g.set_entry_point("reproduce_baseline")
    g.add_conditional_edges("reproduce_baseline", after_baseline)
    g.add_conditional_edges("research", after_research)
    g.add_edge("implement", "validate")
    g.add_conditional_edges("validate", after_validate)
    g.add_edge("recover", "implement")
    g.add_conditional_edges("abandon", after_update)
    g.add_conditional_edges("update_best", after_update)
    g.add_edge("finalize", END)
    return g.compile()

INITIAL: State = {
    "iteration": 0, "baseline_ok": False, "hypothesis": None, "train_args": [], "workdir": None,
    "metrics": None, "error": None, "retry_count": 0, "best_primary": 0.0, "best_iteration": 0,
    "best_workdir": None, "no_improve_streak": 0, "human_interventions": 0, "history": [], "done": False,
}
