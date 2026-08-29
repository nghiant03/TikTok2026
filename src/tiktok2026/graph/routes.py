from tiktok2026.contracts import (
    DecisionAction,
    FailureKind,
    FailureRecord,
    OrchestrationDecision,
    ValidationReport,
    ValidationVerdict,
)
from tiktok2026.graph.state import ProductionState

REPAIRABLE_FAILURES = {
    FailureKind.SYNTAX_IMPORT,
    FailureKind.DEPENDENCY_ENVIRONMENT,
    FailureKind.MISSING_PATH,
    FailureKind.CUDA_OOM,
    FailureKind.CPU_OOM,
    FailureKind.NAN_DIVERGENCE,
    FailureKind.SCHEMA_MISMATCH,
    FailureKind.EVALUATOR_OUTPUT,
    FailureKind.TIMEOUT,
    FailureKind.DISK,
    FailureKind.CORRUPTED_CHECKPOINT,
}

ACTION_ROUTES = {
    DecisionAction.RESEARCH: "research",
    DecisionAction.IMPLEMENT: "implement",
    DecisionAction.VALIDATE: "proposal_validation",
    DecisionAction.REPAIR: "repair",
    DecisionAction.REPLICATE: "research",
    DecisionAction.INCREASE_FIDELITY: "research",
    DecisionAction.REVISIT_BRANCH: "research",
    DecisionAction.STOP: "finalize",
}


def route_after_failure(state: ProductionState, failure: FailureRecord) -> str:
    if failure.kind in REPAIRABLE_FAILURES and state["repair_attempts"] < 2:
        return "repair"
    # ``persist_failure`` is a persistence operation, not a retry edge.  Once
    # classified and persisted, a non-repairable/exhausted failure can only
    # leave the experiment through the finite export path.
    return "export"


def route_after_validation(state: ProductionState, report: ValidationReport) -> str:
    del state
    if report.verdict == ValidationVerdict.APPROVED:
        if report.stage.value == "proposal":
            return "create_worktree"
        if report.stage.value == "implementation":
            return "register_source"
        return "interpret"
    if report.verdict == ValidationVerdict.REPAIRABLE:
        return "repair"
    return "persist_failure"


def route_after_orchestration(decision: OrchestrationDecision) -> str:
    return ACTION_ROUTES[decision.action]


def route_after_frontier(state: ProductionState) -> str:
    return "finalize" if state["terminal_reason"] else "orchestrate"


def route_pending(state: ProductionState) -> str:
    route = state["pending_route"]
    if route is None:
        raise ValueError("controller transition did not provide a pending route")
    return route
