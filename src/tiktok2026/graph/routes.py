from tiktok2026.contracts import (
    DecisionAction,
    FailureKind,
    FailureRecord,
    OrchestrationDecision,
    RunPhase,
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

REPAIRABLE_PHASES = {
    RunPhase.RESEARCH,
    RunPhase.IMPLEMENT,
    RunPhase.EXECUTE,
    RunPhase.EVALUATE,
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


def route_after_failure(
    state: ProductionState,
    failure: FailureRecord,
    max_repairs: int = 2,
) -> str:
    if (
        failure.kind in REPAIRABLE_FAILURES
        and state.get("current_experiment_id") is not None
        and state["phase"] in REPAIRABLE_PHASES
        and state["repair_attempts"] < max_repairs
    ):
        return "repair"
    if (
        failure.kind in REPAIRABLE_FAILURES | {FailureKind.UNSTABLE_VALIDATION}
        and state.get("current_experiment_id") is not None
        and state["phase"] in REPAIRABLE_PHASES
    ):
        return "orchestrate"
    return "terminal"


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
