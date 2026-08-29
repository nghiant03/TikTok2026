from tiktok2026.contracts import (
    DecisionAction,
    FailureKind,
    FailureRecord,
    Fidelity,
    OrchestrationDecision,
    RunPhase,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
)
from tiktok2026.graph.routes import (
    route_after_failure,
    route_after_frontier,
    route_after_orchestration,
    route_after_validation,
)
from tiktok2026.graph.state import ProductionState


def state(**overrides: object) -> ProductionState:
    values: ProductionState = {
        "run_id": "run-1",
        "phase": RunPhase.EXECUTE,
        "current_experiment_id": "experiment-1",
        "current_hypothesis_id": "hypothesis-1",
        "active_worktree_id": "worktree-1",
        "latest_validation_report_id": None,
        "latest_execution_result_id": None,
        "latest_evaluation_result_id": None,
        "orchestration_decision_id": None,
        "repair_attempts": 0,
        "fidelity": Fidelity.SMOKE,
        "pending_route": None,
        "terminal_reason": None,
        "state_version": 1,
    }
    values.update(overrides)  # type: ignore[typeddict-item]
    return values


def repairable_failure(attempt: int) -> FailureRecord:
    return FailureRecord(
        failure_id=f"failure-{attempt}",
        experiment_id="experiment-1",
        kind=FailureKind.SYNTAX_IMPORT,
        evidence_refs=("execution-1",),
        repair_attempt=attempt,
    )


def test_graph_state_contains_references_not_artifacts() -> None:
    assert set(ProductionState.__annotations__) == {
        "run_id",
        "phase",
        "current_experiment_id",
        "current_hypothesis_id",
        "active_worktree_id",
        "latest_validation_report_id",
        "latest_execution_result_id",
        "latest_evaluation_result_id",
        "orchestration_decision_id",
        "repair_attempts",
        "fidelity",
        "pending_route",
        "terminal_reason",
        "state_version",
    }


def test_repairable_failure_routes_to_repair_until_limit() -> None:
    assert route_after_failure(state(repair_attempts=1), repairable_failure(1)) == "repair"
    assert route_after_failure(state(repair_attempts=2), repairable_failure(2)) == "persist_failure"


def test_nonrepairable_failure_is_persisted() -> None:
    failure = FailureRecord(
        failure_id="failure-science",
        experiment_id="experiment-1",
        kind=FailureKind.SCIENTIFIC_NON_IMPROVEMENT,
        evidence_refs=("evaluation-1",),
        repair_attempt=0,
        scientific_evidence=True,
    )

    assert route_after_failure(state(), failure) == "persist_failure"


def test_approved_result_routes_to_interpretation() -> None:
    report = ValidationReport(
        report_id="report-1",
        experiment_id="experiment-1",
        stage=ValidationStage.RESULT,
        verdict=ValidationVerdict.APPROVED,
        leakage_risk="none",
    )

    assert route_after_validation(state(), report) == "interpret"


def test_stop_decision_routes_to_finalization() -> None:
    decision = OrchestrationDecision(
        decision_id="decision-1",
        action=DecisionAction.STOP,
        rationale="budget exhausted",
    )

    assert route_after_orchestration(decision) == "finalize"
    assert route_after_frontier(state(terminal_reason="converged")) == "finalize"


def test_frontier_update_cycles_to_orchestration() -> None:
    assert route_after_frontier(state(phase=RunPhase.PERSIST)) == "orchestrate"
