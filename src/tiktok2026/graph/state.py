from typing import TypedDict

from tiktok2026.contracts import Fidelity, RunPhase


class ProductionState(TypedDict):
    run_id: str
    phase: RunPhase
    current_experiment_id: str | None
    current_hypothesis_id: str | None
    active_worktree_id: str | None
    latest_validation_report_id: str | None
    latest_execution_result_id: str | None
    latest_evaluation_result_id: str | None
    orchestration_decision_id: str | None
    repair_attempts: int
    fidelity: Fidelity
    pending_route: str | None
    terminal_reason: str | None
    state_version: int
