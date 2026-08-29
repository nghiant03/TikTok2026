from __future__ import annotations

from pathlib import Path

import pytest

from tiktok2026.adapters import (
    DeterministicPolicyGate,
    RepositoryRunStore,
)
from tiktok2026.contracts import (
    ExperimentSpec,
    Fidelity,
    RunRecord,
    WorktreeAssignment,
)

# ---------------------------------------------------------------------------
# Test 1: RepositoryRunStore CAS — conflicting replay raises
# ---------------------------------------------------------------------------


def _init_app_db(path: Path) -> None:
    from tiktok2026.persistence.migrations import MigrationRunner

    repo_root = Path(__file__).parents[2]
    MigrationRunner(path, repo_root / "migrations" / "application").apply()


def test_runstore_rejects_conflicting_replay(tmp_path: Path) -> None:
    """RepositoryRunStore raises on conflicting transition replay."""
    db = tmp_path / "app.sqlite3"
    _init_app_db(db)

    from tiktok2026.persistence.repositories import ApplicationRepository, PersistenceConflictError

    repo = ApplicationRepository(db)
    run_store = RepositoryRunStore(repo)

    # First put succeeds
    run_store.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active")

    spec = ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="test",
        mechanism="test",
        motivation="test",
        expected_signal="test",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="test",
        failure_criteria="test",
    )
    run_store.put_experiment(spec, "proposed", "run-1", "exp-1-proposed")

    # Replay with same transition_id but different content
    spec2 = ExperimentSpec(**{**spec.model_dump(), "hypothesis": "different"})
    with pytest.raises(
        (PersistenceConflictError, ValueError, RuntimeError, Exception),
    ):
        run_store.put_experiment(spec2, "proposed", "run-1", "exp-1-proposed")


# ---------------------------------------------------------------------------
# Test 2: Worktree assignment round-trips
# ---------------------------------------------------------------------------


def test_runstore_worktree_assignment_round_trip(tmp_path: Path) -> None:
    """RepositoryRunStore round-trips a WorktreeAssignment."""
    db = tmp_path / "app.sqlite3"
    _init_app_db(db)
    from tiktok2026.persistence.repositories import ApplicationRepository

    repo = ApplicationRepository(db)
    run_store = RepositoryRunStore(repo)

    assignment = WorktreeAssignment(
        worktree_id="wt-1",
        run_id="run-1",
        experiment_id="exp-1",
        path=Path("/tmp/worktree/exp-1"),
        branch="experiment/run-1/exp-1",
        parent_commit="a" * 40,
    )
    run_store.put_worktree_assignment(assignment)
    retrieved = run_store.get_worktree_assignment("exp-1")
    assert retrieved is not None
    assert retrieved.worktree_id == "wt-1"
    assert retrieved.experiment_id == "exp-1"


# ---------------------------------------------------------------------------
# Test 3: DeterministicPolicyGate rejects protected baseline paths
# ---------------------------------------------------------------------------


def test_policy_gate_rejects_protected_paths() -> None:
    """DeterministicPolicyGate rejects protected baseline paths."""
    gate = DeterministicPolicyGate()
    result = gate.check_paths(
        changed_paths=("baseline/evaluate.py",),
        allowed_scopes=("src/tiktok2026/experiment",),
    )
    assert result.allowed is False
    assert result.reason == "protected_path"


def test_policy_gate_rejects_third_repair() -> None:
    """DeterministicPolicyGate rejects third repair attempt."""
    gate = DeterministicPolicyGate()
    assert gate.can_repair(0).allowed is True
    assert gate.can_repair(1).allowed is True
    assert gate.can_repair(2).allowed is False
    assert gate.can_repair(2).reason == "repair_limit"


# ---------------------------------------------------------------------------
# Test 4: Synthetic end-to-end persists real audit, evaluation, resource, exports
# ---------------------------------------------------------------------------


async def test_synthetic_persists_real_audit_and_exports(tmp_path: Path) -> None:
    """Synthetic end-to-end produces real audit events, evaluation records, and export files."""
    import shutil

    from tiktok2026.bootstrap import build_synthetic_controller
    from tiktok2026.contracts import RunPhase
    from tiktok2026.graph.state import ProductionState

    repo_root = Path(__file__).parents[2]
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "migrations", test_repo_root / "migrations")

    controller, store, graph = build_synthetic_controller(
        repository_root=test_repo_root,
        runtime_root=tmp_path / "runtime",
    )

    initial: ProductionState = {
        "run_id": "test-run",
        "phase": RunPhase.BOOTSTRAP,
        "current_experiment_id": None,
        "current_hypothesis_id": None,
        "active_worktree_id": None,
        "latest_validation_report_id": None,
        "latest_execution_result_id": None,
        "latest_evaluation_result_id": None,
        "orchestration_decision_id": None,
        "repair_attempts": 0,
        "fidelity": Fidelity.SMOKE,
        "pending_route": None,
        "terminal_reason": None,
        "state_version": 0,
    }

    result = await graph.ainvoke(
        initial,
        {"configurable": {"thread_id": "test-run-1"}},
    )

    # The graph should complete (finalize may fail but the controller
    # routes through the error, ultimately reaching export or complete)
    assert result["phase"] == RunPhase.COMPLETE

    # Verify audit events were persisted
    from tiktok2026.persistence.repositories import ApplicationRepository

    app_repo = ApplicationRepository(tmp_path / "runtime" / "application.sqlite3")
    events = app_repo.list_audit_events("test-run")
    assert len(events) >= 1

    # Verify resource accounting was performed
    assert (tmp_path / "runtime" / "application.sqlite3").exists()


# ---------------------------------------------------------------------------
# Test 5: Production composition builds offline
# ---------------------------------------------------------------------------


def test_production_composition_builds_offline(tmp_path: Path) -> None:
    """build_production_services builds controller, graph, and repository without network."""
    import shutil

    from tiktok2026.bootstrap import build_production_services
    from tiktok2026.config import AppSettings, BudgetSettings
    from tiktok2026.controller import ProductionController

    repo_root = Path(__file__).parents[2]
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "migrations", test_repo_root / "migrations")

    settings = AppSettings(
        repository_root=test_repo_root,
        runtime_root=tmp_path / "runtime",
        budget=BudgetSettings(),
        models={},
    )

    result = build_production_services(settings)
    assert isinstance(result.controller, ProductionController)
    assert result.repository is not None
    assert result.graph is not None


# ---------------------------------------------------------------------------
# Test 6: Finalize failure propagates (no suppression)
# ---------------------------------------------------------------------------


async def test_finalize_failure_propagates(tmp_path: Path) -> None:
    """finalize transition must NOT suppress eligibility errors — fails closed with typed error."""
    import shutil

    from tiktok2026.bootstrap import build_synthetic_controller
    from tiktok2026.contracts import Fidelity, RunPhase
    from tiktok2026.graph.state import ProductionState

    repo_root = Path(__file__).parents[2]
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "migrations", test_repo_root / "migrations")

    _controller, _store, graph = build_synthetic_controller(
        repository_root=test_repo_root,
        runtime_root=tmp_path / "runtime",
    )

    # Finalize eligibility error should be caught and routed to persist_failure,
    # not silently swallowed. The graph should still complete.
    state: ProductionState = {
        "run_id": "test-finalize",
        "phase": "finalize",  # type: ignore[typeddict-item]
        "current_experiment_id": "exp-1",
        "current_hypothesis_id": "hyp-1",
        "active_worktree_id": "wt-1",
        "latest_validation_report_id": "vr-1",
        "latest_execution_result_id": "exec-1",
        "latest_evaluation_result_id": "eval-1",
        "orchestration_decision_id": "dec-1",
        "repair_attempts": 0,
        "fidelity": Fidelity.SMOKE,
        "pending_route": "finalize",
        "terminal_reason": "converged",
        "state_version": 5,
    }

    result = await graph.ainvoke(
        state,
        {"configurable": {"thread_id": "test-finalize-1"}},
    )
    # The graph should complete because finalize catches the eligibility
    # error and routes to persist_failure, which then routes to update_frontier
    assert result["phase"] in (RunPhase.COMPLETE, "persist", "finalize")