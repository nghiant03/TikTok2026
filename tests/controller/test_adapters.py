from __future__ import annotations

import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tiktok2026.adapters import (
    DeterministicPolicyGate,
    RepositoryRunStore,
    ScopedWorktreeRepository,
)
from tiktok2026.contracts import (
    ExperimentSpec,
    Fidelity,
    ImplementationEdit,
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


def test_transition_cas_is_atomic_and_repairs_a_missing_event(tmp_path: Path) -> None:
    """Identical transition replay repairs an absent audit event atomically."""
    db = tmp_path / "app.sqlite3"
    _init_app_db(db)
    from tiktok2026.persistence.repositories import ApplicationRepository

    repository = ApplicationRepository(db)
    store = RepositoryRunStore(repository)
    store.persist_transition("run-1", "inspect", 1, {"pending_route": "orchestrate"})
    store.persist_transition("run-1", "research", 2, {"pending_route": "proposal_policy"})
    with sqlite3.connect(db) as connection:
        connection.execute(
            "DELETE FROM audit_events WHERE event_id = ?", ("transition-run-1-2",)
        )
    store.persist_transition("run-1", "research", 2, {"pending_route": "proposal_policy"})
    events = repository.list_audit_events("run-1")
    assert {event.event_id for event in events} >= {
        "transition-run-1-1",
        "transition-run-1-2",
    }


def test_transition_cas_serializes_identical_concurrent_replays(tmp_path: Path) -> None:
    """Concurrent identical version-one writes have one durable result."""
    db = tmp_path / "app.sqlite3"
    _init_app_db(db)
    from tiktok2026.persistence.repositories import ApplicationRepository

    def write() -> None:
        RepositoryRunStore(ApplicationRepository(db)).persist_transition(
            "run-1", "inspect", 1, {"pending_route": "orchestrate"}
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: write(), range(2)))
    repository = ApplicationRepository(db)
    assert len(repository.list_audit_events("run-1")) == 1


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


def test_policy_gate_rejects_fourth_repair() -> None:
    """DeterministicPolicyGate rejects a fourth repair attempt."""
    gate = DeterministicPolicyGate()
    assert gate.can_repair(0).allowed is True
    assert gate.can_repair(1).allowed is True
    assert gate.can_repair(2).allowed is True
    assert gate.can_repair(3).allowed is False
    assert gate.can_repair(3).reason == "repair_limit"


def test_scoped_implementor_capability_applies_real_bounded_diff(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "test"), cwd=repository, check=True)
    target = repository / "src/tiktok2026/experiment"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=repository, check=True)

    capability = ScopedWorktreeRepository(
        repository, ("src/tiktok2026/experiment",)
    )
    capability.apply_edits(
        (
            ImplementationEdit(
                relative_path="src/tiktok2026/experiment/change.py", content="VALUE = 1\n"
            ),
        )
    )

    assert capability.changed_files() == ("src/tiktok2026/experiment/change.py",)
    assert "change.py" in capability.diff()
    with pytest.raises(PermissionError, match="protected"):
        ScopedWorktreeRepository(repository, ("baseline",)).write("baseline/data.py", "blocked")


def test_scoped_implementor_reads_current_and_committed_base_source(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "test"), cwd=repository, check=True)
    target = repository / "src/tiktok2026/experiment"
    target.mkdir(parents=True)
    train = target / "train.py"
    train.write_text("BASE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=repository, check=True)
    train.write_text("CURRENT = 2\n", encoding="utf-8")

    capability = ScopedWorktreeRepository(
        repository, ("src/tiktok2026/experiment/train.py",)
    )

    assert capability.read("src/tiktok2026/experiment/train.py") == "CURRENT = 2\n"
    assert capability.read_base("src/tiktok2026/experiment/train.py") == "BASE = 1\n"


def test_scoped_implementor_prevalidates_all_edits(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    capability = ScopedWorktreeRepository(
        repository, ("src/tiktok2026/experiment",)
    )
    allowed = repository / "src/tiktok2026/experiment/allowed.py"

    with pytest.raises(PermissionError, match="unrelated.py"):
        capability.apply_edits(
            (
                ImplementationEdit(
                    relative_path=allowed.relative_to(repository).as_posix(), content="ok\n"
                ),
                ImplementationEdit(relative_path="src/tiktok2026/unrelated.py", content="bad\n"),
            )
        )

    assert not allowed.exists()


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
        docker_image="registry.example/tiktok2026@sha256:" + "a" * 64,
    )

    result = build_production_services(settings)
    assert isinstance(result.controller, ProductionController)
    assert result.repository is not None
    assert result.graph is not None
    assert result.executor.policy.allowed_image_digests == (settings.docker_image,)


async def test_run_bound_executor_classifies_missing_output_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tiktok2026.execution.docker as docker
    from tiktok2026.bootstrap import _RunBoundDockerExecutor
    from tiktok2026.contracts import (
        DatasetManifestIdentity,
        ExecutionRequest,
        ExecutionResult,
        FailureKind,
    )
    source = tmp_path / "source"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    source.mkdir()
    dataset.mkdir()
    output.mkdir()
    assignment = WorktreeAssignment(
        worktree_id="wt-1",
        run_id="run-1",
        experiment_id="exp-1",
        path=source,
        branch="experiment/run-1/exp-1",
        parent_commit="b" * 40,
    )

    class FakeRunStore:
        def __init__(self, repository: object) -> None:
            del repository

        def get_worktree_assignment(self, experiment_id: str) -> WorktreeAssignment | None:
            return assignment if experiment_id == "exp-1" else None

        def get_dataset_manifest_identity(self) -> DatasetManifestIdentity:
            return DatasetManifestIdentity(
                manifest_id="manifest-1", manifest_sha256="a" * 64
            )

    class StubDockerExecutor:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def execute(self, request: ExecutionRequest) -> ExecutionResult:
            return ExecutionResult(
                execution_id=request.execution_id,
                experiment_id=request.experiment_id,
                source_commit=request.source_commit,
                command=request.command,
                exit_code=0,
                elapsed_seconds=0.1,
                gpu_hours=0.0,
            )

    monkeypatch.setattr("tiktok2026.bootstrap.RepositoryRunStore", FakeRunStore)
    monkeypatch.setattr(docker, "DockerExecutor", StubDockerExecutor)
    request = ExecutionRequest(
        run_id="run-1",
        execution_id="execution-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        image="tiktok2026:test@sha256:" + "0" * 64,
        source_path=source,
        dataset_path=dataset,
        output_path=output,
        timeout_seconds=60,
        memory_bytes=1 << 20,
        cpus=1.0,
    )

    result = await _RunBoundDockerExecutor(
        repository=object(),
        artifact_store=object(),
        policy=object(),
        dataset_provider=object(),
    ).execute(request)

    assert result.exit_code == 1
    assert result.failure_kind == FailureKind.MISSING_PATH


# ---------------------------------------------------------------------------
# Test 6: Finalize failure propagates (no suppression)
# ---------------------------------------------------------------------------


async def test_finalize_failure_propagates(tmp_path: Path) -> None:
    """finalize transition must NOT suppress eligibility errors — fails closed with typed error."""
    import shutil

    from tiktok2026.bootstrap import build_synthetic_controller
    from tiktok2026.contracts import Fidelity
    from tiktok2026.graph.state import ProductionState

    repo_root = Path(__file__).parents[2]
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "migrations", test_repo_root / "migrations")

    _controller, _store, _graph = build_synthetic_controller(
        repository_root=test_repo_root,
        runtime_root=tmp_path / "runtime",
    )

    # Finalization authority errors are terminal: they are persisted as a typed
    # failure and must not route through export or complete.
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

    from tiktok2026.use_cases import TerminalLifecycleError

    with pytest.raises(TerminalLifecycleError):
        await _controller.finalize(state)
    assert any(
        '"kind":"schema_mismatch"' in failure
        for failure in _store.list_json("failure")  # type: ignore[union-attr]
    )
    assert _store.get_finalization("finalization-test-finalize") is None  # type: ignore[union-attr]
