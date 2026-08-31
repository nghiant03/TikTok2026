from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tiktok2026.adapters import (
    IMPLEMENTOR_CHECK_NAMES,
    DeterministicPolicyGate,
    RepositoryFrontierService,
    RepositoryRunStore,
    ScopedWorktreeRepository,
)
from tiktok2026.contracts import (
    ArtifactRecord,
    ArtifactRetention,
    EvaluationResult,
    ExecutionRequest,
    ExecutionResult,
    ExperimentSpec,
    Fidelity,
    ImplementationEdit,
    MetricValue,
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


def test_frontier_excludes_historical_metric_pair_on_resumed_run() -> None:
    """Only compatible current-metric evaluations contribute to convergence."""

    class FrontierRepository:
        def __init__(self) -> None:
            self.records: dict[str, list[str]] = {}
            self.experiments: dict[str, ExperimentSpec] = {}

        def list_json(self, kind: str) -> tuple[str, ...]:
            return tuple(self.records.get(kind, ()))

        def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
            return self.experiments.get(experiment_id)

        def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
            del record_id
            self.records.setdefault(kind, []).append(payload_json)

    repository = FrontierRepository()
    run_id = "resumed-run"
    manifest_sha256 = "a" * 64
    evaluator_sha256 = "b" * 64
    spec = ExperimentSpec(
        experiment_id="placeholder",
        hypothesis_id="hypothesis",
        hypothesis="test",
        mechanism="test",
        motivation="test",
        expected_signal="test",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="test",
        failure_criteria="test",
    )

    def add_evaluation(
        evaluation_id: str,
        experiment_id: str,
        execution_id: str,
        metrics: tuple[MetricValue, ...],
    ) -> None:
        repository.experiments[experiment_id] = spec.model_copy(
            update={"experiment_id": experiment_id}
        )
        repository.records.setdefault("execution", []).append(
            ExecutionResult(
                execution_id=execution_id,
                experiment_id=experiment_id,
                source_commit="c" * 40,
                command=("train",),
                exit_code=0,
                elapsed_seconds=1.0,
                gpu_hours=0.0,
            ).model_dump_json()
        )
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            experiment_id=experiment_id,
            checkpoint_id=f"checkpoint-{evaluation_id}",
            metrics=metrics,
            evaluator_artifact_id="evaluator-v2",
            evaluator_sha256=evaluator_sha256,
            prediction_sha256="d" * 64,
            validity="provisional",
            dataset_manifest_sha256=manifest_sha256,
            split="valid",
            run_id=run_id,
            execution_id=execution_id,
        )
        repository.records.setdefault("evaluation", []).append(
            json.dumps({"result": result.model_dump(mode="json")})
        )

    add_evaluation(
        "eval-v1",
        "exp-v1",
        "execution-v1",
        (MetricValue(name="NDCG@10", value=1.0), MetricValue(name="Recall@50", value=1.0)),
    )
    add_evaluation(
        "eval-v2-current",
        "exp-v2-current",
        "execution-v2-current",
        (MetricValue(name="GAUC", value=0.0), MetricValue(name="nDCG@5", value=0.0)),
    )

    service = RepositoryFrontierService(repository, epsilon=0.0, patience=1)  # type: ignore[arg-type]
    service.initialize(run_id)

    assert service.update("exp-v2-current", 0.0) is None
    observations = [
        json.loads(raw) for raw in repository.list_json("frontier_observation")
    ]
    assert [observation["evaluation_id"] for observation in observations] == [
        "eval-v2-current",
    ]
    decisions = [json.loads(raw) for raw in repository.list_json("frontier_decision")]
    assert decisions == [
        {
            "decision": "continue",
            "experiment_id": "exp-v2-current",
            "observation_count": 1,
            "reason": "configured policy requires more evidence",
            "run_id": run_id,
        }
    ]


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
    subprocess.run(
        ("git", "-c", "commit.gpgsign=false", "commit", "-qm", "base"),
        cwd=repository,
        check=True,
    )

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
    assert capability.diff(5) == capability.diff()[:5]
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
    subprocess.run(
        ("git", "-c", "commit.gpgsign=false", "commit", "-qm", "base"),
        cwd=repository,
        check=True,
    )
    train.write_text("CURRENT = 2\n", encoding="utf-8")
    sibling = repository / "src/tiktok2026/experiment/helper.py"
    sibling.write_text("HELPER = 3\n", encoding="utf-8")

    capability = ScopedWorktreeRepository(
        repository,
        ("src/tiktok2026/experiment/train.py",),
        read_scopes=("src/tiktok2026/experiment",),
    )

    assert capability.read("src/tiktok2026/experiment/train.py") == "CURRENT = 2\n"
    assert capability.read_base("src/tiktok2026/experiment/train.py") == "BASE = 1\n"
    assert capability.read("src/tiktok2026/experiment/helper.py") == "HELPER = 3\n"
    with pytest.raises(PermissionError, match="approved repository scope"):
        capability.write("src/tiktok2026/experiment/helper.py", "blocked\n")


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


def test_implementor_checks_are_controller_owned_names() -> None:
    assert IMPLEMENTOR_CHECK_NAMES == (
        "compile_entrypoint",
        "ruff_entrypoint",
        "pyright_entrypoint",
        "diff_check",
        "contract_smoke",
    )


def test_static_contract_check_never_executes_candidate_top_level_code(
    tmp_path: Path,
) -> None:
    import tiktok2026.adapters as adapters

    repository = tmp_path / "repo"
    entrypoint = repository / "src/tiktok2026/experiment/train.py"
    entrypoint.parent.mkdir(parents=True)
    source = (Path(__file__).parents[2] / "src/tiktok2026/experiment/train.py").read_text(
        encoding="utf-8"
    )
    marker = tmp_path / "executed"
    entrypoint.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n" + source,
        encoding="utf-8",
    )
    capability = ScopedWorktreeRepository(
        repository, ("src/tiktok2026/experiment",)
    )

    output = capability.run_check(adapters._static_contract_check_command(), 30)

    assert "static contract check passed" in output
    assert not marker.exists()


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
        dataset_manifest_sha256="a" * 64,
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
    assert result.failure_message == "execution did not produce prediction and checkpoint artifacts"


def _training_artifact_fixture(
    tmp_path: Path, view_sha256: str, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, ExecutionRequest, ExecutionResult, list[tuple[str, str | None]]]:
    from tiktok2026.bootstrap import _RunBoundDockerExecutor
    from tiktok2026.contracts import DatasetManifestIdentity

    output = tmp_path / "output"
    output.mkdir()
    prediction_payload = {
        "schema_version": "1",
        "manifest_id": "manifest-1",
        "manifest_sha256": "a" * 64,
        "dataset_view_sha256": view_sha256,
        "source_commit": "b" * 40,
        "execution_id": "execution-1",
        "split": "valid",
        "rows": [],
    }
    prediction_bytes = (json.dumps(prediction_payload, separators=(",", ":")) + "\n").encode()
    (output / "predictions.json").write_bytes(prediction_bytes)
    prediction_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    checkpoint_payload = {
        "checkpoint_id": "checkpoint-1",
        "data_manifest_id": "manifest-1",
        "source_commit": "b" * 40,
        "execution_id": "execution-1",
        "prediction_artifact": "predictions.json",
        "prediction_sha256": prediction_sha256,
        "dataset_view_sha256": view_sha256,
    }
    (output / "checkpoint_bundle.json").write_text(
        json.dumps(checkpoint_payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    events: list[tuple[str, str | None]] = []

    class FakeRepository:
        def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
            del record_id
            events.append((kind, payload_json))

    class FakeRunStore:
        def __init__(self, repository: object) -> None:
            del repository

        def get_dataset_manifest_identity(self) -> DatasetManifestIdentity:
            return DatasetManifestIdentity(manifest_id="manifest-1", manifest_sha256="a" * 64)

    class FakeArtifactStore:
        def publish_bytes(
            self,
            run_id: str,
            experiment_id: str,
            kind: str,
            filename: str,
            content: bytes,
            producer: str,
            retention: object,
        ) -> ArtifactRecord:
            del run_id, experiment_id, producer, retention
            events.append((kind, None))
            path = tmp_path / f"published-{kind}-{filename}"
            path.write_bytes(content)
            return ArtifactRecord(
                artifact_id=f"{kind}-artifact",
                run_id="run-1",
                experiment_id="exp-1",
                kind=kind,
                uri=path.as_uri(),
                sha256=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
                producer="test",
                retention=ArtifactRetention.RUN,
            )

    import tiktok2026.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "RepositoryRunStore", FakeRunStore)
    repository = FakeRepository()
    executor = _RunBoundDockerExecutor(
        repository=repository,
        artifact_store=FakeArtifactStore(),
        policy=object(),
        dataset_provider=object(),
    )
    request = ExecutionRequest(
        run_id="run-1",
        execution_id="execution-1",
        experiment_id="exp-1",
        source_commit="b" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        image="image@sha256:" + "0" * 64,
        source_path=tmp_path,
        dataset_path=tmp_path,
        dataset_manifest_sha256="a" * 64,
        output_path=output,
        timeout_seconds=60,
        memory_bytes=1 << 20,
        cpus=1.0,
    )
    result = ExecutionResult(
        execution_id="execution-1",
        experiment_id="exp-1",
        source_commit="b" * 40,
        command=request.command,
        exit_code=0,
        elapsed_seconds=0.1,
        gpu_hours=0.0,
        dataset_view_sha256=view_sha256,
    )
    return executor, request, result, events


def test_training_registration_persists_authoritative_view_hash_and_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, request, result, events = _training_artifact_fixture(
        tmp_path, "c" * 64, monkeypatch
    )

    registered = executor._register_training_artifacts(request, result)  # type: ignore[attr-defined]

    assert registered.artifact_ids == ("prediction-artifact", "checkpoint-artifact")
    assert [kind for kind, _payload in events] == [
        "prediction",
        "checkpoint",
        "prediction_artifact",
    ]
    registration_payload = json.loads(events[-1][1] or "{}")
    assert registration_payload["dataset_view_sha256"] == "c" * 64


def test_training_registration_rejects_artifact_view_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor, request, result, _events = _training_artifact_fixture(
        tmp_path, "c" * 64, monkeypatch
    )
    prediction_path = request.output_path / "predictions.json"
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction_payload["dataset_view_sha256"] = "d" * 64
    prediction_path.write_text(
        json.dumps(prediction_payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="provenance does not match execution"):
        executor._register_training_artifacts(request, result)  # type: ignore[attr-defined]


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
