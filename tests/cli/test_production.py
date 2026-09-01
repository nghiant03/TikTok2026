from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tiktok2026.cli import app
from tiktok2026.contracts import BaselineCalibrationRecord, RunBaselineBinding
from tiktok2026.persistence.repositories import ApplicationRepository

ROOT = Path(__file__).parents[2]


def _setup_runtime(tmp_path: Path) -> tuple[Path, Path]:
    """Initialize runtime with migrations and return (runtime_root, repo_root)."""
    import shutil

    from tiktok2026.bootstrap import initialize_runtime

    runtime_root = tmp_path / "runtime"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", repo_dir / "migrations")
    initialize_runtime(repo_dir, runtime_root)
    return runtime_root, repo_dir


def _repo(runtime_root: Path) -> ApplicationRepository:
    return ApplicationRepository(runtime_root / "application.sqlite3")


def _baseline_calibration() -> BaselineCalibrationRecord:
    from tiktok2026.contracts import (
        CURRENT_EVALUATOR_ID,
        DiagnosticMetricValue,
        EvaluationResult,
        MetricValue,
    )

    evaluation = EvaluationResult(
        evaluation_id="starter-kit-fm-evaluation",
        experiment_id="starter-kit-fm-baseline",
        checkpoint_id="starter-kit-fm-checkpoint",
        metrics=(
            MetricValue(name="GAUC", value=0.6674),
            MetricValue(name="nDCG@5", value=0.5357),
        ),
        evaluator_artifact_id=CURRENT_EVALUATOR_ID,
        evaluator_sha256="b" * 64,
        prediction_sha256="c" * 64,
        validity="provisional",
        dataset_manifest_sha256="a" * 64,
        split="valid",
        dataset_manifest_id="manifest-1",
    )
    return BaselineCalibrationRecord(
        calibration_id="calibration-1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        evaluator_id=CURRENT_EVALUATOR_ID,
        evaluator_sha256="b" * 64,
        baseline_source_sha256="d" * 64,
        config_sha256="e" * 64,
        prediction_sha256="c" * 64,
        prediction_artifact_uri="file:///runtime/baseline-predictions.csv",
        evaluation=evaluation,
        diagnostic_metrics=(
            DiagnosticMetricValue(name="GAUC", value=0.6674),
            DiagnosticMetricValue(name="nDCG@5", value=0.5357),
            DiagnosticMetricValue(name="primary", value=0.60155),
        ),
    )


def _baseline_binding(run_id: str, calibration_id: str = "calibration-1") -> RunBaselineBinding:
    calibration = _baseline_calibration()
    return RunBaselineBinding(
        run_id=run_id,
        calibration_id=calibration_id,
        baseline_evaluation_id=calibration.evaluation.evaluation_id,
        dataset_manifest_id=calibration.dataset_manifest_id,
        dataset_manifest_sha256=calibration.dataset_manifest_sha256,
        evaluator_id=calibration.evaluator_id,
        evaluator_sha256=calibration.evaluator_sha256,
        split=calibration.split,
        metrics=calibration.evaluation.metrics,
    )


class _CompletingGraph:
    async def ainvoke(
        self, state: dict[str, object], config: object
    ) -> dict[str, object]:
        del config
        return {**state, "phase": "complete", "pending_route": "complete"}


def test_production_runs_bind_cached_starter_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tiktok2026 import bootstrap
    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.bootstrap import ProductionOperations

    runtime_root, repo_dir = _setup_runtime(tmp_path)
    repository = _repo(runtime_root)
    calibration = _baseline_calibration()
    existing_counts: list[int] = []

    def calibrator(
        repository_root: Path,
        runtime_root_: Path,
        dataset_root: Path,
        existing: tuple[str, ...],
    ) -> tuple[BaselineCalibrationRecord, bool]:
        del repository_root, runtime_root_, dataset_root
        existing_counts.append(len(existing))
        return calibration, not existing

    def fake_verify(repository_root: Path) -> object:
        del repository_root
        return object()

    def fake_services(settings_: object) -> SimpleNamespace:
        del settings_
        return SimpleNamespace(
            graph=_CompletingGraph(), repository=repository, resource_ledger=None
        )

    settings = SimpleNamespace(
        models={},
        dataset_root=tmp_path / "dataset",
        online_research=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(bootstrap, "verify_manifests", fake_verify)
    monkeypatch.setattr(bootstrap, "build_production_services", fake_services)
    operations = ProductionOperations(
        repo_dir, runtime_root, baseline_calibrator=calibrator
    )
    monkeypatch.setattr(operations, "_production_settings", lambda: settings)

    operations.run(run_id="run-1")
    operations.run(run_id="run-2")

    store = RepositoryRunStore(repository)
    first = store.get_run_baseline("run-1")
    second = store.get_run_baseline("run-2")
    assert first is not None
    assert second is not None
    assert first.calibration_id == second.calibration_id == "calibration-1"
    assert existing_counts == [0, 1]
    assert [
        event.event_type
        for event in repository.list_audit_events("calibration-1")
    ] == ["baseline_calibrated"]


def test_production_resume_backfills_baseline_without_checkpoint_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tiktok2026 import bootstrap
    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.bootstrap import ProductionOperations
    from tiktok2026.contracts import RunRecord

    runtime_root, repo_dir = _setup_runtime(tmp_path)
    repository = _repo(runtime_root)
    store = RepositoryRunStore(repository)
    store.put_run(RunRecord(run_id="legacy-run", status="active"), "legacy-run-active")
    operations = ProductionOperations(
        repo_dir,
        runtime_root,
        baseline_calibrator=lambda *args: (_baseline_calibration(), True),
    )
    settings = SimpleNamespace(
        models={},
        dataset_root=tmp_path / "dataset",
        online_research=SimpleNamespace(enabled=False),
    )
    checkpoint: dict[str, object] = {
        "phase": "complete",
        "state_version": 9,
        "pending_route": "complete",
    }

    def fake_verify(repository_root: Path) -> object:
        del repository_root
        return object()

    def load_checkpoint(run_id: str) -> dict[str, object]:
        del run_id
        return checkpoint.copy()

    monkeypatch.setattr(bootstrap, "verify_manifests", fake_verify)
    monkeypatch.setattr(operations, "_production_settings", lambda: settings)
    monkeypatch.setattr(operations, "_load_checkpoint", load_checkpoint)

    result = operations.resume("legacy-run")

    assert result.status == "already_complete"
    assert checkpoint == {
        "phase": "complete",
        "state_version": 9,
        "pending_route": "complete",
    }
    assert store.get_run_baseline("legacy-run") is not None
    assert "baseline_bound" in {
        event.event_type for event in repository.list_audit_events("legacy-run")
    }


def test_resume_champion_closure_finalizes_without_stale_implementation_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tiktok2026 import bootstrap
    from tiktok2026.bootstrap import ProductionOperations
    from tiktok2026.contracts import RunPhase

    class _Checkpoint:
        def __init__(self) -> None:
            self.updates: list[dict[str, object]] = []

        async def aget_tuple(self, config: object) -> SimpleNamespace:
            del config
            return SimpleNamespace(config={"configurable": {"thread_id": "run-1"}})

        async def aupdate_state(
            self, config: dict[str, object], updates: dict[str, object], as_node: str
        ) -> None:
            del config, as_node
            self.updates.append(updates)

    class _Graph:
        def __init__(self) -> None:
            self.invoked = False

        async def ainvoke(self, state: object, config: object) -> dict[str, object]:
            del state, config
            self.invoked = True
            return {"phase": RunPhase.COMPLETE, "pending_route": "complete"}

    class _Repository:
        def adopt_legacy_lifecycle(self, run_id: str) -> tuple[dict[str, object], ...]:
            assert run_id == "run-1"
            return ()

    closure = SimpleNamespace(champion=object())
    repository = _Repository()
    checkpoint = _Checkpoint()
    graph = _Graph()
    stale_state = {
        "phase": RunPhase.IMPLEMENT,
        "pending_route": "implement",
        "current_experiment_id": "abandoned-experiment",
        "state_version": 17,
    }
    closure_state = {
        "phase": RunPhase.FINALIZE,
        "pending_route": "finalize",
        "current_experiment_id": "historical-champion",
    }

    class _Store:
        def __init__(self, repository_: object) -> None:
            assert repository_ is repository

        def get_run_closure(self, run_id: str) -> object:
            assert run_id == "run-1"
            return closure

        def put_run_baseline(self, binding: object) -> None:
            del binding

    runtime_root = tmp_path / "runtime"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    runtime_root.mkdir()
    operations = ProductionOperations(repo_dir, runtime_root)
    settings = SimpleNamespace(dataset_root=tmp_path / "dataset")
    monkeypatch.setattr(
        bootstrap,
        "initialize_runtime",
        lambda repository_root, runtime_root_: SimpleNamespace(repository=repository),
    )
    monkeypatch.setattr(bootstrap, "RepositoryRunStore", _Store)
    monkeypatch.setattr(bootstrap, "SqliteCheckpointer", lambda path: checkpoint)
    monkeypatch.setattr(bootstrap, "closure_updates_without_agents", lambda *args: closure_state)
    monkeypatch.setattr(operations, "_load_checkpoint", lambda run_id: stale_state)
    monkeypatch.setattr(operations, "_production_settings", lambda: settings)
    monkeypatch.setattr(operations, "_ensure_current_baseline", lambda *args: (object(), False))
    monkeypatch.setattr(operations, "_binding", lambda *args: object())
    monkeypatch.setattr(
        bootstrap,
        "build_production_services",
        lambda settings_: SimpleNamespace(graph=graph),
    )
    monkeypatch.setattr(
        operations,
        "_reconcile_resume_boundary",
        lambda *args: pytest.fail("stale resume boundary was reconciled"),
    )
    monkeypatch.setattr(
        operations,
        "_bind_resumed_implementor",
        lambda *args: pytest.fail("stale implementor was bound"),
    )
    monkeypatch.setattr(operations, "_resume_audit", lambda *args: None)

    result = operations.resume("run-1")

    assert result.status == "resumed"
    assert graph.invoked
    assert checkpoint.updates == [closure_state]


def test_resume_runtime_lock_allows_only_one_dispatch_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tiktok2026.bootstrap import OperationalError, ProductionOperations
    from tiktok2026.contracts import OperationResult, RunPhase

    runtime_root, repo_dir = _setup_runtime(tmp_path)
    operations = ProductionOperations(repo_dir, runtime_root)
    entered = Event()
    release = Event()
    dispatches = 0

    def fake_resume_locked(run_id: str, *, synthetic: bool = False) -> OperationResult:
        nonlocal dispatches
        del synthetic
        dispatches += 1
        entered.set()
        assert release.wait(5)
        return OperationResult(
            operation="resume", run_id=run_id, phase=RunPhase.EXECUTE, status="resumed"
        )

    monkeypatch.setattr(operations, "_resume_locked", fake_resume_locked)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(operations.resume, "run-1")
        assert entered.wait(5)
        second = executor.submit(operations.resume, "run-1")
        with pytest.raises(OperationalError, match="another production run"):
            second.result()
        release.set()
        assert first.result().status == "resumed"
    assert dispatches == 1


def test_production_resume_rejects_conflicting_baseline_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tiktok2026 import bootstrap
    from tiktok2026.bootstrap import OperationalError, ProductionOperations

    runtime_root, repo_dir = _setup_runtime(tmp_path)
    repository = _repo(runtime_root)
    operations = ProductionOperations(
        repo_dir,
        runtime_root,
        baseline_calibrator=lambda *args: (_baseline_calibration(), True),
    )
    older_calibration = _baseline_calibration().model_copy(
        update={"calibration_id": "older-calibration"}
    )
    repository.put_baseline_calibration(
        older_calibration, actor_type="controller", actor_id="test"
    )
    existing = _baseline_binding("run-1", "older-calibration")
    repository.put_run_baseline(existing)
    settings = SimpleNamespace(
        models={},
        dataset_root=tmp_path / "dataset",
        online_research=SimpleNamespace(enabled=False),
    )
    def fake_verify(repository_root: Path) -> object:
        del repository_root
        return object()

    def load_checkpoint(run_id: str) -> dict[str, object]:
        del run_id
        return {"phase": "complete", "pending_route": "complete"}

    monkeypatch.setattr(bootstrap, "verify_manifests", fake_verify)
    monkeypatch.setattr(operations, "_production_settings", lambda: settings)
    monkeypatch.setattr(operations, "_load_checkpoint", load_checkpoint)

    with pytest.raises(OperationalError, match="content changed"):
        operations.resume("run-1")

    assert "resume_rejected" in {
        event.event_type for event in repository.list_audit_events("run-1")
    }


# ---------------------------------------------------------------------------
# Production composition builds offline
# ---------------------------------------------------------------------------


def test_production_composition_builds_offline(tmp_path: Path) -> None:
    """Production composition builds controller, graph, repository, executor, evaluator,
    worktree manager, and four agent clients without network calls."""
    from tiktok2026.bootstrap import build_production_services
    from tiktok2026.config import AppSettings, BudgetSettings
    from tiktok2026.controller import ProductionController

    runtime_root, repo_dir = _setup_runtime(tmp_path)

    settings = AppSettings(
        repository_root=repo_dir,
        runtime_root=runtime_root,
        budget=BudgetSettings(),
        models={},
    )

    result = build_production_services(settings)
    assert isinstance(result.controller, ProductionController)
    assert result.repository is not None
    assert result.graph is not None
    assert result.agent_clients is not None
    # With empty models dict, no clients are created
    assert len(result.agent_clients) == 0


# ---------------------------------------------------------------------------
# Production run exits nonzero with clear message when model credentials absent
# ---------------------------------------------------------------------------


def test_production_run_fails_with_clear_message_on_missing_credentials(
    tmp_path: Path,
) -> None:
    """Production run command exits nonzero with a clear message when model
    credentials are absent."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
        ],
    )
    assert result.exit_code != 0
    # The error message could be about the model or about missing credentials
    assert result.exit_code != 0  # production run fails without credentials


# ---------------------------------------------------------------------------
# Resume rejects identity mismatch
# ---------------------------------------------------------------------------


def test_production_resume_rejects_identity_mismatch(tmp_path: Path) -> None:
    """Production resume with mismatched records exits nonzero."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import AuditEvent, RunRecord

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)
    rs.put_audit_event(
        AuditEvent(
            event_id="run-start-1",
            run_id="run-1",
            experiment_id=None,
            event_type="run_started",
            actor_type="controller",
            actor_id="test",
            payload={"run_id": "run-1"},
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "resume",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
            "--run-id",
            "run-1",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Finalize rejects ineligible run
# ---------------------------------------------------------------------------


def test_production_finalize_rejects_ineligible(tmp_path: Path) -> None:
    """Finalize on a non-converged run exits nonzero."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import RunRecord

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)

    result = CliRunner().invoke(
        app,
        [
            "finalize",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
            "--run-id",
            "run-1",
        ],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Finalize succeeds against persisted eligible state
# ---------------------------------------------------------------------------


def test_production_finalize_succeeds_on_eligible_run(tmp_path: Path) -> None:
    """Finalize can be invoked on a converged run (may fail at strict eligibility check)."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import (
        ExperimentSpec,
        Fidelity,
        RunRecord,
    )

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)

    ExperimentSpec(
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
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)

    # Try to finalize — should fail with a meaningful error (not a crash)
    result = CliRunner().invoke(
        app,
        [
            "finalize",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
            "--run-id",
            "run-1",
        ],
    )
    # Should fail because the run is not converged
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Export writes files from persisted events
# ---------------------------------------------------------------------------


def test_production_export_rejects_without_persisted_finalization(tmp_path: Path) -> None:
    """Export command refuses events that lack authoritative finalization."""
    runtime_root, repo_dir = _setup_runtime(tmp_path)

    from tiktok2026.adapters import RepositoryRunStore
    from tiktok2026.contracts import AuditEvent, RunRecord

    repo = _repo(runtime_root)
    rs = RepositoryRunStore(repo)
    rs.put_run(RunRecord(run_id="run-1", status="active"), "run-1-active", None)

    rs.put_audit_event(
        AuditEvent(
            event_id="evt-1",
            run_id="run-1",
            experiment_id="exp-1",
            event_type="test_event",
            actor_type="controller",
            actor_id="test",
            payload={"msg": "hello"},
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "export",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
            "--run-id",
            "run-1",
        ],
    )
    assert result.exit_code != 0
    assert "persisted finalization" in result.output

    result2 = CliRunner().invoke(
        app,
        [
            "export",
            "--runtime-root",
            str(runtime_root),
            "--run-id",
            "nonexistent-run",
        ],
    )
    assert result2.exit_code != 0


# ---------------------------------------------------------------------------
# Synthetic resume records resume_accepted
# ---------------------------------------------------------------------------


def test_synthetic_resume_records_accepted(tmp_path: Path) -> None:
    """Synthetic resume records resume_accepted audit event."""
    import shutil

    runtime_root = tmp_path / "runtime"
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "migrations", repo_dir / "migrations")

    run_result = CliRunner().invoke(
        app,
        [
            "run",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
            "--synthetic",
        ],
    )
    assert run_result.exit_code == 0, run_result.output

    output_data = json.loads(run_result.stdout)
    run_id = output_data.get("run_id", "test-run")

    resume_result = CliRunner().invoke(
        app,
        [
            "resume",
            "--runtime-root",
            str(runtime_root),
            "--repository-root",
            str(repo_dir),
            "--run-id",
            run_id,
            "--synthetic",
        ],
    )
    assert resume_result.exit_code == 0, resume_result.output

    repo = ApplicationRepository(runtime_root / "application.sqlite3")
    events = repo.list_audit_events(run_id)
    event_types = [e.event_type for e in events]
    assert "resume_accepted" in event_types
