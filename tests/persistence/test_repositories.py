import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from tiktok2026.adapters import RepositoryRunStore
from tiktok2026.contracts import (
    ArtifactRecord,
    ArtifactRetention,
    AuditEvent,
    BaselineCalibrationRecord,
    DiagnosticMetricValue,
    EvaluationResult,
    EvaluatorIdentity,
    ExperimentRegistryEntry,
    ExperimentSpec,
    FailureKind,
    FailureRecord,
    Fidelity,
    FinalTestAuthorizationRequest,
    FinalTestRequest,
    MetricValue,
    RunBaselineBinding,
    RunRecord,
    SourceRegistration,
)
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import (
    ApplicationRepository,
    FinalTestAccessError,
    PersistedFinalTestClaimResolver,
    PersistenceConflictError,
)
from tiktok2026.repository.diffs import patch_signature


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="A deterministic test",
        mechanism="Exercise persistence",
        motivation="Verify canonical storage",
        expected_signal="Records can be reconstructed",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Record persists",
        failure_criteria="Record is unavailable",
    )


def register_patch(
    repository: ApplicationRepository, root: Path, run_id: str = "run-1"
) -> tuple[str, Path]:
    patch = root / "artifacts" / run_id / "exp-1" / "patch.diff"
    patch.parent.mkdir(parents=True)
    patch.write_text("patch\n", encoding="utf-8")
    patch_sha256 = patch_signature(patch.read_text(encoding="utf-8"))
    artifact_path = patch.with_name(f"patch-{patch_sha256}.diff")
    patch.rename(artifact_path)
    repository.register_artifact(
        ArtifactRecord(
            artifact_id=f"patch-{patch_sha256}",
            run_id=run_id,
            experiment_id="exp-1",
            kind="source_patch",
            uri=artifact_path.resolve().as_uri(),
            sha256=patch_sha256,
            size_bytes=artifact_path.stat().st_size,
            producer="test",
            retention=ArtifactRetention.PROVENANCE,
        )
    )
    return patch_sha256, artifact_path


def test_experiment_write_is_idempotent(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        status="proposed",
        run_id="run-1",
        transition_id="exp-proposed",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        status="proposed",
        run_id="run-1",
        transition_id="exp-proposed",
        expected_predecessor=None,
    )
    assert repository.get_experiment("exp-1") == spec()


def test_run_experiment_state_survives_restart_without_synthetic_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "app.sqlite3"
    repository = ApplicationRepository(database)
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(), "proposed", "run-1", "exp-proposed", expected_predecessor=None
    )

    ApplicationRepository(database).initialize()

    with sqlite3.connect(database) as connection:
        states = connection.execute(
            "SELECT sequence, transition_id, predecessor_transition_id "
            "FROM authority_run_experiment_states"
        ).fetchall()
    assert states == [(1, "exp-proposed", None)]


def test_run_experiment_replay_binds_predecessor_and_reads_validate_integrity(
    tmp_path: Path,
) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(), "proposed", "run-1", "exp-proposed", expected_predecessor=None
    )

    with pytest.raises(PersistenceConflictError, match="content changed"):
        repository.put_experiment(
            spec(),
            "proposed",
            "run-1",
            "exp-proposed",
            expected_predecessor="different-predecessor",
        )

    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "UPDATE authority_run_experiment_states SET status = 'converged' "
            "WHERE run_id = 'run-1' AND experiment_id = 'exp-1'"
        )
    with pytest.raises(PersistenceConflictError, match="integrity check failed"):
        repository.list_experiments_by_status("run-1", "proposed")


def test_experiment_status_is_scoped_to_run(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    for run_id in ("run-a", "run-b"):
        repository.put_run(
            RunRecord(run_id=run_id, status="running"),
            transition_id=f"{run_id}-active",
            expected_predecessor=None,
        )
    repository.put_experiment(spec(), "proposed", "run-a", "exp-proposed", None)
    repository.put_experiment(spec(), "proposed", "run-b", "exp-proposed", None)

    repository.put_experiment(
        spec(), "completed", "run-a", "exp-completed", "exp-proposed"
    )

    run_a_proposals = repository.list_experiments_by_status("run-a", "proposed")
    run_b_proposals = repository.list_experiments_by_status("run-b", "proposed")
    assert run_a_proposals == ()
    assert tuple(item.experiment_id for item in run_b_proposals) == (
        "exp-1",
    )


def test_legacy_failure_replay_preserves_unbound_json_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    store = RepositoryRunStore(repository)
    legacy = FailureRecord(
        failure_id="legacy-failure-" + "x" * 500,
        kind=FailureKind.SCHEMA_MISMATCH,
        evidence_refs=tuple("legacy-evidence-" + "x" * 500 for _ in range(100)),
        repair_attempt=100,
    )
    repository.put_json("failure", legacy.failure_id, legacy.model_dump_json())

    store.put_failure(legacy, "run-a")

    assert repository.list_json("failure") == (legacy.model_dump_json(),)
    with pytest.raises(ValueError, match="does not match"):
        store.put_failure(legacy.model_copy(update={"run_id": "run-b"}), "run-a")
    with pytest.raises(PersistenceConflictError, match="legacy content changed"):
        store.put_failure(legacy.model_copy(update={"evidence_refs": ("changed",)}), "run-a")

    # A malformed unrelated legacy row is not parsed while locating a new ID.
    repository.put_json("failure", "malformed-history", "not-json")
    current = FailureRecord(
        failure_id="new-failure",
        kind=FailureKind.TIMEOUT,
        evidence_refs=("current-evidence",),
        repair_attempt=0,
    )
    store.put_failure(current, "run-a")
    assert any('"failure_id":"new-failure"' in item for item in repository.list_json("failure"))


def test_experiment_registry_source_is_bounded_and_reports_total(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        "proposed",
        "run-1",
        "experiment-proposed",
        expected_predecessor=None,
    )

    entries, total = repository.list_experiments(limit=1)

    assert total == 1
    assert entries == (
        ExperimentRegistryEntry(
            experiment_id="exp-1",
            hypothesis_id="hyp-1",
            hypothesis="A deterministic test",
            mechanism="Exercise persistence",
            status="registered",
        ),
    )

    excluded_entries, excluded_total = repository.list_experiments(
        limit=1, exclude_experiment_id="exp-1"
    )
    assert excluded_entries == ()
    assert excluded_total == 0


def test_audit_event_round_trips(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    event = AuditEvent(
        event_id="event-1",
        run_id="run-1",
        event_type="run_created",
        actor_type="controller",
        actor_id="bootstrap",
        payload={"profile": "test"},
    )
    repository.put_audit_event(event)
    assert repository.list_audit_events("run-1") == (event,)


def test_baseline_authority_migration_creates_typed_tables(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()

    with sqlite3.connect(repository.database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(authority_run_baseline_bindings)"
        ).fetchall()

    assert {
        "authority_baseline_calibrations",
        "authority_run_baseline_bindings",
    } <= tables
    assert any(
        foreign_key[2] == "authority_baseline_calibrations"
        for foreign_key in foreign_keys
    )


def test_run_baseline_binding_is_immutable_idempotent_and_audited(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    calibration = BaselineCalibrationRecord(
        calibration_id="calibration-1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        evaluator_id="provisional-within-user-v2",
        evaluator_sha256="b" * 64,
        baseline_source_sha256="c" * 64,
        config_sha256="d" * 64,
        prediction_sha256="e" * 64,
        prediction_artifact_uri="file:///calibration/predictions.json",
        evaluation=EvaluationResult(
            evaluation_id="baseline-evaluation-1",
            experiment_id="baseline",
            checkpoint_id="baseline-checkpoint",
            metrics=(
                MetricValue(name="GAUC", value=0.6674),
                MetricValue(name="nDCG@5", value=0.5357),
            ),
            evaluator_artifact_id="provisional-within-user-v2",
            evaluator_sha256="b" * 64,
            prediction_sha256="e" * 64,
            validity="provisional",
            dataset_manifest_id="manifest-1",
            dataset_manifest_sha256="a" * 64,
            split="valid",
        ),
        diagnostic_metrics=(
            DiagnosticMetricValue(name="GAUC", value=0.6674),
            DiagnosticMetricValue(name="nDCG@5", value=0.5357),
            DiagnosticMetricValue(name="primary", value=0.60155),
        ),
    )
    repository.put_baseline_calibration(calibration, "controller", "test")
    assert repository.list_baseline_calibrations() == (calibration,)
    with sqlite3.connect(repository.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM records WHERE kind = 'baseline_calibration'"
        ).fetchone() == (0,)
        connection.execute(
            "DELETE FROM audit_events WHERE event_id = ?",
            ("baseline-calibrated-calibration-1",),
        )
    repository.put_baseline_calibration(calibration, "controller", "test")
    assert [event.event_type for event in repository.list_audit_events("calibration-1")] == [
        "baseline_calibrated"
    ]
    with pytest.raises(PersistenceConflictError, match="content changed"):
        repository.put_baseline_calibration(
            calibration.model_copy(update={"config_sha256": "f" * 64}),
            "controller",
            "test",
        )
    binding = RunBaselineBinding(
        run_id="run-1",
        calibration_id="calibration-1",
        baseline_evaluation_id="baseline-evaluation-1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        evaluator_id="provisional-within-user-v2",
        evaluator_sha256="b" * 64,
        metrics=(
            MetricValue(name="GAUC", value=0.6674),
            MetricValue(name="nDCG@5", value=0.5357),
        ),
    )

    with pytest.raises(ValueError, match="unknown calibration"):
        repository.put_run_baseline(
            binding.model_copy(update={"calibration_id": "unknown-calibration"})
        )
    with pytest.raises(ValueError, match="does not match its calibration"):
        repository.put_run_baseline(
            binding.model_copy(
                update={"run_id": "run-2", "baseline_evaluation_id": "wrong-evaluation"}
            )
        )
    with pytest.raises(ValueError, match="typed atomic persistence"):
        repository.put_json(
            "baseline_calibration", calibration.calibration_id, calibration.model_dump_json()
        )
    repository.put_run_baseline(binding)
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "DELETE FROM audit_events WHERE event_id = ?",
            ("baseline-bound-run-1",),
        )
    repository.put_run_baseline(binding)

    assert repository.get_run_baseline("run-1") == binding
    events = repository.list_audit_events("run-1")
    assert [event.event_type for event in events] == ["baseline_bound"]

    changed = binding.model_copy(
        update={
            "metrics": (
                MetricValue(name="GAUC", value=0.7),
                MetricValue(name="nDCG@5", value=0.6),
            )
        }
    )
    with pytest.raises(PersistenceConflictError, match="content changed"):
        repository.put_run_baseline(changed)


def test_authority_records_reject_conflicting_replays(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        status="proposed",
        run_id="run-1",
        transition_id="exp-proposed",
        expected_predecessor=None,
    )

    with pytest.raises(PersistenceConflictError):
        repository.put_experiment(
            spec().model_copy(update={"motivation": "changed"}),
            status="proposed",
            run_id="run-1",
            transition_id="exp-proposed",
            expected_predecessor=None,
        )

    patch_sha256, patch = register_patch(repository, tmp_path)
    registration = SourceRegistration(
        experiment_id="exp-1",
        run_id="run-1",
        parent_commit="a" * 40,
        source_commit="b" * 40,
        patch_sha256=patch_sha256,
        patch_artifact_id=f"patch-{patch_sha256}",
        patch_artifact_uri=patch.resolve().as_uri(),
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    repository.put_source_registration(registration)
    with pytest.raises(PersistenceConflictError):
        repository.put_source_registration(
            registration.model_copy(update={"source_commit": "d" * 40})
        )


def test_source_registrations_are_append_only_revisions(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        status="proposed",
        run_id="run-1",
        transition_id="exp-proposed",
        expected_predecessor=None,
    )
    patch_sha256, patch = register_patch(repository, tmp_path)
    first = SourceRegistration(
        experiment_id="exp-1",
        run_id="run-1",
        parent_commit="a" * 40,
        source_commit="b" * 40,
        patch_sha256=patch_sha256,
        patch_artifact_id=f"patch-{patch_sha256}",
        patch_artifact_uri=patch.resolve().as_uri(),
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    second = first.model_copy(
        update={
            "registration_id": f"source-{'c' * 40}",
            "revision": 1,
            "source_commit": "c" * 40,
        }
    )

    repository.put_source_registration(first)
    repository.put_source_registration(second)

    assert repository.get_source_registration("exp-1") == second
    assert repository.get_source_registration_by_id(first.registration_id) == first
    assert repository.get_source_registration_by_id(second.registration_id) == second


def test_transitions_use_compare_and_swap_predecessors(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    repository.put_run(
        RunRecord(run_id="run-1", status="running"),
        transition_id="run-running",
        expected_predecessor=None,
    )
    with pytest.raises(PersistenceConflictError, match="content changed"):
        repository.put_run(
            RunRecord(run_id="run-1", status="converged"),
            transition_id="run-running",
            expected_predecessor=None,
        )
    with pytest.raises(PersistenceConflictError, match="stale predecessor"):
        repository.put_run(
            RunRecord(run_id="run-1", status="reopen"),
            transition_id="run-reopen",
            expected_predecessor="wrong",
        )
    repository.put_run(
        RunRecord(run_id="run-1", status="reopen"),
        transition_id="run-reopen",
        expected_predecessor="run-running",
    )
    repository.put_run(
        RunRecord(run_id="run-1", status="converged"),
        transition_id="run-reconverged",
        expected_predecessor="run-reopen",
    )

def test_final_test_is_controller_owned_and_single_use(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="converged"),
        transition_id="run-converged",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        status="converged",
        run_id="run-1",
        transition_id="exp-converged",
        expected_predecessor=None,
    )
    patch_sha256, patch = register_patch(repository, tmp_path)
    source = SourceRegistration(
        experiment_id="exp-1",
        run_id="run-1",
        parent_commit="a" * 40,
        source_commit="b" * 40,
        patch_sha256=patch_sha256,
        patch_artifact_id=f"patch-{patch_sha256}",
        patch_artifact_uri=patch.resolve().as_uri(),
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    repository.put_source_registration(source)
    repository.put_evaluator_identity(
        EvaluatorIdentity(
            evaluator_id="official-v1", evaluator_sha256="d" * 64, validity="official"
        )
    )
    authorization = FinalTestAuthorizationRequest(
        run_id="run-1",
        experiment_id="exp-1",
        source_commit=source.source_commit,
        evaluator_id="official-v1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="c" * 64,
        split="test",
        checkpoint_id="checkpoint-1",
        execution_id="execution-1",
        prediction_artifact_id="predictions-1",
        prediction_sha256="d" * 64,
    )
    claim = repository.authorize_final_test(authorization)
    assert PersistedFinalTestClaimResolver(repository).resolve(claim.claim_id) == claim
    request = FinalTestRequest(
        claim_id=claim.claim_id,
        finalization_id="final-1",
        run_id="run-1",
        experiment_id="exp-1",
        source_commit=source.source_commit,
        checkpoint_id="checkpoint-1",
        evaluation_id="evaluation-1",
        bundle_artifact_id="bundle-1",
        evaluator_id="official-v1",
    )

    finalization = repository.complete_final_test(request)
    assert finalization.validity == "official"
    assert finalization.consumed_test_access
    assert any(
        event.event_type == "final_test_completed"
        for event in repository.list_audit_events("run-1")
    )
    with pytest.raises(FinalTestAccessError, match="already been consumed"):
        repository.complete_final_test(
            request.model_copy(update={"finalization_id": "final-2"})
        )


def test_final_test_rejects_unregistered_source(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_run(
        RunRecord(run_id="run-1", status="converged"),
        transition_id="run-converged",
        expected_predecessor=None,
    )
    repository.put_experiment(
        spec(),
        status="converged",
        run_id="run-1",
        transition_id="exp-converged",
        expected_predecessor=None,
    )
    with pytest.raises(FinalTestAccessError, match="registered source"):
        repository.authorize_final_test(
            FinalTestAuthorizationRequest(
                run_id="run-1",
                experiment_id="exp-1",
                source_commit="a" * 40,
                evaluator_id="official-v1",
            )
        )


def test_generic_authority_persistence_is_rejected(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    with pytest.raises(ValueError, match="authority kind"):
        repository.put_json("finalization", "final-1", "{}")


def test_populated_legacy_database_is_adopted_without_duplicate_writes(tmp_path: Path) -> None:
    legacy_migrations = tmp_path / "legacy-migrations"
    legacy_migrations.mkdir()
    repository_root = Path(__file__).parents[2]
    for version in ("001_initial.sql", "002_pipeline.sql", "003_repository_support.sql"):
        shutil.copyfile(
            repository_root / "migrations" / "application" / version,
            legacy_migrations / version,
        )
    database = tmp_path / "legacy.sqlite3"
    MigrationRunner(database, legacy_migrations).apply()
    legacy_spec = spec().model_copy(update={"experiment_id": "legacy-exp"})
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO experiments "
            "(experiment_id, hypothesis_id, parent_experiment_id, status, source_commit, "
            "spec_json, created_at, updated_at) VALUES (?, ?, NULL, ?, NULL, ?, ?, ?)",
            (
                "legacy-exp",
                "hyp-legacy",
                "evaluated",
                legacy_spec.model_dump_json(),
                "now",
                "now",
            ),
        )
        connection.execute(
            "INSERT INTO runs (run_id, status, final_test_claimed, created_at) "
            "VALUES (?, ?, 0, ?)",
            ("legacy-run", "reopen", "now"),
        )
    repository = ApplicationRepository(database)
    repository.initialize()
    repository.initialize()
    assert repository.get_experiment("legacy-exp") is not None


def test_existing_source_registration_migrates_to_revision_zero(tmp_path: Path) -> None:
    old_migrations = tmp_path / "old-migrations"
    old_migrations.mkdir()
    repository_root = Path(__file__).parents[2]
    migration_root = repository_root / "migrations" / "application"
    for version in (
        "001_initial.sql",
        "002_pipeline.sql",
        "003_repository_support.sql",
        "004_authority_provenance.sql",
        "005_resource_authority.sql",
    ):
        shutil.copyfile(migration_root / version, old_migrations / version)
    database = tmp_path / "old.sqlite3"
    MigrationRunner(database, old_migrations).apply()
    registration = SourceRegistration(
        experiment_id="exp-1",
        run_id="run-1",
        parent_commit="a" * 40,
        source_commit="b" * 40,
        patch_sha256="c" * 64,
        patch_artifact_id=f"patch-{'c' * 64}",
        patch_artifact_uri="file:///tmp/patch.diff",
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    legacy_payload = registration.model_dump(mode="json")
    legacy_payload.pop("registration_id")
    legacy_payload.pop("revision")
    payload = json.dumps(legacy_payload, sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO authority_experiments VALUES (?, ?, ?, ?)",
            ("exp-1", spec().model_dump_json(), "d" * 64, "now"),
        )
        connection.execute(
            "INSERT INTO source_registrations VALUES (?, ?, ?, ?)",
            ("exp-1", payload, "e" * 64, "now"),
        )

    repository = ApplicationRepository(database)
    repository.initialize()

    migrated = repository.get_source_registration("exp-1")
    assert migrated is not None
    assert migrated.registration_id == f"source-{'b' * 40}"
    assert migrated.revision == 0
