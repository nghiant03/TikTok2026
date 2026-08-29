import shutil
import sqlite3
from pathlib import Path

import pytest

from tiktok2026.contracts import (
    ArtifactRecord,
    ArtifactRetention,
    AuditEvent,
    EvaluatorIdentity,
    ExperimentSpec,
    Fidelity,
    FinalTestAuthorizationRequest,
    FinalTestRequest,
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
