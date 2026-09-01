import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from tiktok2026.contracts import (
    EvaluationResult,
    ExecutionResult,
    FailureKind,
    FailureRecord,
    FullAttemptClaimRequest,
    MetricValue,
    ProvenanceRequest,
    ResourceReservation,
    ScoredObservation,
    ScoredObservationRequest,
    SourceRegistration,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
)
from tiktok2026.persistence.repositories import ApplicationRepository, PersistenceConflictError

COMMIT = "a" * 40
SOURCE_ID = f"source-{COMMIT}"


def _claim_request(execution_id: str, attempt_id: str | None = None) -> FullAttemptClaimRequest:
    return FullAttemptClaimRequest(
        attempt_id=attempt_id or f"attempt-{execution_id}",
        execution_id=execution_id,
        run_id="run-1",
        experiment_id="experiment-1",
        source_registration_id=SOURCE_ID,
        source_commit=COMMIT,
    )


def _observation(
    attempt_id: str, execution_id: str, observation_id: str, primary_score: float = 0.6
) -> ScoredObservationRequest:
    return ScoredObservationRequest(
        observation_id=observation_id,
        run_id="run-1",
        experiment_id="experiment-1",
        attempt_id=attempt_id,
        execution_id=execution_id,
        evaluation_id=f"evaluation-{observation_id}",
        checkpoint_id=f"checkpoint-{observation_id}",
        source_commit=COMMIT,
        evaluator_id="evaluator-1",
        evaluator_sha256="b" * 64,
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="c" * 64,
        validity="provisional",
        primary_score=primary_score,
        validation_report_id="report-1",
        validation_evidence_refs=("evidence-1",),
    )


def _repository(tmp_path: Path) -> ApplicationRepository:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(repository.database) as connection:
        cutoff = datetime.fromisoformat(
            str(
                connection.execute(
                    "SELECT applied_at FROM schema_migrations WHERE version = 10"
                ).fetchone()[0]
            )
        )
    source_created_at = (cutoff - timedelta(seconds=2)).isoformat()
    source = SourceRegistration(
        registration_id=SOURCE_ID,
        revision=0,
        experiment_id="experiment-1",
        run_id="run-1",
        parent_commit=COMMIT,
        source_commit=COMMIT,
        patch_sha256="d" * 64,
        patch_artifact_id="patch-" + "d" * 64,
        patch_artifact_uri="file:///unused",
        allowed_scopes=("src",),
        eligible=True,
    )
    payload = source.model_dump_json()
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "INSERT INTO runs (run_id, status, final_test_claimed, created_at) VALUES (?, ?, 0, ?)",
            ("run-1", "execute", now),
        )
        connection.execute(
            "INSERT INTO authority_experiments "
            "(experiment_id, spec_json, content_sha256, created_at) "
            "VALUES (?, '{}', ?, ?)",
            ("experiment-1", hashlib.sha256(b"{}").hexdigest(), now),
        )
        connection.execute(
            "INSERT INTO authority_run_experiment_states "
            "(run_id, experiment_id, sequence, status, transition_id, "
            "predecessor_transition_id, content_sha256, created_at) "
            "VALUES (?, ?, 1, 'completed', ?, NULL, ?, ?)",
            (
                "run-1",
                "experiment-1",
                "test-completed",
                ApplicationRepository._run_experiment_state_hash(
                    "run-1", "experiment-1", 1, "completed", "test-completed", None
                ),
                now,
            ),
        )
        connection.execute(
            "INSERT INTO source_registrations "
            "(registration_id, experiment_id, revision, registration_json, content_sha256, "
            "created_at, run_id, source_commit, eligible) VALUES (?, ?, 0, ?, ?, ?, ?, ?, 1)",
            (
                SOURCE_ID,
                "experiment-1",
                payload,
                hashlib.sha256(payload.encode()).hexdigest(),
                source_created_at,
                "run-1",
                COMMIT,
            ),
        )
    return repository


def _seed_observation_dependencies(
    repository: ApplicationRepository,
    observation: ScoredObservationRequest,
    persist_report: bool = True,
) -> None:
    execution = ExecutionResult(
        execution_id=observation.execution_id,
        experiment_id=observation.experiment_id,
        source_registration_id=SOURCE_ID,
        source_commit=COMMIT,
        command=("python", "train.py"),
        exit_code=0,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        checkpoint_id=observation.checkpoint_id,
        dataset_manifest_id=observation.dataset_manifest_id,
        dataset_manifest_sha256=observation.dataset_manifest_sha256,
    )
    evaluation = EvaluationResult(
        evaluation_id=observation.evaluation_id,
        experiment_id=observation.experiment_id,
        checkpoint_id=observation.checkpoint_id,
        metrics=(
            MetricValue(name="GAUC", value=0.5),
            MetricValue(name="nDCG@5", value=2 * observation.primary_score - 0.5),
        ),
        evaluator_artifact_id=observation.evaluator_id,
        evaluator_sha256=observation.evaluator_sha256,
        prediction_sha256="e" * 64,
        validity=observation.validity,
        dataset_manifest_id=observation.dataset_manifest_id,
        dataset_manifest_sha256=observation.dataset_manifest_sha256,
        split="valid",
        run_id=observation.run_id,
        source_commit=observation.source_commit,
        execution_id=observation.execution_id,
    )
    provenance = ProvenanceRequest(
        run_id=observation.run_id,
        experiment_id=observation.experiment_id,
        source_commit=observation.source_commit,
        execution_id=observation.execution_id,
        dataset_manifest_id=observation.dataset_manifest_id,
        dataset_manifest_sha256=observation.dataset_manifest_sha256,
        evaluator_id=observation.evaluator_id,
        evaluator_sha256=observation.evaluator_sha256,
    )
    repository.put_json("execution", execution.execution_id, execution.model_dump_json())
    repository.put_json(
        "evaluation",
        evaluation.evaluation_id,
        json.dumps(
            {
                "result": evaluation.model_dump(mode="json"),
                "provenance": provenance.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    report = ValidationReport(
        report_id=observation.validation_report_id,
        experiment_id=observation.experiment_id,
        stage=ValidationStage.RESULT,
        verdict=ValidationVerdict.APPROVED,
        leakage_risk="none",
        evidence_refs=observation.validation_evidence_refs,
    )
    subject: dict[str, object] = {
        "evaluation_result": evaluation.model_dump(mode="json"),
        "execution_result": execution.model_dump(mode="json", exclude={"dataset_valid_rows"}),
    }
    subject_json = json.dumps(subject, sort_keys=True, separators=(",", ":"))
    operation = ValidationOperationIdentity(
        operation_id=f"operation-{observation.validation_report_id}",
        run_id=observation.run_id,
        experiment_id=observation.experiment_id,
        stage=ValidationStage.RESULT,
        repair_attempt=0,
        subject_sha256=hashlib.sha256(subject_json.encode()).hexdigest(),
    )
    if persist_report:
        repository.put_validation_report(report, observation.run_id, operation, subject)


def _insert_legacy_reservations(
    repository: ApplicationRepository, execution_ids: tuple[str, ...]
) -> None:
    with sqlite3.connect(repository.database) as connection:
        cutoff = datetime.fromisoformat(
            str(connection.execute(
                "SELECT applied_at FROM schema_migrations WHERE version = 10"
            ).fetchone()[0])
        )
        now = (cutoff - timedelta(seconds=1)).isoformat()
        connection.execute(
            "INSERT INTO run_states (run_id, status, transition_id, content_sha256, created_at) "
            "VALUES ('run-1', 'active', 'legacy-active', ?, ?)",
            (hashlib.sha256(b"legacy-active").hexdigest(), now),
        )
        for index, execution_id in enumerate(execution_ids):
            reservation = ResourceReservation(
                reservation_id=f"reservation-{execution_id}",
                run_id="run-1",
                experiment_id="experiment-1",
                gpu_hours=0.0,
                wall_seconds=1.0,
                tokens=0,
                disk_bytes=0,
            )
            connection.execute(
                "INSERT INTO authority_resource_reservations "
                "(reservation_id, reservation_json, status, created_at) "
                "VALUES (?, ?, 'consumed', ?)",
                (reservation.reservation_id, reservation.model_dump_json(), now),
            )
            connection.execute(
                "INSERT INTO authority_resource_operations "
                "(operation_id, reservation_id, operation, usage_json, created_at) "
                "VALUES (?, ?, 'reserve', NULL, ?)",
                (f"legacy-reserve-{index:03d}-{execution_id}", reservation.reservation_id, now),
            )


def test_attempt_claim_is_idempotent_and_conflicts_on_changed_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = repository.claim_full_attempt(_claim_request("execution-1"))
    assert repository.claim_full_attempt(_claim_request("execution-1")) == first
    with pytest.raises(PersistenceConflictError):
        repository.claim_full_attempt(_claim_request("execution-1", attempt_id="different-attempt"))


def test_shared_records_are_hashed_on_write_and_verified_on_read(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.put_json("custom", "record-1", '{"value":1}')
    repository.persist_transition("run-1", "test", 1, {"value": 1})
    with sqlite3.connect(repository.database) as connection:
        rows = connection.execute(
            "SELECT payload_json, content_sha256 FROM records "
            "WHERE kind IN ('custom', 'transition')"
        ).fetchall()
    assert all(digest == hashlib.sha256(payload.encode()).hexdigest() for payload, digest in rows)
    assert repository.list_json("custom") == ('{"value":1}',)


def test_attempt_claims_are_capped_and_ordered(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for index in range(1, 51):
        repository.claim_full_attempt(_claim_request(f"execution-{index}"))
    assert repository.claim_full_attempt(_claim_request("execution-51")) is None
    claims = repository.list_full_attempt_claims("run-1")
    assert tuple(claim.attempt_sequence for claim in claims) == tuple(range(1, 51))
    assert repository.count_full_attempt_claims("run-1") == 50


def test_concurrent_claims_are_contiguous_and_capped(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    barrier = Barrier(50)

    def claim(index: int) -> int:
        barrier.wait()
        claim = repository.claim_full_attempt(_claim_request(f"thread-execution-{index}"))
        assert claim is not None
        return claim.attempt_sequence

    with ThreadPoolExecutor(max_workers=50) as executor:
        sequences = list(executor.map(claim, range(50)))
    assert sorted(sequences) == list(range(1, 51))
    assert repository.count_full_attempt_claims("run-1") == 50
    assert repository.claim_full_attempt(_claim_request("thread-execution-51")) is None


def test_scored_observation_validates_provenance_and_is_idempotent(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    attempt = repository.claim_full_attempt(_claim_request("execution-1"))
    observation = _observation(attempt.attempt_id, attempt.execution_id, "observation-1")
    _seed_observation_dependencies(repository, observation)
    record = repository.put_scored_observation(observation)
    assert repository.put_scored_observation(observation) == record
    assert repository.get_scored_observation(observation.observation_id) == record
    assert repository.list_scored_observations("run-1") == (record,)


def test_legacy_lifecycle_adoption_is_idempotent_and_excludes_failed_execution(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    successful = _observation(
        "attempt-legacy-success", "execution-legacy-success", "legacy-success"
    )
    _seed_observation_dependencies(repository, successful)
    failed = ExecutionResult(
        execution_id="execution-legacy-failed",
        experiment_id="experiment-1",
        source_registration_id=SOURCE_ID,
        source_commit=COMMIT,
        command=("python", "train.py"),
        exit_code=1,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        failure_kind="timeout",
        failure_message="legacy timeout",
    )
    repository.put_json("execution", failed.execution_id, failed.model_dump_json())
    repository.put_json(
        "failure",
        "failure-run-1-2",
        FailureRecord(
            failure_id="failure-run-1-2",
            experiment_id="experiment-1",
            kind=FailureKind.TIMEOUT,
            evidence_refs=(failed.execution_id,),
            repair_attempt=0,
        ).model_dump_json(),
    )
    _insert_legacy_reservations(
        repository, (successful.execution_id, failed.execution_id)
    )
    repository.persist_transition(
        "run-1",
        "persist_failure",
        1,
        {
            "terminal_reason": "failure:timeout",
            "evidence": (failed.execution_id,),
            "pending_route": "persist_failure",
        },
    )

    first = repository.adopt_legacy_lifecycle("run-1")
    second = repository.adopt_legacy_lifecycle("run-1")

    assert first[0]["adopted_attempts"] == 2
    assert first[0]["adopted_observations"] == 1
    assert second[0]["adopted_observations"] == 1
    assert repository.count_full_attempt_claims("run-1") == 2
    assert len(repository.list_scored_observations("run-1")) == 1
    assert (
        len(
            [
                event
                for event in repository.list_audit_events("run-1")
                if event.event_id == "lifecycle-adoption-run-1"
            ]
        )
        == 1
    )


def test_legacy_adoption_rejects_reservation_payload_identity_mismatch(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    indexed_id = "reservation-execution-mismatch"
    payload_reservation = ResourceReservation(
        reservation_id="reservation-different-payload",
        run_id="run-1",
        experiment_id="experiment-1",
        gpu_hours=0.0,
        wall_seconds=1.0,
        tokens=0,
        disk_bytes=0,
    )
    with sqlite3.connect(repository.database) as connection:
        cutoff = datetime.fromisoformat(
            str(
                connection.execute(
                    "SELECT applied_at FROM schema_migrations WHERE version = 10"
                ).fetchone()[0]
            )
        )
        created_at = (cutoff - timedelta(seconds=1)).isoformat()
        connection.execute(
            "INSERT INTO authority_resource_reservations "
            "(reservation_id, reservation_json, status, created_at) "
            "VALUES (?, ?, 'consumed', ?)",
            (indexed_id, payload_reservation.model_dump_json(), created_at),
        )
        connection.execute(
            "INSERT INTO authority_resource_operations "
            "(operation_id, reservation_id, operation, usage_json, created_at) "
            "VALUES (?, ?, 'reserve', NULL, ?)",
            ("legacy-reserve-mismatch", indexed_id, created_at),
        )

    with pytest.raises(PersistenceConflictError, match="identity mismatch"):
        repository.adopt_legacy_lifecycle("run-1")
    assert repository.count_full_attempt_claims("run-1") == 0


def test_legacy_adoption_rejects_duplicate_reserve_operations_before_claims(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    reservation = ResourceReservation(
        reservation_id="reservation-execution-duplicate",
        run_id="run-1",
        experiment_id="experiment-1",
        gpu_hours=0.0,
        wall_seconds=1.0,
        tokens=0,
        disk_bytes=0,
    )
    with sqlite3.connect(repository.database) as connection:
        cutoff = datetime.fromisoformat(
            str(
                connection.execute(
                    "SELECT applied_at FROM schema_migrations WHERE version = 10"
                ).fetchone()[0]
            )
        )
        created_at = (cutoff - timedelta(seconds=1)).isoformat()
        connection.execute(
            "INSERT INTO authority_resource_reservations "
            "(reservation_id, reservation_json, status, created_at) "
            "VALUES (?, ?, 'consumed', ?)",
            (reservation.reservation_id, reservation.model_dump_json(), created_at),
        )
        for index in range(2):
            connection.execute(
                "INSERT INTO authority_resource_operations "
                "(operation_id, reservation_id, operation, usage_json, created_at) "
                "VALUES (?, ?, 'reserve', NULL, ?)",
                (f"legacy-reserve-duplicate-{index}", reservation.reservation_id, created_at),
            )

    with pytest.raises(PersistenceConflictError, match="exactly one reserve operation"):
        repository.adopt_legacy_lifecycle("run-1")
    assert repository.count_full_attempt_claims("run-1") == 0


def test_legacy_adoption_blocks_ambiguous_missing_result_without_partial_claims(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    second_commit = "b" * 40
    second_source = SourceRegistration(
        registration_id=f"source-{second_commit}",
        revision=1,
        experiment_id="experiment-1",
        run_id="run-1",
        parent_commit=COMMIT,
        source_commit=second_commit,
        patch_sha256="e" * 64,
        patch_artifact_id="patch-e",
        patch_artifact_uri="file:///unused",
        allowed_scopes=("src",),
        eligible=True,
    )
    payload = second_source.model_dump_json()
    with sqlite3.connect(repository.database) as connection:
        cutoff = datetime.fromisoformat(
            str(
                connection.execute(
                    "SELECT applied_at FROM schema_migrations WHERE version = 10"
                ).fetchone()[0]
            )
        )
        connection.execute(
            "INSERT INTO source_registrations "
            "(registration_id, experiment_id, revision, registration_json, content_sha256, "
            "created_at, run_id, source_commit, eligible) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (
                second_source.registration_id,
                second_source.experiment_id,
                second_source.revision,
                payload,
                hashlib.sha256(payload.encode()).hexdigest(),
                (cutoff - timedelta(seconds=2)).isoformat(),
                second_source.run_id,
                second_source.source_commit,
            ),
        )
    _insert_legacy_reservations(repository, ("execution-missing",))

    with pytest.raises(PersistenceConflictError, match="ambiguous legacy lifecycle"):
        repository.adopt_legacy_lifecycle("run-1")
    assert repository.count_full_attempt_claims("run-1") == 0


def test_legacy_adoption_closes_on_the_fiftieth_reservation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    execution_ids = tuple(f"execution-missing-{index:02d}" for index in range(50))
    _insert_legacy_reservations(repository, execution_ids)

    summaries = repository.adopt_legacy_lifecycle("run-1")

    closure = repository.get_run_closure("run-1")
    assert summaries[0]["adopted_attempts"] == 50
    assert summaries[0]["closure_reason"] == "attempt_cap"
    assert closure is not None
    assert closure.reason == "attempt_cap"
    assert closure.champion is None
    assert repository.count_full_attempt_claims("run-1") == 50


def test_legacy_adoption_rejects_corrupt_execution_without_partial_claims(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    execution = _observation("attempt-corrupt", "execution-corrupt", "observation-corrupt")
    _seed_observation_dependencies(repository, execution)
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "UPDATE records SET content_sha256 = ? WHERE kind = 'execution' AND record_id = ?",
            ("0" * 64, execution.execution_id),
        )
    _insert_legacy_reservations(repository, (execution.execution_id,))

    with pytest.raises(PersistenceConflictError, match="ambiguous legacy lifecycle"):
        repository.adopt_legacy_lifecycle("run-1")
    assert repository.count_full_attempt_claims("run-1") == 0


def test_legacy_adoption_rejects_source_created_after_reservation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _insert_legacy_reservations(repository, ("execution-postdated-source",))
    with sqlite3.connect(repository.database) as connection:
        cutoff = datetime.fromisoformat(
            str(
                connection.execute(
                    "SELECT applied_at FROM schema_migrations WHERE version = 10"
                ).fetchone()[0]
            )
        )
        connection.execute(
            "UPDATE source_registrations SET created_at = ? WHERE registration_id = ?",
            ((cutoff + timedelta(seconds=1)).isoformat(), SOURCE_ID),
        )

    with pytest.raises(PersistenceConflictError, match="ambiguous legacy lifecycle"):
        repository.adopt_legacy_lifecycle("run-1")
    assert repository.count_full_attempt_claims("run-1") == 0


def test_legacy_adoption_ignores_post_migration_reservations(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _insert_legacy_reservations(repository, ())
    now = datetime.now(UTC).isoformat()
    reservation = ResourceReservation(
        reservation_id="reservation-modern-execution",
        run_id="run-1",
        experiment_id="experiment-1",
        gpu_hours=0.0,
        wall_seconds=1.0,
        tokens=0,
        disk_bytes=0,
    )
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "INSERT INTO authority_resource_reservations "
            "(reservation_id, reservation_json, status, created_at) "
            "VALUES (?, ?, 'consumed', ?)",
            (reservation.reservation_id, reservation.model_dump_json(), now),
        )
        connection.execute(
            "INSERT INTO authority_resource_operations "
            "(operation_id, reservation_id, operation, usage_json, created_at) "
            "VALUES ('modern-reserve', ?, 'reserve', NULL, ?)",
            (reservation.reservation_id, now),
        )

    summaries = repository.adopt_legacy_lifecycle("run-1")
    assert summaries[0]["adopted_attempts"] == 0
    assert repository.count_full_attempt_claims("run-1") == 0


def test_observation_rejects_mismatched_evaluation_provenance(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    attempt = repository.claim_full_attempt(_claim_request("execution-1"))
    observation = _observation(attempt.attempt_id, attempt.execution_id, "observation-1")
    _seed_observation_dependencies(repository, observation)
    payload = json.loads(repository.list_json("evaluation")[0])
    payload["result"]["checkpoint_id"] = "checkpoint-forged"
    forged_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "UPDATE records SET payload_json = ?, content_sha256 = ? "
            "WHERE kind = 'evaluation' AND record_id = ?",
            (forged_payload, hashlib.sha256(forged_payload.encode()).hexdigest(),
             observation.evaluation_id),
        )
    with pytest.raises(ValueError, match="evaluation provenance"):
        repository.put_scored_observation(observation)


def test_result_validation_report_cannot_be_reused_for_another_evaluation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    first_attempt = repository.claim_full_attempt(_claim_request("execution-1"))
    first = _observation(first_attempt.attempt_id, first_attempt.execution_id, "observation-1")
    _seed_observation_dependencies(repository, first)
    repository.put_scored_observation(first)

    second_attempt = repository.claim_full_attempt(_claim_request("execution-2"))
    second = _observation(second_attempt.attempt_id, second_attempt.execution_id, "observation-2")
    second = second.model_copy(update={"validation_report_id": first.validation_report_id})
    _seed_observation_dependencies(repository, second, persist_report=False)
    with pytest.raises(ValueError, match="subject provenance"):
        repository.put_scored_observation(second)


def test_plateau_closure_derives_champion_from_ordered_observations(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    records: list[ScoredObservation] = []
    for index, score in enumerate((0.5, 0.501, 0.502, 0.503)):
        attempt = repository.claim_full_attempt(_claim_request(f"execution-{index}"))
        observation = _observation(
            attempt.attempt_id, attempt.execution_id, f"observation-{index}", score
        )
        observation = observation.model_copy(update={"validation_report_id": f"report-{index}"})
        _seed_observation_dependencies(repository, observation)
        records.append(repository.put_scored_observation(observation))
    # Deliberately make storage chronology disagree with scientific attempt order.
    with sqlite3.connect(repository.database) as connection:
        for index, record in enumerate(reversed(records)):
            forged = record.model_copy(
                update={"scored_at": datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)}
            )
            payload = forged.model_dump_json()
            connection.execute(
                "UPDATE authority_scored_observations SET observation_json = ?, "
                "content_sha256 = ?, scored_at = ? WHERE observation_id = ?",
                (payload, hashlib.sha256(payload.encode()).hexdigest(),
                 forged.scored_at.isoformat(), forged.observation_id),
            )
    with pytest.raises(PersistenceConflictError, match="plateau closure"):
        repository.close_run("run-1", "plateau")


def test_observation_replay_precedes_state_and_closure_checks(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    attempt = repository.claim_full_attempt(_claim_request("execution-1"))
    request = _observation(attempt.attempt_id, attempt.execution_id, "observation-1")
    _seed_observation_dependencies(repository, request)
    record = repository.put_scored_observation(request)
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "INSERT INTO authority_run_experiment_states "
            "(run_id, experiment_id, sequence, status, transition_id, "
            "predecessor_transition_id, content_sha256, created_at) "
            "VALUES (?, ?, 2, 'converged', ?, ?, ?, ?)",
            (
                "run-1",
                "experiment-1",
                "test-converged",
                "test-completed",
                ApplicationRepository._run_experiment_state_hash(
                    "run-1",
                    "experiment-1",
                    2,
                    "converged",
                    "test-converged",
                    "test-completed",
                ),
                datetime.now(UTC).isoformat(),
            ),
        )
    assert repository.put_scored_observation(request) == record
    with pytest.raises(PersistenceConflictError, match="content changed"):
        repository.put_scored_observation(request.model_copy(update={"primary_score": 0.61}))
    with sqlite3.connect(repository.database) as connection:
        connection.execute(
            "INSERT INTO authority_run_experiment_states "
            "(run_id, experiment_id, sequence, status, transition_id, "
            "predecessor_transition_id, content_sha256, created_at) "
            "VALUES (?, ?, 3, 'completed', ?, ?, ?, ?)",
            (
                "run-1",
                "experiment-1",
                "test-completed-again",
                "test-converged",
                ApplicationRepository._run_experiment_state_hash(
                    "run-1",
                    "experiment-1",
                    3,
                    "completed",
                    "test-completed-again",
                    "test-converged",
                ),
                datetime.now(UTC).isoformat(),
            ),
        )
    for index in range(2, 51):
        repository.claim_full_attempt(_claim_request(f"execution-{index}"))
    repository.close_run("run-1", "attempt_cap")
    assert repository.put_scored_observation(request) == record


def test_observation_uniqueness_and_no_post_closure_writes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    attempt = repository.claim_full_attempt(_claim_request("execution-1"))
    observation = _observation(attempt.attempt_id, attempt.execution_id, "observation-1")
    _seed_observation_dependencies(repository, observation)
    repository.put_scored_observation(observation)
    with pytest.raises(PersistenceConflictError):
        repository.put_scored_observation(
            observation.model_copy(update={"observation_id": "other"})
        )
    for index in range(2, 51):
        repository.claim_full_attempt(_claim_request(f"execution-{index}"))
    closure = repository.close_run("run-1", "attempt_cap")
    assert closure.attempt_count == 50
    with pytest.raises(PersistenceConflictError, match="already closed"):
        repository.claim_full_attempt(_claim_request("execution-51"))
    with pytest.raises(PersistenceConflictError, match="already closed"):
        repository.put_scored_observation(
            observation.model_copy(update={"observation_id": "replay-2"})
        )


def test_closure_is_repository_derived_and_replay_checks_reason(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    for index in range(1, 51):
        repository.claim_full_attempt(_claim_request(f"execution-{index}"))
    closure = repository.close_run("run-1", "attempt_cap")
    assert closure.attempt_count == 50
    assert closure.scored_observation_count == 0
    assert closure.champion is None
    assert repository.close_run("run-1", "attempt_cap") == closure
    with pytest.raises(PersistenceConflictError, match="another reason"):
        repository.close_run("run-1", "plateau")
