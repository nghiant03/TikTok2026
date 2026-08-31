from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from tiktok2026.contracts import (
    MAX_FULL_ATTEMPTS,
    ArtifactRecord,
    AuditEvent,
    BaselineCalibrationRecord,
    BlockerResolution,
    ChampionBinding,
    CriterionAssessmentStatus,
    CriterionResolutionClaim,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionResult,
    ExperimentRegistryEntry,
    ExperimentSpec,
    FailureRecord,
    FinalizationRecord,
    FinalTestAuthorizationRequest,
    FinalTestClaim,
    FinalTestRequest,
    FullAttemptClaimRequest,
    FullScientificAttemptClaim,
    ImplementationCriterionAssessment,
    ImplementationValidationAuthority,
    ProvenanceRequest,
    ProvisionalFinalizationRequest,
    ResourceReservation,
    RunBaselineBinding,
    RunClosure,
    RunRecord,
    ScoredObservation,
    ScoredObservationRequest,
    SourceRegistration,
    ValidationBlocker,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
)
from tiktok2026.persistence.migrations import MigrationRunner, application_migrations_path
from tiktok2026.policies.lifecycle import convergence_reason
from tiktok2026.repository.diffs import patch_signature


class PersistenceConflictError(RuntimeError):
    """An immutable authority record was replayed with different content."""


class FinalTestAccessError(PermissionError):
    """A final-test request does not satisfy controller policy."""


class PersistedFinalTestClaimResolver:
    """Resolve immutable Phase 1 claims for evaluator-side authorization."""

    def __init__(self, repository: ApplicationRepository) -> None:
        self.repository = repository

    def resolve(self, claim_id: str) -> FinalTestClaim | None:
        return self.repository.get_final_test_claim(claim_id)


def _content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()


def _authority_timestamp(value: object, evidence: str) -> datetime:
    if not isinstance(value, str):
        raise PersistenceConflictError(f"malformed {evidence} timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PersistenceConflictError(f"malformed {evidence} timestamp") from error
    if parsed.tzinfo is None:
        raise PersistenceConflictError(f"timezone missing from {evidence} timestamp")
    return parsed.astimezone(UTC)


class ApplicationRepository:
    def __init__(self, database: Path) -> None:
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        MigrationRunner(self.database, application_migrations_path()).apply()
        self._adopt_legacy_records()

    def _adopt_legacy_records(self) -> None:
        """Copy legacy records into append-only authority tables once."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(UTC).isoformat()
            experiments = connection.execute(
                "SELECT experiment_id, spec_json, status FROM experiments"
            ).fetchall()
            for experiment_id, payload, status in experiments:
                digest = _content_hash(payload)
                existing = connection.execute(
                    "SELECT content_sha256 FROM authority_experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                if existing is not None and existing[0] != digest:
                    raise PersistenceConflictError(
                        f"legacy experiment {experiment_id} conflicts with authority"
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO authority_experiments "
                    "(experiment_id, spec_json, content_sha256, created_at) VALUES (?, ?, ?, ?)",
                    (experiment_id, payload, digest, now),
                )
                if (
                    connection.execute(
                        "SELECT 1 FROM experiment_states WHERE experiment_id = ? LIMIT 1",
                        (experiment_id,),
                    ).fetchone()
                    is None
                ):
                    connection.execute(
                        "INSERT INTO experiment_states "
                        "(experiment_id, status, transition_id, content_sha256, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            experiment_id,
                            status,
                            f"legacy-experiment-{experiment_id}",
                            _content_hash(f"legacy-experiment-{experiment_id}:{status}"),
                            now,
                        ),
                    )
            runs = connection.execute("SELECT run_id, status FROM runs").fetchall()
            for run_id, status in runs:
                if (
                    connection.execute(
                        "SELECT 1 FROM run_states WHERE run_id = ? LIMIT 1", (run_id,)
                    ).fetchone()
                    is None
                ):
                    connection.execute(
                        "INSERT INTO run_states "
                        "(run_id, status, transition_id, content_sha256, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            run_id,
                            status,
                            f"legacy-run-{run_id}",
                            _content_hash(f"legacy-run-{run_id}:{status}"),
                            now,
                        ),
                    )
            # Baseline calibration predates its dedicated authority table in
            # some runtime databases.  Adopt only this explicitly supported
            # legacy authority kind; other generic records remain generic.
            legacy_calibrations = connection.execute(
                "SELECT record_id, payload_json, content_sha256 FROM records "
                "WHERE kind = 'baseline_calibration' ORDER BY record_id"
            ).fetchall()
            for record_id, legacy_payload, legacy_digest in legacy_calibrations:
                if legacy_digest is None or _content_hash(legacy_payload) != legacy_digest:
                    raise PersistenceConflictError(
                        f"persisted baseline calibration {record_id} integrity check failed"
                    )
                record = BaselineCalibrationRecord.model_validate_json(legacy_payload)
                payload = record.model_dump_json()
                digest = _content_hash(payload)
                existing = connection.execute(
                    "SELECT content_sha256 FROM authority_baseline_calibrations "
                    "WHERE calibration_id = ?",
                    (record.calibration_id,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != digest:
                        raise PersistenceConflictError(
                            f"baseline calibration {record.calibration_id} conflicts with authority"
                        )
                    continue
                if record_id != record.calibration_id:
                    raise PersistenceConflictError(
                        f"legacy baseline calibration {record_id} has mismatched identity"
                    )
                connection.execute(
                    "INSERT INTO authority_baseline_calibrations "
                    "(calibration_id, payload_json, content_sha256, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (record.calibration_id, payload, digest, now),
                )

    @staticmethod
    def _transition(
        entity: str,
        identity: str,
        status: str,
        transition_id: str,
        previous_id: str | None,
    ) -> str:
        payload = json.dumps(
            {
                "entity": entity,
                "identity": identity,
                "status": status,
                "transition_id": transition_id,
                "previous": previous_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return _content_hash(payload)

    @staticmethod
    def _insert_audit(connection: sqlite3.Connection, event: AuditEvent) -> None:
        payload = event.model_dump_json()
        row = connection.execute(
            "SELECT payload_json FROM audit_events WHERE event_id = ?", (event.event_id,)
        ).fetchone()
        if row is not None:
            if row[0] != payload:
                raise PersistenceConflictError(f"audit event {event.event_id} content changed")
            return
        connection.execute(
            "INSERT INTO audit_events "
            "(event_id, run_id, experiment_id, event_type, actor_type, actor_id, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.run_id,
                event.experiment_id,
                event.event_type,
                event.actor_type,
                event.actor_id,
                payload,
                event.created_at.isoformat(),
            ),
        )

    def persist_transition(
        self,
        run_id: str,
        operation: str,
        state_version: int,
        updates: dict[str, object],
    ) -> None:
        """Atomically append a controller transition and its audit event.

        The transition table is represented by the generic records table, but
        the read/compare/write and audit insertion deliberately share one
        ``BEGIN IMMEDIATE`` transaction.  This is the authority boundary for
        graph transitions; callers must not implement a split CAS themselves.
        """
        if state_version < 1:
            raise PersistenceConflictError("transition versions start at one")
        payload = json.dumps(
            {
                "run_id": run_id,
                "operation": operation,
                "state_version": state_version,
                "updates": updates,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        record_id = f"{run_id}:{state_version}"
        event_id = f"transition-{run_id}-{state_version}"
        now = datetime.now(UTC)
        event = AuditEvent(
            event_id=event_id,
            run_id=run_id,
            event_type="controller_transition",
            actor_type="controller",
            actor_id="production-controller",
            payload={
                "operation": operation,
                "state_version": state_version,
                "updates": updates,
            },
            created_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json, content_sha256 FROM records "
                "WHERE kind = 'transition' AND record_id = ?",
                (record_id,),
            ).fetchone()
            if existing is not None:
                if existing[1] is None or _content_hash(existing[0]) != existing[1]:
                    raise PersistenceConflictError("record transition integrity check failed")
                if existing[0] != payload:
                    raise PersistenceConflictError(f"transition {record_id} content changed")
                # An interrupted transaction cannot leave this path half
                # committed, but older data may lack its audit event.  Repair
                # that event during an identical, idempotent retry.
                audit = connection.execute(
                    "SELECT payload_json FROM audit_events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
                if audit is None:
                    self._insert_audit(connection, event)
                else:
                    actual = json.loads(audit[0])
                    expected = event.model_dump(mode="json")
                    actual.pop("created_at", None)
                    expected.pop("created_at", None)
                    if actual != expected:
                        raise PersistenceConflictError(f"audit event {event_id} content changed")
                return

            rows = connection.execute(
                "SELECT payload_json, content_sha256 FROM records WHERE kind = 'transition'"
            ).fetchall()
            versions: set[int] = set()
            for raw, digest in rows:
                if digest is None or _content_hash(raw) != digest:
                    raise PersistenceConflictError("record transition integrity check failed")
                value = json.loads(raw)
                if value.get("run_id") == run_id:
                    versions.add(int(value["state_version"]))
            if state_version == 1:
                if versions:
                    raise PersistenceConflictError("transition version one has a predecessor")
            elif state_version - 1 not in versions or any(
                version > state_version for version in versions
            ):
                raise PersistenceConflictError("transition CAS predecessor is stale")
            connection.execute(
                "INSERT INTO records (kind, record_id, payload_json, content_sha256) "
                "VALUES ('transition', ?, ?, ?)",
                (record_id, payload, _content_hash(payload)),
            )
            audit = connection.execute(
                "SELECT payload_json FROM audit_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if audit is None:
                self._insert_audit(connection, event)
            else:
                actual = json.loads(audit[0])
                expected = event.model_dump(mode="json")
                actual.pop("created_at", None)
                expected.pop("created_at", None)
                if actual != expected:
                    raise PersistenceConflictError(f"audit event {event.event_id} content changed")

    def register_artifact(self, record: ArtifactRecord) -> None:
        payload = record.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT content_sha256 FROM authority_artifacts WHERE artifact_id = ?",
                (record.artifact_id,),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise PersistenceConflictError(f"artifact {record.artifact_id} content changed")
                return
            connection.execute(
                "INSERT INTO authority_artifacts "
                "(artifact_id, artifact_json, content_sha256, created_at) VALUES (?, ?, ?, ?)",
                (record.artifact_id, payload, digest, now),
            )
            self._insert_audit(
                connection,
                self._automatic_event(
                    "artifact_registered",
                    record.run_id,
                    record.experiment_id,
                    {"artifact_id": record.artifact_id},
                    now,
                ),
            )

    def register(self, record: ArtifactRecord) -> None:
        self.register_artifact(record)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT artifact_json FROM authority_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        return ArtifactRecord.model_validate_json(row[0]) if row else None

    def _validate_source(
        self,
        connection: sqlite3.Connection,
        experiment_id: str,
        run_id: str,
        source_commit: str,
    ) -> SourceRegistration:
        row = connection.execute(
            "SELECT registration_json FROM source_registrations WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            raise FinalTestAccessError("final test requires a registered source")
        source = SourceRegistration.model_validate_json(row[0])
        if (
            not source.eligible
            or source.run_id != run_id
            or source.source_commit != source_commit
            or not source.allowed_scopes
            or source.patch_artifact_id != f"patch-{source.patch_sha256}"
        ):
            raise FinalTestAccessError("source is not eligible for final test")
        try:
            self._validate_patch_artifact(connection, source, experiment_id)
        except ValueError as error:
            raise FinalTestAccessError(str(error)) from error
        return source

    def _validate_patch_artifact(
        self,
        connection: sqlite3.Connection,
        source: SourceRegistration,
        experiment_id: str,
    ) -> None:
        expected = (
            self.database.parent
            / "artifacts"
            / source.run_id
            / experiment_id
            / f"{source.patch_artifact_id}.diff"
        ).resolve()
        self._validate_artifact_record(
            connection,
            source.patch_artifact_id,
            source.run_id,
            experiment_id,
            "source_patch",
            expected_path=expected,
            normalized=True,
        )

    def _validate_artifact_record(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
        run_id: str,
        experiment_id: str,
        kind: str,
        *,
        expected_path: Path | None = None,
        normalized: bool = False,
    ) -> ArtifactRecord:
        artifact_row = connection.execute(
            "SELECT artifact_json FROM authority_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if artifact_row is None:
            raise ValueError("source patch artifact is not registered")
        artifact = ArtifactRecord.model_validate_json(artifact_row[0])
        expected_parent = (self.database.parent / "artifacts" / run_id / experiment_id).resolve()
        artifact_path = Path(artifact.uri.removeprefix("file://")).resolve()
        if not artifact_path.is_file():
            raise ValueError("source patch artifact is unavailable")
        if (
            artifact.kind != kind
            or artifact.artifact_id != artifact_id
            or artifact.run_id != run_id
            or artifact.experiment_id != experiment_id
            or (
                artifact_path != expected_path
                if expected_path is not None
                else artifact_path.parent != expected_parent / artifact_id
            )
            or artifact.size_bytes != artifact_path.stat().st_size
        ):
            raise ValueError("source patch artifact provenance is invalid")
        content = artifact_path.read_bytes()
        digest = (
            patch_signature(content.decode(encoding="utf-8"))
            if normalized
            else hashlib.sha256(content).hexdigest()
        )
        if artifact.sha256 != digest:
            raise ValueError("source patch artifact checksum mismatch")
        return artifact

    @staticmethod
    def _automatic_event(
        event_type: str,
        run_id: str,
        experiment_id: str | None,
        payload: dict[str, object],
        now: str,
    ) -> AuditEvent:
        event_id = f"authority-{event_type}-{run_id}-{_content_hash(str(payload))[:16]}"
        return AuditEvent(
            event_id=event_id,
            run_id=run_id,
            experiment_id=experiment_id,
            event_type=event_type,
            actor_type="controller",
            actor_id="application-repository",
            payload=payload,
            created_at=datetime.fromisoformat(now),
        )

    @staticmethod
    def _resolution_id(report_id: str, blocker_id: str, evidence_refs: tuple[str, ...]) -> str:
        material = json.dumps(
            {
                "report_id": report_id,
                "blocker_id": blocker_id,
                "evidence_refs": evidence_refs,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"resolution-{_content_hash(material)}"

    @staticmethod
    def _criterion_occurrence_id(report_id: str, criterion_id: str) -> str:
        material = json.dumps(
            {"report_id": report_id, "criterion_id": criterion_id},
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"criterion-occurrence-{_content_hash(material)}"

    @staticmethod
    def _resolution_claim_id(report_id: str, criterion_id: str) -> str:
        material = json.dumps(
            {"report_id": report_id, "criterion_id": criterion_id},
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"resolution-claim-{_content_hash(material)}"

    @staticmethod
    def _criterion_assessment_status(
        assessment: ImplementationCriterionAssessment,
    ) -> str:
        return assessment.status.value

    def _persist_criterion_occurrence(
        self,
        connection: sqlite3.Connection,
        report: ValidationReport,
        assessment: ImplementationCriterionAssessment,
        blocker_ids: tuple[str, ...],
        now: str,
    ) -> None:
        criterion_id = str(assessment.criterion_id)
        assessment_payload = assessment.model_dump_json()
        digest = _content_hash(assessment_payload)
        occurrence_id = self._criterion_occurrence_id(report.report_id, criterion_id)
        existing = connection.execute(
            "SELECT assessment_json, content_sha256 FROM "
            "authority_validation_criterion_occurrences WHERE occurrence_id = ?",
            (occurrence_id,),
        ).fetchone()
        if existing is not None:
            if existing[1] != digest:
                raise PersistenceConflictError(
                    f"criterion occurrence {occurrence_id} content changed"
                )
        else:
            connection.execute(
                "INSERT INTO authority_validation_criterion_occurrences "
                "(occurrence_id, report_id, experiment_id, stage, criterion_id, status, "
                "assessment_json, content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    occurrence_id,
                    report.report_id,
                    report.experiment_id,
                    report.stage.value,
                    criterion_id,
                    self._criterion_assessment_status(assessment),
                    assessment_payload,
                    digest,
                    now,
                ),
            )
        for blocker_id in blocker_ids:
            blocker = connection.execute(
                "SELECT 1 FROM authority_validation_blockers WHERE blocker_id = ?",
                (blocker_id,),
            ).fetchone()
            if blocker is None:
                raise ValueError(f"validation blocker {blocker_id} does not exist")
            connection.execute(
                "INSERT OR IGNORE INTO authority_validation_criterion_occurrence_blockers "
                "(occurrence_id, blocker_id) VALUES (?, ?)",
                (occurrence_id, blocker_id),
            )

    def _persist_resolution_claim(
        self,
        connection: sqlite3.Connection,
        report: ValidationReport,
        claim: CriterionResolutionClaim,
        now: str,
    ) -> None:
        criterion_id = str(claim.criterion_id)
        payload = claim.model_dump_json()
        digest = _content_hash(payload)
        claim_id = self._resolution_claim_id(report.report_id, criterion_id)
        existing = connection.execute(
            "SELECT claim_json, content_sha256 FROM authority_validation_resolution_claims "
            "WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if existing is not None:
            if existing[1] != digest:
                raise PersistenceConflictError(f"resolution claim {claim_id} content changed")
        else:
            connection.execute(
                "INSERT INTO authority_validation_resolution_claims "
                "(claim_id, report_id, experiment_id, stage, criterion_id, status, claim_json, "
                "content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    report.report_id,
                    report.experiment_id,
                    report.stage.value,
                    criterion_id,
                    claim.status.value,
                    payload,
                    digest,
                    now,
                ),
            )

    def _put_validation_operation(
        self,
        connection: sqlite3.Connection,
        operation: ValidationOperationIdentity,
        subject_json: str,
        now: str,
    ) -> None:
        operation_json = operation.model_dump_json()
        digest = _content_hash(operation_json + subject_json)
        subject_for_identity = dict(json.loads(subject_json))
        subject_for_identity.pop("validation_operation", None)
        if (
            _content_hash(json.dumps(subject_for_identity, sort_keys=True, separators=(",", ":")))
            != operation.subject_sha256
        ):
            raise PersistenceConflictError("validation subject identity does not match operation")
        authority = subject_for_identity.get("implementation_authority")
        if operation.stage == ValidationStage.IMPLEMENTATION:
            if not isinstance(authority, dict):
                raise PersistenceConflictError(
                    "implementation validation requires implementation authority"
                )
            try:
                parsed_authority = ImplementationValidationAuthority.model_validate(authority)
            except ValueError as error:
                raise PersistenceConflictError(
                    "implementation validation authority is invalid"
                ) from error
            if parsed_authority.diff_sha256 != operation.implementation_diff_sha256:
                raise PersistenceConflictError("validation authority diff does not match operation")
            if parsed_authority.evidence_id != (
                f"implementation-diff-{parsed_authority.diff_sha256}"
            ):
                raise PersistenceConflictError(
                    "implementation authority evidence is not controller-derived"
                )
        existing = connection.execute(
            "SELECT content_sha256 FROM authority_validation_operations WHERE operation_id = ?",
            (operation.operation_id,),
        ).fetchone()
        if existing is not None:
            if existing[0] != digest:
                raise PersistenceConflictError(
                    f"validation operation {operation.operation_id} content changed"
                )
            return
        connection.execute(
            "INSERT INTO authority_validation_operations "
            "(operation_id, run_id, experiment_id, stage, repair_attempt, subject_sha256, "
            "implementation_diff_sha256, operation_json, subject_json, content_sha256, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                operation.operation_id,
                operation.run_id,
                operation.experiment_id,
                operation.stage.value,
                operation.repair_attempt,
                operation.subject_sha256,
                operation.implementation_diff_sha256,
                operation_json,
                subject_json,
                digest,
                now,
            ),
        )

    def put_validation_report(
        self,
        report: ValidationReport,
        run_id: str,
        operation: ValidationOperationIdentity,
        subject: dict[str, object],
    ) -> None:
        """Append a validation report and its blocker ledger entries atomically."""
        if (
            operation.run_id != run_id
            or operation.experiment_id != report.experiment_id
            or operation.stage != report.stage
        ):
            raise ValueError("validation operation identity does not match report")
        if operation.stage == ValidationStage.IMPLEMENTATION and (
            operation.implementation_diff_sha256 is None
        ):
            raise ValueError("implementation validation operation requires a diff identity")
        if operation.stage != ValidationStage.IMPLEMENTATION and (
            operation.implementation_diff_sha256 is not None
        ):
            raise ValueError("only implementation validation may carry a diff identity")
        if report.validation_operation_id not in {"", operation.operation_id}:
            raise PersistenceConflictError("validation report is bound to another operation")
        if set(report.resolves_blocker_ids) & {b.blocker_id for b in report.blockers}:
            raise ValueError("a validation report cannot resolve a blocker it introduces")
        bound_report = report.model_copy(update={"validation_operation_id": operation.operation_id})
        payload = bound_report.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        subject_json = json.dumps(subject or {}, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._put_validation_operation(connection, operation, subject_json, now)
            existing = connection.execute(
                "SELECT report_json, content_sha256 FROM authority_validation_reports "
                "WHERE report_id = ?",
                (report.report_id,),
            ).fetchone()
            if existing is not None and existing[1] != digest:
                raise PersistenceConflictError(
                    f"validation report {report.report_id} content changed"
                )
            operation_report = connection.execute(
                "SELECT report_id, report_json FROM authority_validation_reports "
                "WHERE validation_operation_id = ?",
                (operation.operation_id,),
            ).fetchone()
            if operation_report is not None and operation_report[0] != report.report_id:
                raise PersistenceConflictError(
                    f"validation operation {operation.operation_id} already has a report"
                )
            report_inserted = existing is None
            if report_inserted:
                connection.execute(
                    "INSERT INTO authority_validation_reports "
                    "(report_id, experiment_id, stage, report_json, content_sha256, created_at, "
                    "validation_operation_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        report.report_id,
                        report.experiment_id,
                        report.stage.value,
                        payload,
                        digest,
                        now,
                        operation.operation_id,
                    ),
                )
            for blocker in report.blockers:
                blocker_payload = blocker.model_dump_json()
                blocker_digest = _content_hash(blocker_payload)
                blocker_row = connection.execute(
                    "SELECT blocker_json, content_sha256 FROM authority_validation_blockers "
                    "WHERE blocker_id = ?",
                    (blocker.blocker_id,),
                ).fetchone()
                if blocker_row is not None:
                    existing_blocker = ValidationBlocker.model_validate_json(blocker_row[0])
                    criterion_is_stable = (
                        blocker.criterion_id is not None
                        and existing_blocker.criterion_id == blocker.criterion_id
                        and existing_blocker.experiment_id == blocker.experiment_id
                        and existing_blocker.stage == blocker.stage
                    )
                    if not criterion_is_stable and blocker_row[1] != blocker_digest:
                        raise PersistenceConflictError(
                            f"validation blocker {blocker.blocker_id} content changed"
                        )
                    continue
                connection.execute(
                    "INSERT INTO authority_validation_blockers "
                    "(blocker_id, report_id, experiment_id, stage, blocker_json, "
                    "content_sha256, created_at, validation_operation_id, criterion_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        blocker.blocker_id,
                        report.report_id,
                        blocker.experiment_id,
                        blocker.stage.value,
                        blocker_payload,
                        blocker_digest,
                        now,
                        operation.operation_id,
                        blocker.criterion_id,
                    ),
                )
                self._insert_audit(
                    connection,
                    self._automatic_event(
                        "validation_blocker_persisted",
                        run_id,
                        report.experiment_id,
                        {"report_id": report.report_id, "blocker_id": blocker.blocker_id},
                        now,
                    ),
                )
            if report_inserted:
                self._insert_audit(
                    connection,
                    self._automatic_event(
                        "validation_report_persisted",
                        run_id,
                        report.experiment_id,
                        {
                            "report_id": report.report_id,
                            "stage": report.stage.value,
                            "verdict": report.verdict.value,
                        },
                        now,
                    ),
                )
            criterion_assessments = list(report.criterion_assessments)
            assessment_ids = [str(assessment.criterion_id) for assessment in criterion_assessments]
            if len(set(assessment_ids)) != len(assessment_ids):
                raise ValueError("validation report cannot assess a criterion more than once")
            assessment_by_criterion = {
                str(assessment.criterion_id): assessment for assessment in criterion_assessments
            }
            for blocker in report.blockers:
                if blocker.criterion_id is not None and str(blocker.criterion_id) not in (
                    assessment_by_criterion
                ):
                    assessment = ImplementationCriterionAssessment(
                        criterion_id=blocker.criterion_id,
                        status=CriterionAssessmentStatus.FAIL,
                        evidence_refs=blocker.evidence_refs,
                        details=blocker.text,
                    )
                    criterion_assessments.append(assessment)
                    assessment_by_criterion[str(blocker.criterion_id)] = assessment
            for assessment in criterion_assessments:
                criterion = str(assessment.criterion_id)
                blocker_ids = tuple(
                    blocker.blocker_id
                    for blocker in report.blockers
                    if blocker.criterion_id is not None and str(blocker.criterion_id) == criterion
                )
                self._persist_criterion_occurrence(connection, report, assessment, blocker_ids, now)

            claim_ids = [str(claim.criterion_id) for claim in report.resolution_claims]
            if len(set(claim_ids)) != len(claim_ids):
                raise ValueError("validation report cannot claim a criterion more than once")
            for claim in report.resolution_claims:
                if not claim.evidence_refs:
                    raise ValueError("criterion resolution claims require validation evidence")
                if claim.status not in (
                    CriterionAssessmentStatus.PASS,
                    CriterionAssessmentStatus.PARTIAL,
                ):
                    raise ValueError("criterion resolution claims must be pass or partial")
                self._persist_resolution_claim(connection, report, claim, now)

            if report.resolves_blocker_ids and not report.evidence_refs:
                raise ValueError("blocker resolutions require validation evidence")
            resolutions: list[tuple[str, tuple[str, ...], str | None, str | None]] = [
                (blocker_id, report.evidence_refs, None, None)
                for blocker_id in report.resolves_blocker_ids
            ]
            resolutions.extend(
                (blocker_id, claim.evidence_refs, str(claim.criterion_id), claim.status.value)
                for claim in report.resolution_claims
                for blocker_id in claim.blocker_ids
                if claim.status == CriterionAssessmentStatus.PASS
            )
            for blocker_id, evidence_refs, criterion_id, status in resolutions:
                blocker_row = connection.execute(
                    "SELECT blocker_json, experiment_id, report_id "
                    "FROM authority_validation_blockers WHERE blocker_id = ?",
                    (blocker_id,),
                ).fetchone()
                if blocker_row is None:
                    raise ValueError(f"validation blocker {blocker_id} does not exist")
                blocker = ValidationBlocker.model_validate_json(blocker_row[0])
                if blocker.experiment_id != report.experiment_id:
                    raise ValueError("blocker resolution experiment does not match report")
                if criterion_id is not None and str(blocker.criterion_id) != criterion_id:
                    raise ValueError("criterion resolution claim does not match blocker")
                resolution = BlockerResolution(
                    resolution_id=self._resolution_id(report.report_id, blocker_id, evidence_refs),
                    blocker_id=blocker_id,
                    report_id=report.report_id,
                    experiment_id=report.experiment_id,
                    evidence_refs=evidence_refs,
                    validation_operation_id=operation.operation_id,
                    criterion_id=blocker.criterion_id,
                    status=(CriterionAssessmentStatus(status) if status is not None else None),
                )
                resolution_payload = resolution.model_dump_json()
                resolution_digest = _content_hash(resolution_payload)
                prior = connection.execute(
                    "SELECT resolution_id, resolution_json, content_sha256 "
                    "FROM authority_blocker_resolutions WHERE resolution_id = ?",
                    (resolution.resolution_id,),
                ).fetchone()
                if prior is not None:
                    if prior[2] != resolution_digest:
                        raise PersistenceConflictError(
                            f"blocker resolution {resolution.resolution_id} content changed"
                        )
                    continue
                connection.execute(
                    "INSERT INTO authority_blocker_resolutions "
                    "(resolution_id, blocker_id, report_id, experiment_id, "
                    "resolution_json, content_sha256, created_at, validation_operation_id, "
                    "criterion_id, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        resolution.resolution_id,
                        resolution.blocker_id,
                        resolution.report_id,
                        resolution.experiment_id,
                        resolution_payload,
                        resolution_digest,
                        now,
                        operation.operation_id,
                        resolution.criterion_id,
                        resolution.status,
                    ),
                )
                self._insert_audit(
                    connection,
                    self._automatic_event(
                        "validation_blocker_resolved",
                        run_id,
                        report.experiment_id,
                        {
                            "resolution_id": resolution.resolution_id,
                            "blocker_id": blocker_id,
                            "report_id": report.report_id,
                        },
                        now,
                    ),
                )

    def put_validation_blocker(self, blocker: ValidationBlocker, run_id: str) -> None:
        report = self.get_validation_report(blocker.report_id)
        if report is None:
            raise ValueError("validation blocker requires a persisted report")
        if blocker not in report.blockers:
            raise ValueError("validation blocker does not belong to its persisted report")
        if not report.validation_operation_id:
            raise ValueError("validation blocker requires a controller operation")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT operation_json, subject_json FROM authority_validation_operations "
                "WHERE operation_id = ?",
                (report.validation_operation_id,),
            ).fetchone()
        if row is None:
            raise ValueError("validation blocker operation authority is absent")
        self.put_validation_report(
            report,
            run_id,
            ValidationOperationIdentity.model_validate_json(row[0]),
            json.loads(row[1]),
        )

    def put_blocker_resolution(self, resolution: BlockerResolution, run_id: str) -> None:
        if resolution.status == CriterionAssessmentStatus.PARTIAL:
            raise ValueError("partial criterion claims do not resolve blockers")
        report = self.get_validation_report(resolution.report_id)
        if report is None or report.verdict.value not in {"approved", "repairable"}:
            raise ValueError(
                "blocker resolution requires an approved or repairable validation report"
            )
        claimed_ids = {
            blocker_id for claim in report.resolution_claims for blocker_id in claim.blocker_ids
        }
        partially_claimed_ids = {
            blocker_id
            for claim in report.resolution_claims
            if claim.status == CriterionAssessmentStatus.PARTIAL
            for blocker_id in claim.blocker_ids
        }
        if resolution.blocker_id in partially_claimed_ids:
            raise ValueError("partial criterion claims do not resolve blockers")
        if resolution.blocker_id not in (*report.resolves_blocker_ids, *claimed_ids):
            raise ValueError("resolution is not authorized by its validation report")
        if (
            resolution.experiment_id != report.experiment_id
            or not resolution.evidence_refs
            or not resolution.validation_operation_id
            or resolution.validation_operation_id != report.validation_operation_id
        ):
            raise ValueError("blocker resolution identity or evidence is invalid")
        payload = resolution.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            blocker = connection.execute(
                "SELECT blocker_json, experiment_id, report_id FROM authority_validation_blockers "
                "WHERE blocker_id = ?",
                (resolution.blocker_id,),
            ).fetchone()
            if blocker is None:
                raise ValueError("validation blocker does not exist")
            if blocker[1] != report.experiment_id:
                raise ValueError("blocker resolution experiment does not match blocker")
            blocker_model = ValidationBlocker.model_validate_json(blocker[0])
            if (
                resolution.criterion_id is not None
                and resolution.criterion_id != blocker_model.criterion_id
            ):
                raise ValueError("blocker resolution criterion does not match blocker")
            same_id = connection.execute(
                "SELECT resolution_json, content_sha256 FROM authority_blocker_resolutions "
                "WHERE resolution_id = ?",
                (resolution.resolution_id,),
            ).fetchone()
            if same_id is not None:
                if same_id[1] != digest:
                    raise PersistenceConflictError(
                        f"blocker resolution {resolution.resolution_id} content changed"
                    )
                return
            connection.execute(
                "INSERT INTO authority_blocker_resolutions "
                "(resolution_id, blocker_id, report_id, experiment_id, resolution_json, "
                "content_sha256, created_at, validation_operation_id, criterion_id, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    resolution.resolution_id,
                    resolution.blocker_id,
                    resolution.report_id,
                    resolution.experiment_id,
                    payload,
                    digest,
                    now,
                    resolution.validation_operation_id,
                    resolution.criterion_id,
                    resolution.status,
                ),
            )
            self._insert_audit(
                connection,
                self._automatic_event(
                    "validation_blocker_resolved",
                    run_id,
                    resolution.experiment_id,
                    {
                        "resolution_id": resolution.resolution_id,
                        "blocker_id": resolution.blocker_id,
                        "report_id": resolution.report_id,
                    },
                    now,
                ),
            )

    def get_validation_report(self, report_id: str) -> ValidationReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM authority_validation_reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return ValidationReport.model_validate_json(row[0]) if row else None

    def get_validation_report_by_operation(self, operation_id: str) -> ValidationReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM authority_validation_reports "
                "WHERE validation_operation_id = ?",
                (operation_id,),
            ).fetchone()
        return ValidationReport.model_validate_json(row[0]) if row else None

    def get_validation_report_for_attempt(
        self, run_id: str, experiment_id: str, stage: ValidationStage, repair_attempt: int
    ) -> ValidationReport | None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT report.report_json FROM authority_validation_reports AS report "
                "JOIN authority_validation_operations AS operation "
                "ON operation.operation_id = report.validation_operation_id "
                "WHERE operation.run_id = ? AND operation.experiment_id = ? "
                "AND operation.stage = ? AND operation.repair_attempt = ?",
                (run_id, experiment_id, stage.value, repair_attempt),
            ).fetchall()
        if len(rows) > 1:
            raise PersistenceConflictError("validation attempt has multiple authoritative reports")
        return ValidationReport.model_validate_json(rows[0][0]) if rows else None

    def get_validation_operation(self, operation_id: str) -> ValidationOperationIdentity | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT operation_json FROM authority_validation_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return ValidationOperationIdentity.model_validate_json(row[0]) if row else None

    def list_validation_reports(
        self, experiment_id: str | None = None
    ) -> tuple[ValidationReport, ...]:
        query = "SELECT report_json FROM authority_validation_reports"
        parameters: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            parameters = (experiment_id,)
        query += " ORDER BY created_at, report_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(ValidationReport.model_validate_json(row[0]) for row in rows)

    def list_validation_blockers(
        self, experiment_id: str | None = None
    ) -> tuple[ValidationBlocker, ...]:
        query = "SELECT blocker_json FROM authority_validation_blockers"
        parameters: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            parameters = (experiment_id,)
        query += " ORDER BY created_at, blocker_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(ValidationBlocker.model_validate_json(row[0]) for row in rows)

    def get_validation_blocker(self, blocker_id: str) -> ValidationBlocker | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT blocker_json FROM authority_validation_blockers WHERE blocker_id = ?",
                (blocker_id,),
            ).fetchone()
        return ValidationBlocker.model_validate_json(row[0]) if row else None

    def list_blocker_resolutions(
        self, experiment_id: str | None = None
    ) -> tuple[BlockerResolution, ...]:
        query = "SELECT resolution_json FROM authority_blocker_resolutions"
        parameters: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            parameters = (experiment_id,)
        query += " ORDER BY created_at, resolution_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(BlockerResolution.model_validate_json(row[0]) for row in rows)

    def get_blocker_resolution(self, resolution_id: str) -> BlockerResolution | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT resolution_json FROM authority_blocker_resolutions WHERE resolution_id = ?",
                (resolution_id,),
            ).fetchone()
        return BlockerResolution.model_validate_json(row[0]) if row else None

    def get_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT blocker.blocker_json, blocker.blocker_id, blocker.stage, "
                "blocker.criterion_id "
                "FROM authority_validation_blockers AS blocker "
                "WHERE blocker.experiment_id = ? "
                "ORDER BY blocker.created_at, blocker.blocker_id",
                (experiment_id,),
            ).fetchall()
            unresolved: list[ValidationBlocker] = []
            for blocker_json, blocker_id, stage, criterion_id in rows:
                if criterion_id is None:
                    resolved = connection.execute(
                        "SELECT 1 FROM authority_blocker_resolutions "
                        "WHERE blocker_id = ? AND (status IS NULL OR status = 'pass') LIMIT 1",
                        (blocker_id,),
                    ).fetchone()
                    if resolved is None:
                        unresolved.append(ValidationBlocker.model_validate_json(blocker_json))
                    continue

                occurrence = connection.execute(
                    "SELECT occurrence.report_id, occurrence.status, occurrence.assessment_json "
                    "FROM authority_validation_criterion_occurrences AS occurrence "
                    "WHERE occurrence.experiment_id = ? AND occurrence.stage = ? "
                    "AND occurrence.criterion_id = ? "
                    "ORDER BY occurrence.created_at DESC, occurrence.occurrence_id DESC LIMIT 1",
                    (experiment_id, stage, criterion_id),
                ).fetchone()
                if occurrence is None:
                    resolved = connection.execute(
                        "SELECT 1 FROM authority_blocker_resolutions "
                        "WHERE blocker_id = ? AND (status IS NULL OR status = 'pass') LIMIT 1",
                        (blocker_id,),
                    ).fetchone()
                elif occurrence[1] != CriterionAssessmentStatus.PASS.value:
                    # A newer failed or partial assessment reopens the stable
                    # blocker, even when an older PASS resolution is retained.
                    resolved = None
                else:
                    assessment = json.loads(occurrence[2])
                    has_evidence = bool(assessment.get("evidence_refs"))
                    resolved = connection.execute(
                        "SELECT 1 FROM authority_blocker_resolutions "
                        "WHERE blocker_id = ? AND report_id = ? "
                        "AND (status IS NULL OR status = 'pass') LIMIT 1",
                        (blocker_id, occurrence[0]),
                    ).fetchone()
                    if resolved is None:
                        resolved = connection.execute(
                            "SELECT 1 FROM authority_validation_resolution_claims "
                            "WHERE report_id = ? AND criterion_id = ? AND status = 'pass' "
                            "AND json_array_length(json_extract(claim_json, "
                            "'$.evidence_refs')) > 0 "
                            "LIMIT 1",
                            (occurrence[0], criterion_id),
                        ).fetchone()
                    if resolved is None and has_evidence:
                        # A PASS occurrence is itself an evidence-backed
                        # resolution; claims remain available as the explicit
                        # append-only audit record when supplied.
                        resolved = (1,)
                if resolved is None:
                    unresolved.append(ValidationBlocker.model_validate_json(blocker_json))
        return tuple(unresolved)

    def get_unresolved_blocker_ids(self, experiment_id: str) -> tuple[str, ...]:
        return tuple(blocker.blocker_id for blocker in self.get_unresolved_blockers(experiment_id))

    def list_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        return self.get_unresolved_blockers(experiment_id)

    def get_criterion_repeat_count(
        self,
        experiment_id: str,
        criterion_id: str,
        stage: ValidationStage = ValidationStage.IMPLEMENTATION,
    ) -> int:
        """Return the number of failed or partial criterion occurrences.

        Occurrences are keyed by report and criterion, so replaying an identical
        report cannot inflate the count.  Passing assessments are deliberately not
        escalation signals.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM authority_validation_criterion_occurrences "
                "WHERE experiment_id = ? AND criterion_id = ? "
                "AND stage = ? AND status IN ('fail', 'partial')",
                (experiment_id, criterion_id, stage.value),
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _attempt_identity_digest(claim: FullScientificAttemptClaim) -> str:
        value = claim.model_dump(mode="json")
        value.pop("attempt_sequence", None)
        value.pop("claimed_at", None)
        material = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return _content_hash(material)

    @staticmethod
    def _observation_identity_digest(
        observation: ScoredObservation | ScoredObservationRequest,
    ) -> str:
        value = observation.model_dump(mode="json")
        value.pop("scored_at", None)
        material = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return _content_hash(material)

    def _verified_record(self, connection: sqlite3.Connection, kind: str, record_id: str) -> str:
        row = connection.execute(
            "SELECT payload_json, content_sha256 FROM records WHERE kind = ? AND record_id = ?",
            (kind, record_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"persisted {kind} {record_id} is absent")
        if row[1] is None or _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError(f"persisted {kind} {record_id} integrity check failed")
        return row[0]

    @staticmethod
    def _indexed_identity(
        connection: sqlite3.Connection,
        table: str,
        identity_column: str,
        identity: str,
        columns: tuple[str, ...],
        expected: tuple[object, ...],
    ) -> tuple[object, ...]:
        selected = ", ".join(columns)
        row = connection.execute(
            f"SELECT {selected} FROM {table} WHERE {identity_column} = ?", (identity,)
        ).fetchone()
        if row is None or tuple(row) != expected:
            raise PersistenceConflictError(f"persisted {table} identity columns are inconsistent")
        return row

    def _validate_attempt_source(
        self, connection: sqlite3.Connection, request: FullAttemptClaimRequest
    ) -> None:
        row = self._indexed_identity(
            connection,
            "source_registrations",
            "registration_id",
            request.source_registration_id,
            ("registration_id", "experiment_id", "run_id", "source_commit", "eligible"),
            (
                request.source_registration_id,
                request.experiment_id,
                request.run_id,
                request.source_commit,
                1,
            ),
        )
        source_row = connection.execute(
            "SELECT registration_json, content_sha256 FROM source_registrations "
            "WHERE registration_id = ?",
            (request.source_registration_id,),
        ).fetchone()
        if source_row is None or source_row[1] is None:
            raise PersistenceConflictError("source registration integrity check failed")
        if _content_hash(source_row[0]) != source_row[1]:
            raise PersistenceConflictError("source registration integrity check failed")
        source = SourceRegistration.model_validate_json(source_row[0])
        if (
            tuple(row)
            != (
                source.registration_id,
                source.experiment_id,
                source.run_id,
                source.source_commit,
                1 if source.eligible else 0,
            )
            or not source.eligible
            or source.registration_id != request.source_registration_id
            or source.experiment_id != request.experiment_id
            or source.run_id != request.run_id
            or source.source_commit != request.source_commit
        ):
            raise ValueError("claim source registration is not the exact eligible source")

    def claim_full_attempt(
        self, request: FullAttemptClaimRequest
    ) -> FullScientificAttemptClaim | None:
        """Atomically claim the next fresh full attempt for a run.

        The sequence is assigned while holding ``BEGIN IMMEDIATE``.  A replay is
        keyed by the stable attempt/execution identity and returns the original
        authority record without consuming another sequence.
        """
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT claim_json, identity_sha256, content_sha256, attempt_id, execution_id, "
                "run_id, experiment_id, source_registration_id, source_commit "
                "FROM authority_full_attempt_claims "
                "WHERE execution_id = ? OR attempt_id = ?",
                (request.execution_id, request.attempt_id),
            ).fetchone()
            if existing is not None:
                if existing[2] is None or _content_hash(existing[0]) != existing[2]:
                    raise PersistenceConflictError(
                        "persisted full attempt claim integrity check failed"
                    )
                existing_claim = FullScientificAttemptClaim.model_validate_json(existing[0])
                if tuple(existing[3:]) != (
                    existing_claim.attempt_id,
                    existing_claim.execution_id,
                    existing_claim.run_id,
                    existing_claim.experiment_id,
                    existing_claim.source_registration_id,
                    existing_claim.source_commit,
                ):
                    raise PersistenceConflictError(
                        "persisted full attempt claim identity is inconsistent"
                    )
                identity = self._attempt_identity_digest(existing_claim)
                requested = FullScientificAttemptClaim(
                    **request.model_dump(),
                    attempt_sequence=existing_claim.attempt_sequence,
                    claimed_at=existing_claim.claimed_at,
                )
                if existing[1] != identity or identity != self._attempt_identity_digest(requested):
                    raise PersistenceConflictError(
                        f"full attempt {request.execution_id} identity changed"
                    )
                return existing_claim

            if connection.execute(
                "SELECT 1 FROM authority_run_closures WHERE run_id = ?", (request.run_id,)
            ).fetchone():
                raise PersistenceConflictError("run is already closed")
            self._validate_attempt_source(connection, request)
            next_sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(attempt_sequence), 0) + 1 "
                    "FROM authority_full_attempt_claims WHERE run_id = ?",
                    (request.run_id,),
                ).fetchone()[0]
            )
            if next_sequence > MAX_FULL_ATTEMPTS:
                return None
            record = FullScientificAttemptClaim(
                **request.model_dump(),
                attempt_sequence=next_sequence,
                claimed_at=datetime.now(UTC),
            )
            payload = record.model_dump_json()
            identity_digest = self._attempt_identity_digest(record)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO authority_full_attempt_claims "
                "(attempt_id, execution_id, run_id, experiment_id, source_registration_id, "
                "source_commit, attempt_sequence, max_attempts, attempt_policy_id, claim_json, "
                "identity_sha256, content_sha256, claimed_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.attempt_id,
                    record.execution_id,
                    record.run_id,
                    record.experiment_id,
                    record.source_registration_id,
                    record.source_commit,
                    record.attempt_sequence,
                    record.max_attempts,
                    record.attempt_policy_id,
                    payload,
                    identity_digest,
                    _content_hash(payload),
                    record.claimed_at.isoformat(),
                    now,
                ),
            )
            self._insert_audit(
                connection,
                self._automatic_event(
                    "full_attempt_claimed",
                    record.run_id,
                    record.experiment_id,
                    {
                        "attempt_id": record.attempt_id,
                        "execution_id": record.execution_id,
                        "attempt_sequence": record.attempt_sequence,
                    },
                    now,
                ),
            )
            return record

    def count_full_attempt_claims(self, run_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM authority_full_attempt_claims WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row[0])

    def get_full_attempt_claim(self, attempt_id: str) -> FullScientificAttemptClaim | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT claim_json, content_sha256, attempt_id, execution_id, run_id, "
                "experiment_id, source_registration_id, source_commit "
                "FROM authority_full_attempt_claims "
                "WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            return None
        if row[1] is None or _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError("persisted full attempt claim integrity check failed")
        claim = FullScientificAttemptClaim.model_validate_json(row[0])
        if tuple(row[2:]) != (
            claim.attempt_id,
            claim.execution_id,
            claim.run_id,
            claim.experiment_id,
            claim.source_registration_id,
            claim.source_commit,
        ):
            raise PersistenceConflictError("persisted full attempt claim identity is inconsistent")
        return claim

    def list_full_attempt_claims(
        self, run_id: str | None = None
    ) -> tuple[FullScientificAttemptClaim, ...]:
        query = (
            "SELECT claim_json, content_sha256, attempt_id, execution_id, run_id, "
            "experiment_id, source_registration_id, source_commit "
            "FROM authority_full_attempt_claims"
        )
        parameters: tuple[object, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            parameters = (run_id,)
        query += " ORDER BY run_id, attempt_sequence, attempt_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        claims: list[FullScientificAttemptClaim] = []
        for row in rows:
            if row[1] is None or _content_hash(row[0]) != row[1]:
                raise PersistenceConflictError(
                    "persisted full attempt claim integrity check failed"
                )
            claim = FullScientificAttemptClaim.model_validate_json(row[0])
            if tuple(
                (
                    claim.attempt_id,
                    claim.execution_id,
                    claim.run_id,
                    claim.experiment_id,
                    claim.source_registration_id,
                    claim.source_commit,
                )
            ) != tuple(row[2:]):
                raise PersistenceConflictError(
                    "persisted full attempt claim identity is inconsistent"
                )
            claims.append(claim)
        return tuple(claims)

    def put_scored_observation(
        self, request: ScoredObservationRequest
    ) -> ScoredObservation:
        """Persist one approved score as an immutable, idempotent authority record."""
        identity_digest = self._observation_identity_digest(request)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT observation_json, identity_sha256, content_sha256, run_id, experiment_id, "
                "attempt_id, evaluation_id, checkpoint_id FROM authority_scored_observations "
                "WHERE observation_id = ?",
                (request.observation_id,),
            ).fetchone()
            if existing is not None:
                if existing[2] is None or _content_hash(existing[0]) != existing[2]:
                    raise PersistenceConflictError(
                        "persisted scored observation integrity check failed"
                    )
                existing_observation = ScoredObservation.model_validate_json(existing[0])
                if tuple(existing[3:]) != (
                    existing_observation.run_id,
                    existing_observation.experiment_id,
                    existing_observation.attempt_id,
                    existing_observation.evaluation_id,
                    existing_observation.checkpoint_id,
                ):
                    raise PersistenceConflictError(
                        "persisted scored observation identity is inconsistent"
                    )
                if existing[1] != identity_digest:
                    raise PersistenceConflictError(
                        f"scored observation {request.observation_id} content changed"
                    )
                return existing_observation
            observation = ScoredObservation(
                **request.model_dump(),
                scored_at=datetime.now(UTC),
            )
            payload = observation.model_dump_json()
            now = datetime.now(UTC).isoformat()
            self._validate_observation(connection, observation)
            if connection.execute(
                "SELECT 1 FROM authority_run_closures WHERE run_id = ?", (observation.run_id,)
            ).fetchone():
                raise PersistenceConflictError("run is already closed")
            duplicate = connection.execute(
                "SELECT observation_id FROM authority_scored_observations "
                "WHERE attempt_id = ? OR evaluation_id = ?",
                (observation.attempt_id, observation.evaluation_id),
            ).fetchone()
            if duplicate is not None:
                raise PersistenceConflictError(
                    f"attempt or evaluation already has observation {duplicate[0]}"
                )
            connection.execute(
                "INSERT INTO authority_scored_observations "
                "(observation_id, run_id, experiment_id, attempt_id, evaluation_id, checkpoint_id, "
                "observation_json, identity_sha256, content_sha256, scored_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    observation.observation_id,
                    observation.run_id,
                    observation.experiment_id,
                    observation.attempt_id,
                    observation.evaluation_id,
                    observation.checkpoint_id,
                    payload,
                    identity_digest,
                    _content_hash(payload),
                    observation.scored_at.isoformat(),
                    now,
                ),
            )
            self._insert_audit(
                connection,
                self._automatic_event(
                    "scored_observation_persisted",
                    observation.run_id,
                    observation.experiment_id,
                    {"observation_id": observation.observation_id},
                    now,
                ),
            )
            return observation

    def _validate_observation(
        self, connection: sqlite3.Connection, observation: ScoredObservation
    ) -> None:
        attempt_row = connection.execute(
            "SELECT claim_json, content_sha256, attempt_id, run_id, experiment_id, "
            "execution_id, source_registration_id, source_commit, attempt_sequence "
            "FROM authority_full_attempt_claims WHERE attempt_id = ?",
            (observation.attempt_id,),
        ).fetchone()
        if attempt_row is None:
            raise ValueError("scored observation requires a persisted full attempt")
        if attempt_row[1] is None or _content_hash(attempt_row[0]) != attempt_row[1]:
            raise PersistenceConflictError("persisted full attempt claim integrity check failed")
        attempt = FullScientificAttemptClaim.model_validate_json(attempt_row[0])
        if tuple(attempt_row[2:]) != (
            attempt.attempt_id,
            attempt.run_id,
            attempt.experiment_id,
            attempt.execution_id,
            attempt.source_registration_id,
            attempt.source_commit,
            attempt.attempt_sequence,
        ):
            raise PersistenceConflictError("persisted full attempt claim identity is inconsistent")
        if (
            observation.run_id != attempt.run_id
            or observation.experiment_id != attempt.experiment_id
            or observation.execution_id != attempt.execution_id
            or observation.source_commit != attempt.source_commit
        ):
            raise ValueError("scored observation does not match its attempt provenance")
        self._validate_attempt_source(
            connection,
            FullAttemptClaimRequest(
                attempt_id=attempt.attempt_id,
                execution_id=attempt.execution_id,
                run_id=attempt.run_id,
                experiment_id=attempt.experiment_id,
                source_registration_id=attempt.source_registration_id,
                source_commit=attempt.source_commit,
            ),
        )

        execution = ExecutionResult.model_validate_json(
            self._verified_record(connection, "execution", observation.execution_id)
        )
        if (
            execution.execution_id != attempt.execution_id
            or execution.experiment_id != attempt.experiment_id
            or execution.source_registration_id != attempt.source_registration_id
            or execution.source_commit != attempt.source_commit
            or execution.execution_kind != "full"
            or execution.exit_code != 0
            or execution.checkpoint_id != observation.checkpoint_id
            or execution.dataset_manifest_id != observation.dataset_manifest_id
            or execution.dataset_manifest_sha256 != observation.dataset_manifest_sha256
        ):
            raise ValueError("scored observation execution provenance is invalid")

        evaluation_payload = json.loads(
            self._verified_record(connection, "evaluation", observation.evaluation_id)
        )
        if not isinstance(evaluation_payload, dict) or "result" not in evaluation_payload:
            raise ValueError("scored observation requires a typed persisted evaluation")
        evaluation = EvaluationResult.model_validate(evaluation_payload["result"])
        provenance = ProvenanceRequest.model_validate(evaluation_payload.get("provenance"))
        if (
            evaluation.evaluation_id != observation.evaluation_id
            or evaluation.experiment_id != observation.experiment_id
            or evaluation.checkpoint_id != observation.checkpoint_id
            or evaluation.run_id != observation.run_id
            or evaluation.source_commit != observation.source_commit
            or evaluation.execution_id != observation.execution_id
            or evaluation.dataset_manifest_id != observation.dataset_manifest_id
            or evaluation.dataset_manifest_sha256 != observation.dataset_manifest_sha256
            or evaluation.split != observation.split
            or evaluation.validity != observation.validity
            or evaluation.evaluator_artifact_id != observation.evaluator_id
            or evaluation.evaluator_sha256 != observation.evaluator_sha256
            or provenance.run_id != observation.run_id
            or provenance.experiment_id != observation.experiment_id
            or provenance.source_commit != observation.source_commit
            or provenance.execution_id != observation.execution_id
            or provenance.dataset_manifest_id != observation.dataset_manifest_id
            or provenance.dataset_manifest_sha256 != observation.dataset_manifest_sha256
            or provenance.evaluator_id != observation.evaluator_id
            or provenance.evaluator_sha256 != observation.evaluator_sha256
        ):
            raise ValueError("scored observation evaluation provenance is invalid")
        metrics = {metric.name: metric.value for metric in evaluation.metrics}
        if set(metrics) != {"GAUC", "nDCG@5"} or len(metrics) != 2:
            raise ValueError("scored observation requires current GAUC and nDCG@5 metrics")
        from decimal import Decimal

        computed_score = (Decimal(str(metrics["GAUC"])) + Decimal(str(metrics["nDCG@5"]))) / 2
        if computed_score != Decimal(str(observation.primary_score)):
            raise ValueError("scored observation primary score is not the metric mean")

        report_row = connection.execute(
            "SELECT report_json, content_sha256, report_id, experiment_id, stage "
            "FROM authority_validation_reports WHERE report_id = ?",
            (observation.validation_report_id,),
        ).fetchone()
        if report_row is None:
            raise ValueError("scored observation requires a persisted result validation report")
        if report_row[1] is None or _content_hash(report_row[0]) != report_row[1]:
            raise PersistenceConflictError("persisted validation report integrity check failed")
        if tuple(report_row[2:]) != (
            observation.validation_report_id,
            observation.experiment_id,
            ValidationStage.RESULT.value,
        ):
            raise PersistenceConflictError("persisted validation report identity is inconsistent")
        report = ValidationReport.model_validate_json(report_row[0])
        if (
            report.experiment_id != observation.experiment_id
            or report.stage != ValidationStage.RESULT
            or report.verdict != ValidationVerdict.APPROVED
            or not report.evidence_refs
            or tuple(report.evidence_refs) != tuple(observation.validation_evidence_refs)
        ):
            raise ValueError("scored observation result validation is not approved provenance")
        operation_id = report.validation_operation_id
        if not operation_id:
            raise ValueError("scored observation requires a bound validation operation")
        operation_row = connection.execute(
            "SELECT operation_json, subject_json, content_sha256, operation_id, run_id, "
            "experiment_id, stage, subject_sha256 FROM authority_validation_operations "
            "WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if operation_row is None:
            raise ValueError("scored observation validation operation is absent")
        operation_json, subject_json = operation_row[0], operation_row[1]
        if (
            operation_row[2] is None
            or _content_hash(operation_json + subject_json) != operation_row[2]
        ):
            raise PersistenceConflictError(
                "persisted validation operation integrity check failed"
            )
        operation = ValidationOperationIdentity.model_validate_json(operation_json)
        if tuple(operation_row[3:]) != (
            operation.operation_id,
            operation.run_id,
            operation.experiment_id,
            operation.stage.value,
            operation.subject_sha256,
        ):
            raise PersistenceConflictError(
                "persisted validation operation identity is inconsistent"
            )
        subject = json.loads(subject_json)
        if not isinstance(subject, dict):
            raise ValueError("validation operation subject must be an object")
        subject = cast(dict[str, object], subject)
        subject_identity = dict(subject)
        subject_identity.pop("validation_operation", None)
        if _content_hash(
            json.dumps(subject_identity, sort_keys=True, separators=(",", ":"))
        ) != operation.subject_sha256:
            raise PersistenceConflictError("validation operation subject hash is invalid")
        if (
            operation.run_id != observation.run_id
            or operation.experiment_id != observation.experiment_id
            or operation.stage != ValidationStage.RESULT
            or subject.get("evaluation_result") != evaluation.model_dump(mode="json")
            or subject.get("execution_result")
            != execution.model_dump(mode="json", exclude={"dataset_valid_rows"})
        ):
            raise ValueError("validation operation subject provenance is invalid")

        state = connection.execute(
            "SELECT status, experiment_id FROM experiment_states WHERE experiment_id = ? "
            "ORDER BY sequence DESC LIMIT 1",
            (observation.experiment_id,),
        ).fetchone()
        if state is None or tuple(state) != ("completed", observation.experiment_id):
            raise ValueError("scored observation requires a completed experiment")

    def get_scored_observation(self, observation_id: str) -> ScoredObservation | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT observation_json, content_sha256, run_id, experiment_id, attempt_id, "
                "evaluation_id, checkpoint_id FROM authority_scored_observations "
                "WHERE observation_id = ?",
                (observation_id,),
            ).fetchone()
        if row is None:
            return None
        if _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError("persisted scored observation integrity check failed")
        observation = ScoredObservation.model_validate_json(row[0])
        if tuple(row[2:]) != (
            observation.run_id,
            observation.experiment_id,
            observation.attempt_id,
            observation.evaluation_id,
            observation.checkpoint_id,
        ):
            raise PersistenceConflictError("persisted scored observation identity is inconsistent")
        return observation

    def list_scored_observations(self, run_id: str | None = None) -> tuple[ScoredObservation, ...]:
        query = (
            "SELECT observation.observation_json, observation.content_sha256, "
            "observation.observation_id, observation.run_id, observation.experiment_id, "
            "observation.attempt_id, observation.evaluation_id, observation.checkpoint_id "
            "FROM authority_scored_observations AS observation "
            "JOIN authority_full_attempt_claims AS attempt "
            "ON attempt.attempt_id = observation.attempt_id"
        )
        parameters: tuple[object, ...] = ()
        if run_id is not None:
            query += " WHERE observation.run_id = ?"
            parameters = (run_id,)
        query += (
            " ORDER BY observation.run_id, attempt.attempt_sequence, observation.observation_id"
        )
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        values: list[ScoredObservation] = []
        for row in rows:
            if row[1] is None or _content_hash(row[0]) != row[1]:
                raise PersistenceConflictError(
                    "persisted scored observation integrity check failed"
                )
            observation = ScoredObservation.model_validate_json(row[0])
            if tuple(row[2:]) != (
                observation.observation_id,
                observation.run_id,
                observation.experiment_id,
                observation.attempt_id,
                observation.evaluation_id,
                observation.checkpoint_id,
            ):
                raise PersistenceConflictError(
                    "persisted scored observation identity is inconsistent"
                )
            values.append(observation)
        return tuple(values)

    def close_run(
        self,
        run_id: str,
        reason: Literal["plateau", "attempt_cap"],
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> RunClosure:
        if reason not in {"plateau", "attempt_cap"}:
            raise ValueError("unsupported run closure reason")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT closure_json, content_sha256, closure_id, run_id, reason "
                "FROM authority_run_closures WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None:
                if existing[1] is None or _content_hash(existing[0]) != existing[1]:
                    raise PersistenceConflictError("persisted run closure integrity check failed")
                closure = RunClosure.model_validate_json(existing[0])
                if tuple(existing[2:]) != (closure.closure_id, closure.run_id, closure.reason):
                    raise PersistenceConflictError("persisted run closure identity is inconsistent")
                if closure.reason != reason:
                    raise PersistenceConflictError("run is already closed for another reason")
                return closure

            claim_rows = connection.execute(
                "SELECT claim_json, content_sha256, attempt_id, run_id, experiment_id, "
                "source_registration_id, source_commit, attempt_sequence "
                "FROM authority_full_attempt_claims "
                "WHERE run_id = ? ORDER BY attempt_sequence, attempt_id",
                (run_id,),
            ).fetchall()
            claims: list[FullScientificAttemptClaim] = []
            for row in claim_rows:
                if row[1] is None or _content_hash(row[0]) != row[1]:
                    raise PersistenceConflictError(
                        "persisted full attempt claim integrity check failed"
                    )
                claim = FullScientificAttemptClaim.model_validate_json(row[0])
                if tuple(row[2:]) != (
                    claim.attempt_id,
                    claim.run_id,
                    claim.experiment_id,
                    claim.source_registration_id,
                    claim.source_commit,
                    claim.attempt_sequence,
                ):
                    raise PersistenceConflictError(
                        "persisted full attempt claim identity is inconsistent"
                    )
                claims.append(claim)
            observations_rows = connection.execute(
                "SELECT observation.observation_json, observation.content_sha256, "
                "observation.observation_id, observation.run_id, observation.experiment_id, "
                "observation.attempt_id, observation.evaluation_id, observation.checkpoint_id, "
                "attempt.attempt_sequence "
                "FROM authority_scored_observations AS observation "
                "JOIN authority_full_attempt_claims AS attempt "
                "ON attempt.attempt_id = observation.attempt_id "
                "AND attempt.run_id = observation.run_id "
                "WHERE observation.run_id = ? "
                "ORDER BY attempt.attempt_sequence, observation.evaluation_id, "
                "observation.observation_id",
                (run_id,),
            ).fetchall()
            observations: list[ScoredObservation] = []
            for row in observations_rows:
                if row[1] is None or _content_hash(row[0]) != row[1]:
                    raise PersistenceConflictError(
                        "persisted scored observation integrity check failed"
                    )
                observation = ScoredObservation.model_validate_json(row[0])
                if tuple(row[2:]) != (
                    observation.observation_id,
                    observation.run_id,
                    observation.experiment_id,
                    observation.attempt_id,
                    observation.evaluation_id,
                    observation.checkpoint_id,
                    row[8],
                ):
                    raise PersistenceConflictError(
                        "persisted scored observation identity is inconsistent"
                    )
                self._validate_observation(connection, observation)
                claim = next(
                    (item for item in claims if item.attempt_id == observation.attempt_id),
                    None,
                )
                if claim is None or row[8] != claim.attempt_sequence:
                    raise PersistenceConflictError("observation attempt sequence is inconsistent")
                observations.append(observation)

            champion: object = None
            if observations:
                by_attempt = {claim.attempt_id: claim for claim in claims}
                candidates = [
                    (observation, by_attempt[observation.attempt_id])
                    for observation in observations
                    if observation.attempt_id in by_attempt
                ]
                if len(candidates) != len(observations):
                    raise PersistenceConflictError("observation has no claim in its run")
                selected, selected_attempt = min(
                    candidates,
                    key=lambda item: (
                        -item[0].primary_score,
                        item[1].attempt_sequence,
                        item[0].evaluation_id,
                    ),
                )
                champion = ChampionBinding(
                    observation_id=selected.observation_id,
                    attempt_id=selected.attempt_id,
                    execution_id=selected.execution_id,
                    evaluation_id=selected.evaluation_id,
                    checkpoint_id=selected.checkpoint_id,
                    source_commit=selected.source_commit,
                    attempt_sequence=selected_attempt.attempt_sequence,
                    primary_score=selected.primary_score,
                )

            scores = [observation.primary_score for observation in observations]
            plateau = convergence_reason(scores, epsilon=epsilon, patience=patience) == "plateau"
            if reason == "plateau" and (not plateau or champion is None):
                raise PersistenceConflictError("plateau closure is not currently satisfied")
            if reason == "attempt_cap" and len(claims) != MAX_FULL_ATTEMPTS:
                raise PersistenceConflictError("attempt-cap closure requires exactly 50 attempts")
            closure = RunClosure(
                closure_id=f"closure-{run_id}-{reason}",
                run_id=run_id,
                reason=reason,
                attempt_count=len(claims),
                scored_observation_count=len(observations),
                champion=champion,
                closed_at=datetime.now(UTC),
            )
            payload = closure.model_dump_json()
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "INSERT INTO authority_run_closures "
                "(closure_id, run_id, reason, closure_json, content_sha256, closed_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    closure.closure_id,
                    closure.run_id,
                    closure.reason,
                    payload,
                    _content_hash(payload),
                    closure.closed_at.isoformat(),
                    now,
                ),
            )
            self._insert_audit(
                connection,
                self._automatic_event(
                    "run_closed",
                    closure.run_id,
                    None,
                    {"closure_id": closure.closure_id, "reason": closure.reason},
                    now,
                ),
            )
            return closure

    def close_run_if_ready(
        self,
        run_id: str,
        *,
        after_failure: bool = False,
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> RunClosure | None:
        """Derive the next authoritative closure without trusting graph state."""
        existing = self.get_run_closure(run_id)
        if existing is not None:
            return existing
        claim_count = self.count_full_attempt_claims(run_id)
        if not after_failure and claim_count >= patience + 1:
            observations = self.list_scored_observations(run_id)
            scores = [observation.primary_score for observation in observations]
            if convergence_reason(scores, epsilon=epsilon, patience=patience) == "plateau":
                return self.close_run(
                    run_id, "plateau", epsilon=epsilon, patience=patience
                )
        if claim_count == MAX_FULL_ATTEMPTS:
            return self.close_run(
                run_id, "attempt_cap", epsilon=epsilon, patience=patience
            )
        return None

    def _insert_adopted_claim(
        self, connection: sqlite3.Connection, request: FullAttemptClaimRequest
    ) -> FullScientificAttemptClaim:
        self._validate_attempt_source(connection, request)
        existing = connection.execute(
            "SELECT claim_json, content_sha256 FROM authority_full_attempt_claims "
            "WHERE execution_id = ? OR attempt_id = ?",
            (request.execution_id, request.attempt_id),
        ).fetchone()
        if existing is not None:
            if existing[1] is None or _content_hash(existing[0]) != existing[1]:
                raise PersistenceConflictError("persisted adopted attempt integrity failed")
            return FullScientificAttemptClaim.model_validate_json(existing[0])
        next_sequence = int(
            connection.execute(
                "SELECT COALESCE(MAX(attempt_sequence), 0) + 1 "
                "FROM authority_full_attempt_claims WHERE run_id = ?",
                (request.run_id,),
            ).fetchone()[0]
        )
        if next_sequence > MAX_FULL_ATTEMPTS:
            raise PersistenceConflictError("legacy lifecycle adoption exceeded the attempt cap")
        record = FullScientificAttemptClaim(
            **request.model_dump(),
            attempt_sequence=next_sequence,
            claimed_at=datetime.now(UTC),
        )
        payload = record.model_dump_json()
        now = datetime.now(UTC).isoformat()
        connection.execute(
            "INSERT INTO authority_full_attempt_claims "
            "(attempt_id, execution_id, run_id, experiment_id, source_registration_id, "
            "source_commit, attempt_sequence, max_attempts, attempt_policy_id, claim_json, "
            "identity_sha256, content_sha256, claimed_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.attempt_id,
                record.execution_id,
                record.run_id,
                record.experiment_id,
                record.source_registration_id,
                record.source_commit,
                record.attempt_sequence,
                record.max_attempts,
                record.attempt_policy_id,
                payload,
                self._attempt_identity_digest(record),
                _content_hash(payload),
                record.claimed_at.isoformat(),
                now,
            ),
        )
        self._insert_audit(
            connection,
            self._automatic_event(
                "full_attempt_claimed",
                record.run_id,
                record.experiment_id,
                {
                    "attempt_id": record.attempt_id,
                    "execution_id": record.execution_id,
                    "attempt_sequence": record.attempt_sequence,
                },
                now,
            ),
        )
        return record

    def _insert_adopted_observation(
        self, connection: sqlite3.Connection, request: ScoredObservationRequest
    ) -> ScoredObservation:
        observation = ScoredObservation(
            **request.model_dump(),
            scored_at=datetime.now(UTC),
        )
        self._validate_observation(connection, observation)
        payload = observation.model_dump_json()
        identity_digest = self._observation_identity_digest(observation)
        duplicate = connection.execute(
            "SELECT observation_id FROM authority_scored_observations "
            "WHERE attempt_id = ? OR evaluation_id = ?",
            (observation.attempt_id, observation.evaluation_id),
        ).fetchone()
        if duplicate is not None:
            raise PersistenceConflictError(
                f"attempt or evaluation already has observation {duplicate[0]}"
            )
        now = datetime.now(UTC).isoformat()
        connection.execute(
            "INSERT INTO authority_scored_observations "
            "(observation_id, run_id, experiment_id, attempt_id, evaluation_id, checkpoint_id, "
            "observation_json, identity_sha256, content_sha256, scored_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                observation.observation_id,
                observation.run_id,
                observation.experiment_id,
                observation.attempt_id,
                observation.evaluation_id,
                observation.checkpoint_id,
                payload,
                identity_digest,
                _content_hash(payload),
                observation.scored_at.isoformat(),
                now,
            ),
        )
        self._insert_audit(
            connection,
            self._automatic_event(
                "scored_observation_persisted",
                observation.run_id,
                observation.experiment_id,
                {"observation_id": observation.observation_id},
                now,
            ),
        )
        return observation

    def adopt_legacy_lifecycle(self, run_id: str | None = None) -> tuple[dict[str, object], ...]:
        """Adopt pre-010 full dispatches from the resource authority.

        Resource reservations are the dispatch ledger for the pre-010 runtime;
        graph transitions are deliberately not consulted because failed
        execution transitions did not carry a latest execution identity.
        The complete plan is validated before the first claim is inserted so an
        ambiguous active history cannot be partially adopted.
        """
        with self._connect() as connection:
            active_runs = {
                str(row[0])
                for row in connection.execute(
                    "SELECT latest.run_id FROM run_states AS latest "
                    "JOIN (SELECT run_id, MAX(sequence) AS sequence FROM run_states "
                    "GROUP BY run_id) AS current ON current.run_id = latest.run_id "
                    "AND current.sequence = latest.sequence WHERE latest.status = 'active'"
                ).fetchall()
            }
            if run_id is not None:
                active_runs.intersection_update({run_id})
            migration_row = connection.execute(
                "SELECT applied_at FROM schema_migrations WHERE version = 10"
            ).fetchone()
            if migration_row is None:
                raise PersistenceConflictError("migration 010 cutoff is unavailable")
            migration_cutoff = _authority_timestamp(migration_row[0], "migration cutoff")
            reservation_rows = connection.execute(
                "SELECT reservation.reservation_id, reservation.reservation_json, "
                "reservation.status, reservation.created_at, operation.operation_id, "
                "operation.reservation_id, operation.operation, operation.usage_json, "
                "operation.created_at "
                "FROM authority_resource_reservations AS reservation "
                "JOIN authority_resource_operations AS operation "
                "ON operation.reservation_id = reservation.reservation_id "
                "WHERE operation.operation = 'reserve' "
                "ORDER BY operation.created_at, operation.operation_id, reservation.reservation_id"
            ).fetchall()
            execution_rows = connection.execute(
                "SELECT record_id, payload_json, content_sha256 FROM records "
                "WHERE kind = 'execution'"
            ).fetchall()
            failure_rows = connection.execute(
                "SELECT record_id, payload_json, content_sha256 FROM records "
                "WHERE kind = 'failure'"
            ).fetchall()
            source_rows = connection.execute(
                "SELECT registration_id, registration_json, content_sha256, created_at, "
                "experiment_id, run_id, source_commit, eligible "
                "FROM source_registrations"
            ).fetchall()
            claim_rows = connection.execute(
                "SELECT claim_json, content_sha256, attempt_id, execution_id, run_id, "
                "experiment_id, source_registration_id, source_commit "
                "FROM authority_full_attempt_claims"
            ).fetchall()

        executions: dict[str, ExecutionResult] = {}
        corrupt_execution_ids: set[str] = set()
        for record_id, payload, digest in execution_rows:
            if digest is None or _content_hash(payload) != digest:
                corrupt_execution_ids.add(str(record_id))
                continue
            try:
                execution = ExecutionResult.model_validate_json(payload)
            except ValueError:
                corrupt_execution_ids.add(str(record_id))
                continue
            if execution.execution_id == record_id:
                executions[execution.execution_id] = execution
            else:
                corrupt_execution_ids.add(str(record_id))

        failures: dict[str, tuple[FailureRecord, ...]] = {}
        for _, payload, digest in failure_rows:
            if digest is None or _content_hash(payload) != digest:
                continue
            try:
                failure = FailureRecord.model_validate_json(payload)
            except ValueError:
                continue
            for evidence_ref in failure.evidence_refs:
                failures.setdefault(evidence_ref, ())
                failures[evidence_ref] += (failure,)

        sources: dict[str, list[tuple[SourceRegistration, datetime]]] = {}
        corrupt_source_identities: set[tuple[str, str]] = set()
        for (
            registration_id,
            payload,
            digest,
            created_at,
            indexed_experiment_id,
            indexed_run_id,
            indexed_source_commit,
            indexed_eligible,
        ) in source_rows:
            if digest is None or _content_hash(payload) != digest:
                try:
                    raw_source = json.loads(payload)
                    if isinstance(raw_source, dict):
                        typed_source = cast(dict[str, object], raw_source)
                        raw_run: object = typed_source.get("run_id")
                        raw_experiment: object = typed_source.get("experiment_id")
                        if isinstance(raw_run, str) and isinstance(raw_experiment, str):
                            corrupt_source_identities.add((raw_run, raw_experiment))
                except (TypeError, ValueError, json.JSONDecodeError):
                    if isinstance(indexed_run_id, str) and isinstance(indexed_experiment_id, str):
                        corrupt_source_identities.add(
                            (indexed_run_id, indexed_experiment_id)
                        )
                if isinstance(indexed_run_id, str) and isinstance(indexed_experiment_id, str):
                    corrupt_source_identities.add((indexed_run_id, indexed_experiment_id))
                continue
            try:
                source = SourceRegistration.model_validate_json(payload)
            except ValueError:
                try:
                    raw_source = json.loads(payload)
                    if isinstance(raw_source, dict):
                        typed_source = cast(dict[str, object], raw_source)
                        raw_run = typed_source.get("run_id")
                        raw_experiment = typed_source.get("experiment_id")
                        if isinstance(raw_run, str) and isinstance(raw_experiment, str):
                            corrupt_source_identities.add((raw_run, raw_experiment))
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                continue
            if (
                source.registration_id != registration_id
                or source.experiment_id != indexed_experiment_id
                or source.run_id != indexed_run_id
                or source.source_commit != indexed_source_commit
                or (1 if source.eligible else 0) != indexed_eligible
            ):
                corrupt_source_identities.add((source.run_id, source.experiment_id))
                continue
            sources.setdefault(source.run_id, []).append(
                (source, _authority_timestamp(created_at, "source registration"))
            )

        claimed_execution_ids: set[str] = set()
        for row in claim_rows:
            claim_payload, claim_digest = row[0], row[1]
            claim = FullScientificAttemptClaim.model_validate_json(claim_payload)
            if claim.run_id not in active_runs:
                continue
            if claim_digest is None or _content_hash(claim_payload) != claim_digest:
                raise PersistenceConflictError(
                    "persisted full attempt claim integrity check failed"
                )
            if tuple(row[2:]) != (
                claim.attempt_id,
                claim.execution_id,
                claim.run_id,
                claim.experiment_id,
                claim.source_registration_id,
                claim.source_commit,
            ):
                raise PersistenceConflictError(
                    "persisted full attempt claim identity is inconsistent"
                )
            claimed_execution_ids.add(claim.execution_id)
        candidates: dict[
            str, list[tuple[datetime, str, str, ExecutionResult | None, SourceRegistration]]
        ] = {}
        ambiguous: dict[str, list[str]] = {}

        reservation_evidence: dict[str, list[tuple[object, ...]]] = {}
        for row in reservation_rows:
            reservation_evidence.setdefault(str(row[0]), []).append(tuple(row))

        for indexed_reservation_id, evidence_rows in reservation_evidence.items():
            pre010_rows = [
                row
                for row in evidence_rows
                if _authority_timestamp(row[8], "reserve operation") < migration_cutoff
            ]
            if not pre010_rows:
                continue
            if len(evidence_rows) != 1:
                raise PersistenceConflictError(
                    f"ambiguous legacy reservation evidence for {indexed_reservation_id}: "
                    "expected exactly one reserve operation"
                )
            (
                reservation_id,
                reservation_payload,
                status,
                reservation_created_at,
                operation_id,
                operation_reservation_id,
                operation,
                usage_json,
                created_at,
            ) = evidence_rows[0]
            operation_timestamp = _authority_timestamp(created_at, "reserve operation")
            indexed_timestamp = _authority_timestamp(
                reservation_created_at, "reservation"
            )
            if (
                reservation_id != indexed_reservation_id
                or operation_reservation_id != indexed_reservation_id
                or operation != "reserve"
                or not isinstance(operation_id, str)
                or not operation_id
                or usage_json is not None
                or indexed_timestamp != operation_timestamp
            ):
                raise PersistenceConflictError(
                    f"ambiguous legacy reservation evidence for {indexed_reservation_id}: "
                    "reservation identity or reserve operation content conflicts"
                )
            if not isinstance(reservation_payload, str):
                raise PersistenceConflictError(
                    f"ambiguous legacy reservation evidence for {indexed_reservation_id}: "
                    "reservation payload is malformed"
                )
            try:
                reservation = ResourceReservation.model_validate_json(reservation_payload)
            except ValueError as error:
                # This method is called only for an explicit resume.  A
                # malformed reservation cannot prove that it was a pre-dispatch
                # cleanup, so treating it as absent would permit an unsafe
                # partial adoption.
                raise PersistenceConflictError(
                    f"ambiguous legacy lifecycle evidence for active run {run_id}: "
                    f"{reservation_id}/{operation_id}"
                ) from error
            if reservation.reservation_id != indexed_reservation_id:
                raise PersistenceConflictError(
                    f"ambiguous legacy reservation evidence for active run {reservation.run_id}: "
                    f"{indexed_reservation_id}/{operation_id} reservation identity mismatch"
                )
            if reservation.run_id not in active_runs or reservation.purpose != "iteration":
                continue
            execution_id = str(reservation_id).removeprefix("reservation-")
            marker = f"{reservation_id}/{operation_id}"
            if not execution_id or execution_id == str(reservation_id):
                ambiguous.setdefault(reservation.run_id, []).append(marker)
                continue
            if execution_id.startswith("smoke-"):
                continue
            if execution_id in claimed_execution_ids:
                continue
            eligible_sources = [
                source
                for source, source_created_at in sources.get(reservation.run_id, [])
                if source.experiment_id == reservation.experiment_id
                and source.eligible
                # The reserve operation is the dispatch-order authority.  The
                # reservation row timestamp may be copied or rewritten during
                # legacy recovery and must not establish source ordering.
                and source_created_at <= operation_timestamp
            ]
            if (reservation.run_id, str(reservation.experiment_id)) in corrupt_source_identities:
                ambiguous.setdefault(reservation.run_id, []).append(marker)
                continue
            execution = executions.get(execution_id)
            if execution_id in corrupt_execution_ids:
                ambiguous.setdefault(reservation.run_id, []).append(marker)
                continue
            if execution is not None:
                source = next(
                    (
                        item
                        for item in eligible_sources
                        if item.registration_id == execution.source_registration_id
                        and item.source_commit == execution.source_commit
                    ),
                    None,
                )
                if execution.execution_kind != "full":
                    continue
                if execution.experiment_id != reservation.experiment_id or source is None:
                    ambiguous.setdefault(reservation.run_id, []).append(marker)
                    continue
            else:
                # A consumed or still-reserved iteration reservation can mean
                # that dispatch started before its result was durable.  The
                # source must still be uniquely recoverable; otherwise resume
                # is blocked rather than guessing.
                if status not in {"consumed", "reserved"} or len(eligible_sources) != 1:
                    ambiguous.setdefault(reservation.run_id, []).append(marker)
                    continue
                source = eligible_sources[0]
                if any(
                    failure.experiment_id not in {None, reservation.experiment_id}
                    for failure in failures.get(execution_id, ())
                ):
                    ambiguous.setdefault(reservation.run_id, []).append(marker)
                    continue
            candidates.setdefault(reservation.run_id, []).append(
                (operation_timestamp, str(operation_id), execution_id, execution, source)
            )

        planned: dict[
            str,
            list[tuple[FullAttemptClaimRequest, ScoredObservationRequest | None]],
        ] = {}
        existing_attempt_counts: dict[str, int] = {}
        existing_observation_counts: dict[str, int] = {}
        for run_id in sorted(set(active_runs) | set(candidates) | set(ambiguous)):
            if ambiguous.get(run_id):
                raise PersistenceConflictError(
                    f"ambiguous legacy lifecycle evidence for active run {run_id}: "
                    + ", ".join(ambiguous[run_id][:8])
                )
            entries = candidates.get(run_id, [])
            entries.sort(key=lambda item: (item[0], item[1], item[2]))
            if len(entries) + self.count_full_attempt_claims(run_id) > MAX_FULL_ATTEMPTS:
                raise PersistenceConflictError(
                    f"legacy lifecycle evidence exceeds the fixed attempt cap for run {run_id}"
                )
            existing_attempt_counts[run_id] = self.count_full_attempt_claims(run_id)
            existing_observation_counts[run_id] = len(self.list_scored_observations(run_id))
            planned[run_id] = []
            for index, (_, _, execution_id, execution, source) in enumerate(entries):
                request = FullAttemptClaimRequest(
                    attempt_id=f"attempt-{execution_id}",
                    execution_id=execution_id,
                    run_id=run_id,
                    experiment_id=source.experiment_id,
                    source_registration_id=source.registration_id,
                    source_commit=source.source_commit,
                )
                observation_request: ScoredObservationRequest | None = None
                if execution is not None and execution.failure_kind is None:
                    provisional_claim = FullScientificAttemptClaim(
                        **request.model_dump(),
                        attempt_sequence=existing_attempt_counts[run_id] + index + 1,
                        claimed_at=datetime.now(UTC),
                    )
                    try:
                        observation_request = self._legacy_scored_observation(
                            execution, provisional_claim
                        )
                    except (PersistenceConflictError, ValueError) as error:
                        raise PersistenceConflictError(
                            f"corrupt legacy observation evidence for {execution_id}"
                        ) from error
                planned[run_id].append((request, observation_request))

        summaries: list[dict[str, object]] = []
        # Validate and insert the entire adoption plan in one transaction.  A
        # later run must not be able to retain claims if an earlier run's
        # authority is found to be corrupt during insertion.
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for run_id in sorted(planned):
                    adopted_attempts = 0
                    adopted_observations = existing_observation_counts[run_id]
                    for request, observation_request in planned[run_id]:
                        self._insert_adopted_claim(connection, request)
                        adopted_attempts += 1
                        if observation_request is not None:
                            self._insert_adopted_observation(connection, observation_request)
                            adopted_observations += 1
                    summary: dict[str, object] = {
                        "run_id": run_id,
                        "adopted_attempts": adopted_attempts,
                        "adopted_observations": adopted_observations,
                        "ambiguous_execution_ids": tuple(ambiguous.get(run_id, ())[:8]),
                    }
                    self._insert_audit(
                        connection,
                        AuditEvent(
                            event_id=f"lifecycle-adoption-{run_id}",
                            run_id=run_id,
                            event_type="legacy_lifecycle_adopted",
                            actor_type="controller",
                            actor_id="production-controller",
                            payload={"run_id": run_id, "adopted": True},
                            created_at=datetime(1970, 1, 1, tzinfo=UTC),
                        ),
                    )
                    summaries.append(summary)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        for summary in summaries:
            closure = self.close_run_if_ready(str(summary["run_id"]))
            if closure is not None:
                summary["closure_reason"] = closure.reason
        return tuple(summaries)

    def _legacy_scored_observation(
        self, execution: ExecutionResult, claim: FullScientificAttemptClaim
    ) -> ScoredObservationRequest | None:
        if execution.checkpoint_id is None:
            return None
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record_id, payload_json, content_sha256 "
                "FROM records WHERE kind = 'evaluation'"
            ).fetchall()
            evaluation_payload: dict[str, object] | None = None
            for record_id, payload, digest in rows:
                try:
                    raw = json.loads(payload)
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    if record_id == f"evaluation-{execution.execution_id}":
                        raise PersistenceConflictError(
                            "legacy evaluation authority is not valid JSON"
                        ) from error
                    continue
                if not isinstance(raw, dict):
                    if record_id == f"evaluation-{execution.execution_id}":
                        raise PersistenceConflictError(
                            "legacy evaluation authority is not an object"
                        )
                    continue
                raw_mapping = cast(dict[str, object], raw)
                raw_result: object = raw_mapping.get("result")
                candidate: dict[str, object] = (
                    cast(dict[str, object], raw_result)
                    if isinstance(raw_result, dict)
                    else raw_mapping
                )
                if candidate.get("execution_id") != execution.execution_id:
                    continue
                if digest is None or _content_hash(payload) != digest:
                    raise PersistenceConflictError("legacy evaluation authority integrity failed")
                evaluation_payload = raw_mapping
                break
            if not isinstance(evaluation_payload, dict):
                return None
            evaluation = EvaluationResult.model_validate(
                evaluation_payload.get("result", evaluation_payload)
            )
            report_rows = connection.execute(
                "SELECT report.report_json, report.content_sha256, "
                "operation.operation_json, operation.subject_json, operation.content_sha256 "
                "FROM authority_validation_reports AS report "
                "JOIN authority_validation_operations AS operation "
                "ON operation.operation_id = report.validation_operation_id "
                "WHERE report.experiment_id = ? AND report.stage = 'result' "
                "ORDER BY report.created_at DESC, report.report_id DESC",
                (execution.experiment_id,),
            ).fetchall()
        if not (
            evaluation.execution_id == execution.execution_id
            and evaluation.run_id == claim.run_id
            and evaluation.experiment_id == execution.experiment_id
            and evaluation.source_commit == execution.source_commit
            and evaluation.checkpoint_id == execution.checkpoint_id
            and evaluation.split == "valid"
            and evaluation.validity in {"provisional", "official"}
        ):
            raise PersistenceConflictError("legacy evaluation provenance is inconsistent")
        if not report_rows:
            return None
        reports: list[tuple[ValidationReport, dict[str, object]]] = []
        for (
            report_json,
            report_digest,
            operation_json,
            subject_json,
            operation_digest,
        ) in report_rows:
            if (
                report_digest is None
                or _content_hash(report_json) != report_digest
                or operation_digest is None
                or _content_hash(operation_json + subject_json) != operation_digest
            ):
                raise PersistenceConflictError("legacy validation authority integrity failed")
            report = ValidationReport.model_validate_json(report_json)
            operation = ValidationOperationIdentity.model_validate_json(operation_json)
            subject = json.loads(subject_json)
            if not isinstance(subject, dict):
                raise PersistenceConflictError("legacy validation subject is not an object")
            if (
                operation.run_id != claim.run_id
                or operation.experiment_id != execution.experiment_id
                or operation.stage != ValidationStage.RESULT
                or report.experiment_id != execution.experiment_id
                or report.stage != ValidationStage.RESULT
            ):
                raise PersistenceConflictError(
                    "legacy validation operation provenance is inconsistent"
                )
            reports.append((report, cast(dict[str, object], subject)))
        exact_subject = {
            "evaluation_result": evaluation.model_dump(mode="json"),
            "execution_result": execution.model_dump(mode="json", exclude={"dataset_valid_rows"}),
        }
        approved = [report for report, _ in reports if report.verdict == ValidationVerdict.APPROVED]
        if not approved:
            return None
        report = next(
            (
                report
                for report, subject in reports
                if report.verdict == ValidationVerdict.APPROVED
                and report.stage == ValidationStage.RESULT
                and subject == exact_subject
            ),
            None,
        )
        if report is None:
            raise PersistenceConflictError("legacy validation subject provenance is inconsistent")
        if (
            evaluation.dataset_manifest_id is None
            or evaluation.dataset_manifest_sha256 is None
            or evaluation.source_commit is None
        ):
            return None
        validity = cast(Literal["provisional", "official"], evaluation.validity)
        return ScoredObservationRequest(
            observation_id=f"observation-{evaluation.evaluation_id}",
            run_id=claim.run_id,
            experiment_id=evaluation.experiment_id,
            attempt_id=claim.attempt_id,
            execution_id=execution.execution_id,
            evaluation_id=evaluation.evaluation_id,
            checkpoint_id=evaluation.checkpoint_id,
            source_commit=evaluation.source_commit,
            evaluator_id=evaluation.evaluator_artifact_id,
            evaluator_sha256=evaluation.evaluator_sha256,
            dataset_manifest_id=evaluation.dataset_manifest_id,
            dataset_manifest_sha256=evaluation.dataset_manifest_sha256,
            split="valid",
            validity=validity,
            primary_score=evaluation.validation_score,
            validation_report_id=report.report_id,
            validation_evidence_refs=report.evidence_refs,
        )

    def get_run_closure(self, run_id: str) -> RunClosure | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT closure_json, content_sha256, closure_id, run_id, reason "
                "FROM authority_run_closures WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        if _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError("persisted run closure integrity check failed")
        closure = RunClosure.model_validate_json(row[0])
        if tuple(row[2:]) != (closure.closure_id, closure.run_id, closure.reason):
            raise PersistenceConflictError("persisted run closure identity is inconsistent")
        return closure

    def put_experiment(
        self,
        spec: ExperimentSpec,
        status: str,
        run_id: str,
        transition_id: str,
        expected_predecessor: str | None,
        audit_event: AuditEvent | None = None,
    ) -> None:
        payload = spec.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone():
                raise ValueError("authoritative experiment transitions require a persisted run")
            row = connection.execute(
                "SELECT content_sha256 FROM authority_experiments WHERE experiment_id = ?",
                (spec.experiment_id,),
            ).fetchone()
            if row is not None and row[0] != digest:
                raise PersistenceConflictError(f"experiment {spec.experiment_id} content changed")
            if row is None:
                connection.execute(
                    "INSERT INTO authority_experiments "
                    "(experiment_id, spec_json, content_sha256, created_at) VALUES (?, ?, ?, ?)",
                    (spec.experiment_id, payload, digest, now),
                )
                if not connection.execute(
                    "SELECT 1 FROM experiments WHERE experiment_id = ?", (spec.experiment_id,)
                ).fetchone():
                    connection.execute(
                        "INSERT INTO experiments "
                        "(experiment_id, hypothesis_id, parent_experiment_id, status, "
                        "source_commit, "
                        "spec_json, created_at, updated_at) VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
                        (
                            spec.experiment_id,
                            spec.hypothesis_id,
                            spec.parent_experiment_id,
                            status,
                            payload,
                            now,
                            now,
                        ),
                    )
            state = connection.execute(
                "SELECT status, transition_id FROM experiment_states WHERE experiment_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (spec.experiment_id,),
            ).fetchone()
            transition_hash = self._transition(
                "experiment",
                spec.experiment_id,
                status,
                transition_id,
                expected_predecessor,
            )
            prior = connection.execute(
                "SELECT content_sha256 FROM experiment_states WHERE transition_id = ?",
                (transition_id,),
            ).fetchone()
            if prior is not None:
                if prior[0] != transition_hash:
                    raise PersistenceConflictError(
                        f"experiment transition {transition_id} content changed"
                    )
                return
            if (state[1] if state else None) != expected_predecessor:
                raise PersistenceConflictError(
                    f"experiment transition {transition_id} has a stale predecessor"
                )
            connection.execute(
                "INSERT INTO experiment_states "
                "(experiment_id, status, transition_id, content_sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (spec.experiment_id, status, transition_id, transition_hash, now),
            )
            event = audit_event or self._automatic_event(
                "experiment_state_changed",
                run_id,
                spec.experiment_id,
                {"status": status, "transition_id": transition_id},
                now,
            )
            self._insert_audit(connection, event)

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT spec_json FROM authority_experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT spec_json FROM experiments WHERE experiment_id = ?", (experiment_id,)
                ).fetchone()
        return ExperimentSpec.model_validate_json(row[0]) if row else None

    def list_experiments(
        self, limit: int = 50, exclude_experiment_id: str | None = None
    ) -> tuple[tuple[ExperimentRegistryEntry, ...], int]:
        if limit < 1:
            raise ValueError("experiment registry limit must be positive")
        with self._connect() as connection:
            where = ""
            parameters: tuple[object, ...] = (limit,)
            if exclude_experiment_id is not None:
                where = "WHERE authority.experiment_id != ? "
                parameters = (exclude_experiment_id, limit)
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM authority_experiments AS authority "
                        "WHERE authority.experiment_id != ?",
                        (exclude_experiment_id,),
                    ).fetchone()[0]
                )
            else:
                total = int(
                    connection.execute("SELECT COUNT(*) FROM authority_experiments").fetchone()[0]
                )
            query = (
                "SELECT authority.spec_json, COALESCE(("
                "SELECT state.status FROM experiment_states AS state "
                "WHERE state.experiment_id = authority.experiment_id "
                "ORDER BY state.sequence DESC LIMIT 1"
                "), 'proposed') "
                "FROM authority_experiments AS authority "
                + where
                + "ORDER BY authority.created_at DESC LIMIT ?"
            )
            rows = connection.execute(query, parameters).fetchall()
        entries: list[ExperimentRegistryEntry] = []
        for payload, status in rows:
            value = json.loads(payload)
            entries.append(
                ExperimentRegistryEntry(
                    experiment_id=str(value["experiment_id"]),
                    hypothesis_id=str(value["hypothesis_id"]),
                    parent_experiment_id=value.get("parent_experiment_id"),
                    hypothesis=str(value["hypothesis"]),
                    mechanism=str(value["mechanism"]),
                    status=str(status),
                )
            )
        return tuple(entries), total

    def put_source_registration(
        self, registration: SourceRegistration, audit_event: AuditEvent | None = None
    ) -> None:
        if not registration.eligible or not registration.allowed_scopes:
            raise ValueError("source registration is incomplete or ineligible")
        if registration.patch_artifact_id != f"patch-{registration.patch_sha256}":
            raise ValueError("patch artifact identity does not match its hash")
        payload = registration.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute(
                "SELECT 1 FROM authority_experiments WHERE experiment_id = ?",
                (registration.experiment_id,),
            ).fetchone():
                raise ValueError("source registration requires a persisted experiment")
            try:
                self._validate_patch_artifact(connection, registration, registration.experiment_id)
            except ValueError as error:
                raise ValueError(str(error)) from error
            row = connection.execute(
                "SELECT content_sha256 FROM source_registrations WHERE registration_id = ?",
                (registration.registration_id,),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise PersistenceConflictError(
                        f"source registration {registration.experiment_id} content changed"
                    )
                if audit_event is not None:
                    self._insert_audit(connection, audit_event)
                return
            revision_row = connection.execute(
                "SELECT registration_id FROM source_registrations "
                "WHERE experiment_id = ? AND revision = ?",
                (registration.experiment_id, registration.revision),
            ).fetchone()
            if revision_row is not None:
                raise PersistenceConflictError(
                    f"source revision {registration.experiment_id}:{registration.revision} changed"
                )
            connection.execute(
                "INSERT INTO source_registrations "
                "(registration_id, experiment_id, revision, registration_json, "
                "content_sha256, created_at, run_id, source_commit, eligible) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    registration.registration_id,
                    registration.experiment_id,
                    registration.revision,
                    payload,
                    digest,
                    now,
                    registration.run_id,
                    registration.source_commit,
                    1 if registration.eligible else 0,
                ),
            )
            event = audit_event or self._automatic_event(
                "source_registered",
                registration.run_id,
                registration.experiment_id,
                {
                    "registration_id": registration.registration_id,
                    "revision": registration.revision,
                    "source_commit": registration.source_commit,
                },
                now,
            )
            self._insert_audit(connection, event)

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT registration_json, content_sha256, registration_id, experiment_id, run_id, "
                "source_commit, eligible FROM source_registrations "
                "WHERE experiment_id = ? ORDER BY revision DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        if row[1] is None or _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError("source registration integrity check failed")
        source = SourceRegistration.model_validate_json(row[0])
        if tuple(row[2:]) != (
            source.registration_id,
            source.experiment_id,
            source.run_id,
            source.source_commit,
            1 if source.eligible else 0,
        ):
            raise PersistenceConflictError("source registration identity is inconsistent")
        return source

    def get_source_registration_by_id(self, registration_id: str) -> SourceRegistration | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT registration_json, content_sha256, registration_id, experiment_id, run_id, "
                "source_commit, eligible FROM source_registrations WHERE registration_id = ?",
                (registration_id,),
            ).fetchone()
        if row is None:
            return None
        if row[1] is None or _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError("source registration integrity check failed")
        source = SourceRegistration.model_validate_json(row[0])
        if tuple(row[2:]) != (
            source.registration_id,
            source.experiment_id,
            source.run_id,
            source.source_commit,
            1 if source.eligible else 0,
        ):
            raise PersistenceConflictError("source registration identity is inconsistent")
        return source

    def put_evaluator_identity(
        self, identity: EvaluatorIdentity, audit_event: AuditEvent | None = None
    ) -> None:
        payload = identity.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT content_sha256 FROM evaluator_identities WHERE evaluator_id = ?",
                (identity.evaluator_id,),
            ).fetchone()
            if row is not None and row[0] != digest:
                raise PersistenceConflictError(f"evaluator {identity.evaluator_id} content changed")
            if row is None:
                connection.execute(
                    "INSERT INTO evaluator_identities "
                    "(evaluator_id, evaluator_json, content_sha256, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (identity.evaluator_id, payload, digest, now),
                )
                self._insert_audit(
                    connection,
                    audit_event
                    or self._automatic_event(
                        "evaluator_registered",
                        "system",
                        None,
                        {"evaluator_id": identity.evaluator_id, "validity": identity.validity},
                        now,
                    ),
                )
            elif audit_event is not None:
                self._insert_audit(connection, audit_event)

    def put_run(
        self,
        run: RunRecord,
        transition_id: str,
        expected_predecessor: str | None,
        audit_event: AuditEvent | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run.run_id,)
            ).fetchone():
                connection.execute(
                    "INSERT INTO runs (run_id, status, final_test_claimed, created_at) "
                    "VALUES (?, ?, 0, ?)",
                    (run.run_id, run.status, now),
                )
            state = connection.execute(
                "SELECT status, transition_id FROM run_states WHERE run_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (run.run_id,),
            ).fetchone()
            transition_hash = self._transition(
                "run", run.run_id, run.status, transition_id, expected_predecessor
            )
            prior = connection.execute(
                "SELECT content_sha256 FROM run_states WHERE transition_id = ?",
                (transition_id,),
            ).fetchone()
            if prior is not None:
                if prior[0] != transition_hash:
                    raise PersistenceConflictError(
                        f"run transition {transition_id} content changed"
                    )
                return
            if (state[1] if state else None) != expected_predecessor:
                raise PersistenceConflictError(
                    f"run transition {transition_id} has a stale predecessor"
                )
            connection.execute(
                "INSERT INTO run_states "
                "(run_id, status, transition_id, content_sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run.run_id, run.status, transition_id, transition_hash, now),
            )
            self._insert_audit(
                connection,
                audit_event
                or self._automatic_event(
                    "run_state_changed",
                    run.run_id,
                    None,
                    {"status": run.status, "transition_id": transition_id},
                    now,
                ),
            )

    def _insert_finalization(
        self,
        connection: sqlite3.Connection,
        finalization: FinalizationRecord,
        audit_event: AuditEvent | None,
        now: str,
    ) -> None:
        payload = finalization.model_dump_json()
        digest = _content_hash(payload)
        row = connection.execute(
            "SELECT content_sha256 FROM authority_finalizations WHERE finalization_id = ?",
            (finalization.finalization_id,),
        ).fetchone()
        if row is not None:
            if row[0] != digest:
                raise PersistenceConflictError(
                    f"finalization {finalization.finalization_id} content changed"
                )
            if audit_event is not None:
                self._insert_audit(connection, audit_event)
            return
        connection.execute(
            "INSERT INTO authority_finalizations "
            "(finalization_id, finalization_json, content_sha256, created_at) "
            "VALUES (?, ?, ?, ?)",
            (finalization.finalization_id, payload, digest, now),
        )
        self._insert_audit(
            connection,
            audit_event
            or self._automatic_event(
                "finalization_persisted",
                finalization.run_id,
                finalization.experiment_id,
                {"validity": finalization.validity},
                now,
            ),
        )

    def persist_provisional_finalization(
        self, request: ProvisionalFinalizationRequest, actor_id: str = "controller"
    ) -> FinalizationRecord:
        if not all(
            (
                request.checkpoint_id,
                request.evaluation_id,
                request.bundle_artifact_id,
            )
        ):
            raise FinalTestAccessError("provisional finalization provenance is incomplete")
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run_status = connection.execute(
                "SELECT status FROM run_states WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (request.run_id,),
            ).fetchone()
            experiment_status = connection.execute(
                "SELECT status FROM experiment_states WHERE experiment_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (request.experiment_id,),
            ).fetchone()
            if run_status is None or run_status[0] != "converged":
                raise FinalTestAccessError("provisional finalization requires a converged run")
            if experiment_status is None or experiment_status[0] != "converged":
                raise FinalTestAccessError(
                    "provisional finalization requires a converged experiment"
                )
            closure_row = connection.execute(
                "SELECT closure_json, content_sha256 FROM authority_run_closures "
                "WHERE run_id = ?",
                (request.run_id,),
            ).fetchone()
            if closure_row is None or _content_hash(closure_row[0]) != closure_row[1]:
                raise FinalTestAccessError("provisional finalization requires a valid run closure")
            closure = RunClosure.model_validate_json(closure_row[0])
            if closure.champion is None:
                raise FinalTestAccessError("run closure has no eligible champion")
            champion = closure.champion
            observation_row = connection.execute(
                "SELECT observation_json, content_sha256 FROM authority_scored_observations "
                "WHERE observation_id = ?",
                (champion.observation_id,),
            ).fetchone()
            if observation_row is None or _content_hash(observation_row[0]) != observation_row[1]:
                raise FinalTestAccessError("run closure champion observation is unavailable")
            observation = ScoredObservation.model_validate_json(observation_row[0])
            if (
                observation.run_id != request.run_id
                or observation.experiment_id != request.experiment_id
                or observation.attempt_id != champion.attempt_id
                or observation.execution_id != champion.execution_id
                or observation.evaluation_id != champion.evaluation_id
                or observation.checkpoint_id != champion.checkpoint_id
                or observation.source_commit != champion.source_commit
                or observation.primary_score != champion.primary_score
                or request.experiment_id != observation.experiment_id
                or request.source_commit != observation.source_commit
                or request.evaluation_id != observation.evaluation_id
                or request.checkpoint_id != observation.checkpoint_id
            ):
                raise FinalTestAccessError("finalization is not bound to the closure champion")
            source_row = connection.execute(
                "SELECT registration_json, content_sha256 FROM source_registrations "
                "WHERE registration_id = ?",
                (f"source-{observation.source_commit}",),
            ).fetchone()
            if source_row is None or _content_hash(source_row[0]) != source_row[1]:
                raise FinalTestAccessError("champion source revision is unavailable")
            source = SourceRegistration.model_validate_json(source_row[0])
            if (
                source.registration_id != f"source-{observation.source_commit}"
                or source.run_id != request.run_id
                or source.experiment_id != request.experiment_id
                or not source.eligible
                or source.source_commit != request.source_commit
            ):
                raise FinalTestAccessError("champion source revision does not match finalization")
            try:
                self._validate_patch_artifact(connection, source, request.experiment_id)
            except ValueError as error:
                raise FinalTestAccessError(str(error)) from error
            spec_row = connection.execute(
                "SELECT spec_json, content_sha256 FROM authority_experiments "
                "WHERE experiment_id = ?",
                (request.experiment_id,),
            ).fetchone()
            if spec_row is None or _content_hash(spec_row[0]) != spec_row[1]:
                raise FinalTestAccessError("champion experiment specification is unavailable")
            spec = ExperimentSpec.model_validate_json(spec_row[0])
            if spec.experiment_id != observation.experiment_id:
                raise FinalTestAccessError("champion experiment specification is inconsistent")
            evaluator_row = connection.execute(
                "SELECT evaluator_json, content_sha256 FROM evaluator_identities "
                "WHERE evaluator_id = ?",
                (request.evaluator_id,),
            ).fetchone()
            if evaluator_row is None or _content_hash(evaluator_row[0]) != evaluator_row[1]:
                raise FinalTestAccessError("evaluator identity is not registered")
            evaluator = EvaluatorIdentity.model_validate_json(evaluator_row[0])
            try:
                evaluation_payload = json.loads(
                    self._verified_record(connection, "evaluation", request.evaluation_id)
                )
            except ValueError as error:
                raise FinalTestAccessError("evaluation provenance is unavailable") from error
            evaluation = EvaluationResult.model_validate(
                evaluation_payload.get("result", evaluation_payload)
            )
            if (
                evaluation.evaluation_id != observation.evaluation_id
                or evaluation.experiment_id != observation.experiment_id
                or evaluation.run_id != request.run_id
                or evaluation.source_commit != observation.source_commit
                or evaluation.execution_id != observation.execution_id
                or evaluation.checkpoint_id != request.checkpoint_id
                or evaluation.evaluator_artifact_id != request.evaluator_id
                or evaluation.evaluator_sha256 != evaluator.evaluator_sha256
                or evaluation.validity != observation.validity
                or evaluation.split != observation.split
            ):
                raise FinalTestAccessError("evaluation provenance does not match finalization")
            try:
                self._validate_artifact_record(
                    connection,
                    request.bundle_artifact_id,
                    request.run_id,
                    request.experiment_id,
                    "finalization_bundle",
                )
            except ValueError as error:
                raise FinalTestAccessError(str(error)) from error
            finalization = FinalizationRecord(
                finalization_id=request.finalization_id,
                run_id=request.run_id,
                experiment_id=request.experiment_id,
                source_commit=request.source_commit,
                checkpoint_id=request.checkpoint_id,
                evaluation_id=request.evaluation_id,
                validity="provisional",
                bundle_artifact_id=request.bundle_artifact_id,
                consumed_test_access=False,
            )
            self._insert_finalization(
                connection,
                finalization,
                AuditEvent(
                    event_id=f"provisional-finalization-{request.finalization_id}",
                    run_id=request.run_id,
                    experiment_id=request.experiment_id,
                    event_type="provisional_finalization_persisted",
                    actor_type="controller",
                    actor_id=actor_id,
                    payload={"finalization_id": request.finalization_id},
                    created_at=datetime.fromisoformat(now),
                ),
                now,
            )
            connection.commit()
            return finalization

    def get_finalization(self, finalization_id: str) -> FinalizationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT finalization_json FROM authority_finalizations WHERE finalization_id = ?",
                (finalization_id,),
            ).fetchone()
        return FinalizationRecord.model_validate_json(row[0]) if row else None

    def get_final_test_claim(self, claim_id: str) -> FinalTestClaim | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT claim_json, content_sha256 FROM final_test_claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
        if row is None:
            return None
        claim_json, content_sha256 = row
        if _content_hash(claim_json) != content_sha256:
            raise FinalTestAccessError("persisted final test claim integrity check failed")
        claim = FinalTestClaim.model_validate_json(claim_json)
        if claim.claim_id != claim_id:
            raise FinalTestAccessError("persisted final test claim identity mismatch")
        return claim

    def authorize_final_test(
        self, request: FinalTestAuthorizationRequest, actor_id: str = "controller"
    ) -> FinalTestClaim:
        now = datetime.now(UTC).isoformat()
        claim_payload = request.model_dump_json()
        claim_id = f"claim-{_content_hash(claim_payload)}"
        claim = FinalTestClaim(claim_id=claim_id, **request.model_dump())
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (request.run_id,)
            ).fetchone()
            status = connection.execute(
                "SELECT status FROM run_states WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (request.run_id,),
            ).fetchone()
            if run is None or status is None or status[0] != "converged":
                raise FinalTestAccessError("final test requires a converged run")
            experiment = connection.execute(
                "SELECT 1 FROM authority_experiments WHERE experiment_id = ?",
                (request.experiment_id,),
            ).fetchone()
            experiment_status = connection.execute(
                "SELECT status FROM experiment_states WHERE experiment_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (request.experiment_id,),
            ).fetchone()
            if (
                experiment is None
                or experiment_status is None
                or experiment_status[0] != "converged"
            ):
                raise FinalTestAccessError("final test requires a converged experiment")
            self._validate_source(
                connection, request.experiment_id, request.run_id, request.source_commit
            )
            evaluator_row = connection.execute(
                "SELECT evaluator_json FROM evaluator_identities WHERE evaluator_id = ?",
                (request.evaluator_id,),
            ).fetchone()
            if evaluator_row is None:
                raise FinalTestAccessError("evaluator identity is not registered")
            identity = EvaluatorIdentity.model_validate_json(evaluator_row[0])
            claim = claim.model_copy(update={"evaluator_sha256": identity.evaluator_sha256})
            existing = connection.execute(
                "SELECT claim_json, claim_id FROM final_test_claims WHERE run_id = ?",
                (request.run_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != claim.model_dump_json():
                    raise FinalTestAccessError("final test access has already been claimed")
                return claim
            connection.execute(
                "INSERT INTO final_test_claims "
                "(claim_id, run_id, claim_json, content_sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    claim_id,
                    request.run_id,
                    claim.model_dump_json(),
                    _content_hash(claim.model_dump_json()),
                    now,
                ),
            )
            self._insert_audit(
                connection,
                AuditEvent(
                    event_id=f"final-test-claim-{claim_id}",
                    run_id=request.run_id,
                    experiment_id=request.experiment_id,
                    event_type="final_test_authorized",
                    actor_type="controller",
                    actor_id=actor_id,
                    payload={"claim_id": claim_id, "evaluator_id": identity.evaluator_id},
                    created_at=datetime.fromisoformat(now),
                ),
            )
            connection.commit()
            return claim

    def complete_final_test(
        self, request: FinalTestRequest, actor_id: str = "controller"
    ) -> FinalizationRecord:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            claim_row = connection.execute(
                "SELECT claim_json FROM final_test_claims WHERE claim_id = ?", (request.claim_id,)
            ).fetchone()
            if claim_row is None:
                raise FinalTestAccessError("final test completion requires an authorization claim")
            claim = FinalTestClaim.model_validate_json(claim_row[0])
            if any(
                (
                    claim.run_id != request.run_id,
                    claim.experiment_id != request.experiment_id,
                    claim.source_commit != request.source_commit,
                    claim.evaluator_id != request.evaluator_id,
                )
            ):
                raise FinalTestAccessError(
                    "final test result does not match its authorization claim"
                )
            evaluator_row = connection.execute(
                "SELECT evaluator_json FROM evaluator_identities WHERE evaluator_id = ?",
                (claim.evaluator_id,),
            ).fetchone()
            if evaluator_row is None:
                raise FinalTestAccessError("evaluator identity is not registered")
            identity = EvaluatorIdentity.model_validate_json(evaluator_row[0])
            completion = connection.execute(
                "SELECT finalization_id FROM final_test_completions WHERE claim_id = ?",
                (claim.claim_id,),
            ).fetchone()
            if completion is not None and completion[0] != request.finalization_id:
                raise FinalTestAccessError("final test access has already been consumed")
            finalization = FinalizationRecord(
                finalization_id=request.finalization_id,
                run_id=claim.run_id,
                experiment_id=claim.experiment_id,
                source_commit=claim.source_commit,
                checkpoint_id=request.checkpoint_id,
                evaluation_id=request.evaluation_id,
                validity=identity.validity,
                bundle_artifact_id=request.bundle_artifact_id,
                consumed_test_access=True,
            )
            payload = finalization.model_dump_json()
            existing = connection.execute(
                "SELECT finalization_json FROM authority_finalizations WHERE finalization_id = ?",
                (finalization.finalization_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise PersistenceConflictError(
                        f"finalization {finalization.finalization_id} content changed"
                    )
                return finalization
            connection.execute(
                "INSERT INTO authority_finalizations "
                "(finalization_id, finalization_json, content_sha256, created_at) "
                "VALUES (?, ?, ?, ?)",
                (finalization.finalization_id, payload, _content_hash(payload), now),
            )
            connection.execute(
                "INSERT INTO final_test_completions "
                "(claim_id, finalization_id, content_sha256, created_at) VALUES (?, ?, ?, ?)",
                (claim.claim_id, finalization.finalization_id, _content_hash(payload), now),
            )
            self._insert_audit(
                connection,
                AuditEvent(
                    event_id=f"final-test-complete-{request.finalization_id}",
                    run_id=claim.run_id,
                    experiment_id=claim.experiment_id,
                    event_type="final_test_completed",
                    actor_type="controller",
                    actor_id=actor_id,
                    payload={
                        "claim_id": claim.claim_id,
                        "finalization_id": request.finalization_id,
                    },
                    created_at=datetime.fromisoformat(now),
                ),
            )
            connection.commit()
            return finalization

    def consume_final_test_access(
        self, request: FinalTestRequest, actor_id: str = "controller"
    ) -> FinalizationRecord:
        return self.complete_final_test(request, actor_id)

    def put_audit_event(self, event: AuditEvent) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_audit(connection, event)

    def list_audit_events(self, run_id: str) -> tuple[AuditEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM audit_events WHERE run_id = ? "
                "ORDER BY created_at, event_id",
                (run_id,),
            ).fetchall()
        return tuple(AuditEvent.model_validate_json(row[0]) for row in rows)

    def _ensure_audit_event(
        self,
        connection: sqlite3.Connection,
        event: AuditEvent,
        *,
        preserve_existing_actor: bool = False,
    ) -> None:
        """Insert an audit event or verify an identical event already exists."""
        row = connection.execute(
            "SELECT payload_json FROM audit_events WHERE event_id = ?",
            (event.event_id,),
        ).fetchone()
        if row is None:
            self._insert_audit(connection, event)
            return
        actual = json.loads(row[0])
        expected = event.model_dump(mode="json")
        # Persistence timestamps are not identity.  This permits an
        # interrupted write to repair its audit row on an identical replay.
        actual.pop("created_at", None)
        expected.pop("created_at", None)
        if preserve_existing_actor:
            actual.pop("actor_type", None)
            actual.pop("actor_id", None)
            expected.pop("actor_type", None)
            expected.pop("actor_id", None)
        if actual != expected:
            raise PersistenceConflictError(f"audit event {event.event_id} content changed")

    @staticmethod
    def _baseline_calibration_event(
        record: BaselineCalibrationRecord,
        actor_type: Literal["agent", "controller", "human"],
        actor_id: str,
        now: str,
    ) -> AuditEvent:
        return AuditEvent(
            event_id=f"baseline-calibrated-{record.calibration_id}",
            run_id=record.calibration_id,
            event_type="baseline_calibrated",
            actor_type=actor_type,
            actor_id=actor_id,
            payload={
                "calibration_id": record.calibration_id,
                "dataset_manifest_id": record.dataset_manifest_id,
                "dataset_manifest_sha256": record.dataset_manifest_sha256,
                "evaluator_id": record.evaluator_id,
                "evaluator_sha256": record.evaluator_sha256,
                "baseline_source_sha256": record.baseline_source_sha256,
                "config_sha256": record.config_sha256,
                "split": record.split,
            },
            created_at=datetime.fromisoformat(now),
        )

    def list_baseline_calibrations(self) -> tuple[BaselineCalibrationRecord, ...]:
        """List typed calibrations, with a narrow legacy generic fallback."""
        with self._connect() as connection:
            typed_rows = connection.execute(
                "SELECT calibration_id, payload_json, content_sha256 "
                "FROM authority_baseline_calibrations ORDER BY created_at, calibration_id"
            ).fetchall()
            legacy_rows = connection.execute(
                "SELECT record_id, payload_json, content_sha256 FROM records "
                "WHERE kind = 'baseline_calibration' ORDER BY record_id"
            ).fetchall()
        records: list[BaselineCalibrationRecord] = []
        for calibration_id, payload, digest in typed_rows:
            if digest is None or _content_hash(payload) != digest:
                raise PersistenceConflictError(
                    f"baseline calibration {calibration_id} integrity check failed"
                )
            records.append(BaselineCalibrationRecord.model_validate_json(payload))
        typed_ids = {record.calibration_id for record in records}
        for record_id, payload, digest in legacy_rows:
            if digest is None or _content_hash(payload) != digest:
                raise PersistenceConflictError(
                    f"persisted baseline calibration {record_id} integrity check failed"
                )
            if record_id not in typed_ids:
                records.append(BaselineCalibrationRecord.model_validate_json(payload))
        return tuple(records)

    def put_baseline_calibration(
        self,
        record: BaselineCalibrationRecord,
        actor_type: Literal["agent", "controller", "human"],
        actor_id: str,
    ) -> None:
        """Atomically persist an immutable calibration and its audit event."""
        payload = record.model_dump_json()
        digest = _content_hash(payload)
        now = datetime.now(UTC).isoformat()
        event = self._baseline_calibration_event(record, actor_type, actor_id, now)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT content_sha256 FROM authority_baseline_calibrations "
                "WHERE calibration_id = ?",
                (record.calibration_id,),
            ).fetchone()
            if existing is not None and existing[0] != digest:
                raise PersistenceConflictError(
                    f"baseline calibration {record.calibration_id} content changed"
                )
            if existing is None:
                connection.execute(
                    "INSERT INTO authority_baseline_calibrations "
                    "(calibration_id, payload_json, content_sha256, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (record.calibration_id, payload, digest, now),
                )
            self._ensure_audit_event(connection, event, preserve_existing_actor=True)

    def get_run_baseline(self, run_id: str) -> RunBaselineBinding | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM authority_run_baseline_bindings WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return RunBaselineBinding.model_validate_json(row[0]) if row else None

    def put_run_baseline(
        self,
        binding: RunBaselineBinding,
        actor_type: Literal["agent", "controller", "human"] = "controller",
        actor_id: str = "production-operations",
    ) -> None:
        """Persist one immutable run binding and its audit event atomically."""
        payload = binding.model_dump_json()
        event = AuditEvent(
            event_id=f"baseline-bound-{binding.run_id}",
            run_id=binding.run_id,
            event_type="baseline_bound",
            actor_type=actor_type,
            actor_id=actor_id,
            payload={
                "run_id": binding.run_id,
                "calibration_id": binding.calibration_id,
                "baseline_evaluation_id": binding.baseline_evaluation_id,
                "dataset_manifest_id": binding.dataset_manifest_id,
                "dataset_manifest_sha256": binding.dataset_manifest_sha256,
                "evaluator_id": binding.evaluator_id,
                "evaluator_sha256": binding.evaluator_sha256,
                "split": binding.split,
                "metrics": binding.model_dump(mode="json")["metrics"],
            },
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT content_sha256 FROM authority_run_baseline_bindings WHERE run_id = ?",
                (binding.run_id,),
            ).fetchone()
            if existing is not None and existing[0] != _content_hash(payload):
                raise PersistenceConflictError(
                    f"run baseline binding {binding.run_id} content changed"
                )
            calibration = connection.execute(
                "SELECT payload_json FROM authority_baseline_calibrations WHERE calibration_id = ?",
                (binding.calibration_id,),
            ).fetchone()
            if calibration is None:
                raise ValueError("run baseline binding references an unknown calibration")
            calibration_record = BaselineCalibrationRecord.model_validate_json(calibration[0])
            calibration_metrics = {
                metric.name: metric.value for metric in calibration_record.evaluation.metrics
            }
            binding_metrics = {metric.name: metric.value for metric in binding.metrics}
            if (
                calibration_record.evaluation.evaluation_id != binding.baseline_evaluation_id
                or calibration_record.dataset_manifest_id != binding.dataset_manifest_id
                or calibration_record.dataset_manifest_sha256 != binding.dataset_manifest_sha256
                or calibration_record.evaluator_id != binding.evaluator_id
                or calibration_record.evaluator_sha256 != binding.evaluator_sha256
                or calibration_record.split != binding.split
                or calibration_metrics != binding_metrics
            ):
                raise ValueError("run baseline binding does not match its calibration")
            if existing is None:
                connection.execute(
                    "INSERT INTO authority_run_baseline_bindings "
                    "(run_id, calibration_id, payload_json, content_sha256, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        binding.run_id,
                        binding.calibration_id,
                        payload,
                        _content_hash(payload),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            self._ensure_audit_event(connection, event)

    def get_baseline_calibration(self, calibration_id: str) -> BaselineCalibrationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, content_sha256 FROM authority_baseline_calibrations "
                "WHERE calibration_id = ?",
                (calibration_id,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    "SELECT payload_json, content_sha256 FROM records "
                    "WHERE kind = 'baseline_calibration' AND record_id = ?",
                    (calibration_id,),
                ).fetchone()
        if row is None:
            return None
        if row[1] is None or _content_hash(row[0]) != row[1]:
            raise PersistenceConflictError(
                f"baseline calibration {calibration_id} integrity check failed"
            )
        return BaselineCalibrationRecord.model_validate_json(row[0])

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
        """Compatibility store for non-authority artifacts.

        Authority records cannot be updated through this generic seam.
        """
        if kind == "baseline_calibration":
            raise ValueError("baseline calibration requires typed atomic persistence")
        if kind == "run_baseline_binding":
            raise ValueError("run baseline binding requires typed atomic persistence")
        if kind in {
            "experiment",
            "source_registration",
            "finalization",
            "validation_operation",
            "validation_report",
            "validation_blocker",
            "blocker_resolution",
        }:
            raise ValueError(f"generic persistence is not allowed for authority kind {kind}")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json, content_sha256 FROM records WHERE kind = ? AND record_id = ?",
                (kind, record_id),
            ).fetchone()
            if existing is not None:
                if existing[1] is None or _content_hash(existing[0]) != existing[1]:
                    raise PersistenceConflictError(
                        f"record {kind}/{record_id} integrity check failed"
                    )
                if existing[0] != payload_json:
                    raise PersistenceConflictError(f"record {kind}/{record_id} content changed")
            if existing is None:
                connection.execute(
                    "INSERT INTO records (kind, record_id, payload_json, content_sha256) "
                    "VALUES (?, ?, ?, ?)",
                    (kind, record_id, payload_json, _content_hash(payload_json)),
                )

    def list_json(self, kind: str) -> tuple[str, ...]:
        if kind == "baseline_calibration":
            return tuple(record.model_dump_json() for record in self.list_baseline_calibrations())
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json, content_sha256 FROM records "
                "WHERE kind = ? ORDER BY record_id",
                (kind,),
            ).fetchall()
        values: list[str] = []
        for payload, digest in rows:
            if digest is None or _content_hash(payload) != digest:
                raise PersistenceConflictError(f"record {kind} integrity check failed")
            values.append(payload)
        return tuple(values)
