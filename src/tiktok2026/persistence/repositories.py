from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tiktok2026.contracts import (
    ArtifactRecord,
    AuditEvent,
    BlockerResolution,
    EvaluatorIdentity,
    ExperimentRegistryEntry,
    ExperimentSpec,
    FinalizationRecord,
    FinalTestAuthorizationRequest,
    FinalTestClaim,
    FinalTestRequest,
    ImplementationValidationAuthority,
    ProvisionalFinalizationRequest,
    RunRecord,
    SourceRegistration,
    ValidationBlocker,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationStage,
)
from tiktok2026.persistence.migrations import MigrationRunner, application_migrations_path
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
        """Copy 001-003 records into append-only authority tables once."""
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
                if connection.execute(
                    "SELECT 1 FROM experiment_states WHERE experiment_id = ? LIMIT 1",
                    (experiment_id,),
                ).fetchone() is None:
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
                if connection.execute(
                    "SELECT 1 FROM run_states WHERE run_id = ? LIMIT 1", (run_id,)
                ).fetchone() is None:
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
                "SELECT payload_json FROM records WHERE kind = 'transition' AND record_id = ?",
                (record_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise PersistenceConflictError(
                        f"transition {record_id} content changed"
                    )
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
                        raise PersistenceConflictError(
                            f"audit event {event_id} content changed"
                        )
                return

            rows = connection.execute(
                "SELECT payload_json FROM records WHERE kind = 'transition'"
            ).fetchall()
            versions = {
                int(value["state_version"])
                for (raw,) in rows
                if (value := json.loads(raw)).get("run_id") == run_id
            }
            if state_version == 1:
                if versions:
                    raise PersistenceConflictError("transition version one has a predecessor")
            elif state_version - 1 not in versions or any(
                version > state_version for version in versions
            ):
                raise PersistenceConflictError("transition CAS predecessor is stale")
            connection.execute(
                "INSERT INTO records (kind, record_id, payload_json) VALUES ('transition', ?, ?)",
                (record_id, payload),
            )
            self._insert_audit(connection, event)

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
                    raise PersistenceConflictError(
                        f"artifact {record.artifact_id} content changed"
                    )
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
        expected_parent = (
            self.database.parent
            / "artifacts"
            / run_id
            / experiment_id
        ).resolve()
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
        if _content_hash(
            json.dumps(subject_for_identity, sort_keys=True, separators=(",", ":"))
        ) != operation.subject_sha256:
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
                raise PersistenceConflictError(
                    "validation authority diff does not match operation"
                )
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
        bound_report = report.model_copy(
            update={"validation_operation_id": operation.operation_id}
        )
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
                    if blocker_row[1] != blocker_digest:
                        raise PersistenceConflictError(
                            f"validation blocker {blocker.blocker_id} content changed"
                        )
                    continue
                connection.execute(
                    "INSERT INTO authority_validation_blockers "
                    "(blocker_id, report_id, experiment_id, stage, blocker_json, "
                    "content_sha256, created_at, validation_operation_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        blocker.blocker_id,
                        report.report_id,
                        blocker.experiment_id,
                        blocker.stage.value,
                        blocker_payload,
                        blocker_digest,
                        now,
                        operation.operation_id,
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
            if report.resolves_blocker_ids:
                if not report.evidence_refs:
                    raise ValueError("blocker resolutions require validation evidence")
                for blocker_id in report.resolves_blocker_ids:
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
                    resolution = BlockerResolution(
                        resolution_id=self._resolution_id(
                            report.report_id, blocker_id, report.evidence_refs
                        ),
                        blocker_id=blocker_id,
                        report_id=report.report_id,
                        experiment_id=report.experiment_id,
                        evidence_refs=report.evidence_refs,
                        validation_operation_id=operation.operation_id,
                    )
                    resolution_payload = resolution.model_dump_json()
                    resolution_digest = _content_hash(resolution_payload)
                    prior = connection.execute(
                        "SELECT resolution_id, resolution_json, content_sha256 "
                        "FROM authority_blocker_resolutions WHERE blocker_id = ?",
                        (blocker_id,),
                    ).fetchone()
                    if prior is not None:
                        if prior[2] != resolution_digest:
                            raise PersistenceConflictError(
                                f"blocker {blocker_id} already has a different resolution"
                            )
                        continue
                    connection.execute(
                        "INSERT INTO authority_blocker_resolutions "
                        "(resolution_id, blocker_id, report_id, experiment_id, "
                        "resolution_json, content_sha256, created_at, validation_operation_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            resolution.resolution_id,
                            resolution.blocker_id,
                            resolution.report_id,
                            resolution.experiment_id,
                            resolution_payload,
                            resolution_digest,
                            now,
                            operation.operation_id,
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
        report = self.get_validation_report(resolution.report_id)
        if report is None or report.verdict.value != "approved":
            raise ValueError("blocker resolution requires an approved validation report")
        if resolution.blocker_id not in report.resolves_blocker_ids:
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
                "SELECT experiment_id, report_id FROM authority_validation_blockers "
                "WHERE blocker_id = ?",
                (resolution.blocker_id,),
            ).fetchone()
            if blocker is None:
                raise ValueError("validation blocker does not exist")
            if blocker[0] != report.experiment_id:
                raise ValueError("blocker resolution experiment does not match blocker")
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
            existing = connection.execute(
                "SELECT resolution_json, content_sha256 FROM authority_blocker_resolutions "
                "WHERE blocker_id = ?",
                (resolution.blocker_id,),
            ).fetchone()
            if existing is not None:
                if existing[1] != digest:
                    raise PersistenceConflictError(
                        f"blocker {resolution.blocker_id} already has a different resolution"
                    )
                return
            connection.execute(
                "INSERT INTO authority_blocker_resolutions "
                "(resolution_id, blocker_id, report_id, experiment_id, resolution_json, "
                "content_sha256, created_at, validation_operation_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    resolution.resolution_id,
                    resolution.blocker_id,
                    resolution.report_id,
                    resolution.experiment_id,
                    payload,
                    digest,
                    now,
                    resolution.validation_operation_id,
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

    def get_validation_report_by_operation(
        self, operation_id: str
    ) -> ValidationReport | None:
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

    def get_validation_operation(
        self, operation_id: str
    ) -> ValidationOperationIdentity | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT operation_json FROM authority_validation_operations "
                "WHERE operation_id = ?",
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
                "SELECT resolution_json FROM authority_blocker_resolutions "
                "WHERE resolution_id = ?",
                (resolution_id,),
            ).fetchone()
        return BlockerResolution.model_validate_json(row[0]) if row else None

    def get_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT blocker.blocker_json FROM authority_validation_blockers AS blocker "
                "LEFT JOIN authority_blocker_resolutions AS resolution "
                "ON resolution.blocker_id = blocker.blocker_id "
                "WHERE blocker.experiment_id = ? AND resolution.blocker_id IS NULL "
                "ORDER BY blocker.created_at, blocker.blocker_id",
                (experiment_id,),
            ).fetchall()
        return tuple(ValidationBlocker.model_validate_json(row[0]) for row in rows)

    def get_unresolved_blocker_ids(self, experiment_id: str) -> tuple[str, ...]:
        return tuple(blocker.blocker_id for blocker in self.get_unresolved_blockers(experiment_id))

    def list_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        return self.get_unresolved_blockers(experiment_id)

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
            if not connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone():
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
        self, limit: int = 50
    ) -> tuple[tuple[ExperimentRegistryEntry, ...], int]:
        if limit < 1:
            raise ValueError("experiment registry limit must be positive")
        with self._connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM authority_experiments").fetchone()[0]
            )
            rows = connection.execute(
                "SELECT authority.spec_json, COALESCE(("
                "SELECT state.status FROM experiment_states AS state "
                "WHERE state.experiment_id = authority.experiment_id "
                "ORDER BY state.sequence DESC LIMIT 1"
                "), 'proposed') "
                "FROM authority_experiments AS authority "
                "ORDER BY authority.created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
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
                "content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    registration.registration_id,
                    registration.experiment_id,
                    registration.revision,
                    payload,
                    digest,
                    now,
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
                "SELECT registration_json FROM source_registrations "
                "WHERE experiment_id = ? ORDER BY revision DESC LIMIT 1",
                (experiment_id,),
            ).fetchone()
        return SourceRegistration.model_validate_json(row[0]) if row else None

    def get_source_registration_by_id(
        self, registration_id: str
    ) -> SourceRegistration | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT registration_json FROM source_registrations WHERE registration_id = ?",
                (registration_id,),
            ).fetchone()
        return SourceRegistration.model_validate_json(row[0]) if row else None

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
                "SELECT status FROM run_states WHERE run_id = ? "
                "ORDER BY sequence DESC LIMIT 1",
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
            self._validate_source(
                connection, request.experiment_id, request.run_id, request.source_commit
            )
            evaluator = connection.execute(
                "SELECT 1 FROM evaluator_identities WHERE evaluator_id = ?",
                (request.evaluator_id,),
            ).fetchone()
            if evaluator is None:
                raise FinalTestAccessError("evaluator identity is not registered")
            evaluation = connection.execute(
                "SELECT payload_json FROM records WHERE kind = 'evaluation' AND record_id = ?",
                (request.evaluation_id,),
            ).fetchone()
            if evaluation is None:
                raise FinalTestAccessError("evaluation provenance is unavailable")
            evaluation_payload = json.loads(evaluation[0])
            if (
                evaluation_payload.get("experiment_id") != request.experiment_id
                or evaluation_payload.get("checkpoint_id") != request.checkpoint_id
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
                "SELECT finalization_json FROM authority_finalizations "
                "WHERE finalization_id = ?",
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

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
        """Compatibility store for non-authority artifacts.

        Authority records use their typed methods above and cannot be updated here.
        """
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
                "SELECT payload_json FROM records WHERE kind = ? AND record_id = ?",
                (kind, record_id),
            ).fetchone()
            if existing is not None and existing[0] != payload_json:
                raise PersistenceConflictError(f"record {kind}/{record_id} content changed")
            if existing is None:
                connection.execute(
                    "INSERT INTO records (kind, record_id, payload_json) VALUES (?, ?, ?)",
                    (kind, record_id, payload_json),
                )

    def list_json(self, kind: str) -> tuple[str, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM records WHERE kind = ? ORDER BY record_id", (kind,)
            ).fetchall()
        return tuple(row[0] for row in rows)
