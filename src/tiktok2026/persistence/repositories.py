from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tiktok2026.contracts import (
    ArtifactRecord,
    AuditEvent,
    EvaluatorIdentity,
    ExperimentSpec,
    FinalizationRecord,
    FinalTestAuthorizationRequest,
    FinalTestClaim,
    FinalTestRequest,
    ProvisionalFinalizationRequest,
    RunRecord,
    SourceRegistration,
)
from tiktok2026.persistence.migrations import MigrationRunner, application_migrations_path
from tiktok2026.repository.diffs import patch_signature


class PersistenceConflictError(RuntimeError):
    """An immutable authority record was replayed with different content."""


class FinalTestAccessError(PermissionError):
    """A final-test request does not satisfy controller policy."""


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
                "SELECT content_sha256 FROM source_registrations WHERE experiment_id = ?",
                (registration.experiment_id,),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise PersistenceConflictError(
                        f"source registration {registration.experiment_id} content changed"
                    )
                if audit_event is not None:
                    self._insert_audit(connection, audit_event)
                return
            connection.execute(
                "INSERT INTO source_registrations "
                "(experiment_id, registration_json, content_sha256, created_at) "
                "VALUES (?, ?, ?, ?)",
                (registration.experiment_id, payload, digest, now),
            )
            event = audit_event or self._automatic_event(
                "source_registered",
                registration.run_id,
                registration.experiment_id,
                {"source_commit": registration.source_commit},
                now,
            )
            self._insert_audit(connection, event)

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT registration_json FROM source_registrations WHERE experiment_id = ?",
                (experiment_id,),
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
        if kind in {"experiment", "source_registration", "finalization"}:
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
