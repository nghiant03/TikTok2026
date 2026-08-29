from __future__ import annotations

from pathlib import Path

from tiktok2026.contracts import (
    AuditEvent,
    ContractModel,
    EvaluationResult,
    ExperimentSpec,
    FailureRecord,
    FinalizationRecord,
    PolicyDecisionModel,
    ProvenanceRequest,
    ResourceReservation,
    ResourceState,
    RunRecord,
    SourceRegistration,
    WorktreeAssignment,
)
from tiktok2026.observability.exports import export_records
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.persistence.resources import ResourceLedger
from tiktok2026.policies.lifecycle import can_repair
from tiktok2026.policies.paths import check_changed_paths

# ---------------------------------------------------------------------------
# RepositoryRunStore — wraps ApplicationRepository as a RunStore
# ---------------------------------------------------------------------------


class RepositoryRunStore:
    """RunStore over ApplicationRepository with typed persistence.

    Uses the generic ``put_json``/``list_json`` for records that lack
    dedicated authority methods (evaluations, worktree assignments).
    Experiment, run, audit, source registration, and finalization go
    through the repository's typed authority methods.
    """

    def __init__(self, repository: ApplicationRepository) -> None:
        self._repo = repository

    # --- typed authority methods ---

    def put_experiment(
        self,
        spec: ExperimentSpec,
        status: str,
        run_id: str,
        transition_id: str,
        expected_predecessor: str | None = None,
        audit_event: ContractModel | None = None,
    ) -> None:
        self._repo.put_experiment(
            spec=spec,
            status=status,
            run_id=run_id,
            transition_id=transition_id,
            expected_predecessor=expected_predecessor,
            audit_event=audit_event,  # type: ignore[arg-type]
        )

    def put_run(self, record: RunRecord, transition_id: str) -> None:
        self._repo.put_run(
            run=record,
            transition_id=transition_id,
            expected_predecessor=None,
        )

    def put_audit_event(self, event: ContractModel) -> None:
        if isinstance(event, AuditEvent):
            self._repo.put_audit_event(event)

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None:
        return self._repo.get_source_registration(experiment_id)

    def persist_provisional_finalization(
        self, request: ContractModel
    ) -> FinalizationRecord:
        from tiktok2026.contracts import ProvisionalFinalizationRequest
        return self._repo.persist_provisional_finalization(
            request if isinstance(request, ProvisionalFinalizationRequest)
            else ProvisionalFinalizationRequest(**request.model_dump())
        )

    # --- generic record methods ---

    def put_evaluation(self, result: EvaluationResult, provenance: ProvenanceRequest) -> None:
        self._repo.put_json("evaluation", result.evaluation_id, result.model_dump_json())

    def put_failure(self, record: FailureRecord, run_id: str) -> None:
        self._repo.put_json("failure", record.failure_id, record.model_dump_json())

    def put_worktree_assignment(self, assignment: WorktreeAssignment) -> None:
        self._repo.put_json(
            "worktree_assignment", assignment.experiment_id, assignment.model_dump_json()
        )

    def get_worktree_assignment(self, experiment_id: str) -> WorktreeAssignment | None:
        for record_json in self._repo.list_json("worktree_assignment"):
            record = WorktreeAssignment.model_validate_json(record_json)
            if record.experiment_id == experiment_id:
                return record
        return None

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
        self._repo.put_json(kind, record_id, payload_json)

    def list_json(self, kind: str) -> tuple[str, ...]:
        return self._repo.list_json(kind)


# ---------------------------------------------------------------------------
# DeterministicPolicyGate — wraps pure policy functions
# ---------------------------------------------------------------------------


class DeterministicPolicyGate:
    """PolicyGate backed by the pure policy functions in ``policies/``."""

    def check_paths(
        self, changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
    ) -> PolicyDecisionModel:
        decision = check_changed_paths(changed_paths, allowed_scopes)
        return PolicyDecisionModel(allowed=decision.allowed, reason=decision.reason)

    def can_repair(self, repair_attempts: int) -> PolicyDecisionModel:
        decision = can_repair(repair_attempts)
        return PolicyDecisionModel(allowed=decision.allowed, reason=decision.reason)


# ---------------------------------------------------------------------------
# LedgerResourceAccountant — wraps ResourceLedger
# ---------------------------------------------------------------------------


class LedgerResourceAccountant:
    """ResourceAccountant backed by ResourceLedger."""

    def __init__(self, ledger: ResourceLedger) -> None:
        self._ledger = ledger

    def state(self) -> ResourceState:
        return self._ledger.state()

    def reserve(self, reservation: ContractModel) -> bool:
        if isinstance(reservation, ResourceReservation):
            return self._ledger.reserve(reservation)
        return False

    def consume(self, reservation_id: str, **usage: float | int) -> bool:
        return self._ledger.consume(
            reservation_id,
            gpu_hours=usage.get("gpu_hours"),  # type: ignore[arg-type]
            wall_seconds=usage.get("wall_seconds"),  # type: ignore[arg-type]
            tokens=usage.get("tokens"),  # type: ignore[arg-type]
            disk_bytes=usage.get("disk_bytes"),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# RepositoryExportService — wraps export_records
# ---------------------------------------------------------------------------


class RepositoryExportService:
    """ExportService that reconstructs records from ApplicationRepository
    and writes deterministic JSONL + Markdown exports."""

    def __init__(self, repository: ApplicationRepository, runtime_root: Path) -> None:
        self._repo = repository
        self._runtime_root = runtime_root

    async def export_run(self, run_id: str, output_dir: Path | None = None) -> dict[str, Path]:
        events = self._repo.list_audit_events(run_id)
        records = tuple(event.model_dump(mode="json") for event in events)
        dest = output_dir or self._runtime_root / "exports" / run_id
        jsonl_path, md_path = export_records(run_id, records, dest)
        return {"jsonl": jsonl_path, "markdown": md_path}