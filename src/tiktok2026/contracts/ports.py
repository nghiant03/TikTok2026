from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tiktok2026.contracts.models import (
    ArtifactRecord,
    ContractModel,
    EvaluationRequest,
    EvaluationResult,
    ExecutionRequest,
    ExecutionResult,
    ExperimentSpec,
    FailureRecord,
    FinalizationRecord,
    PolicyDecisionModel,
    ProvenanceRequest,
    ResourceState,
    RunRecord,
    SourceRegistration,
    WorktreeAssignment,
)


class AgentClient(Protocol):
    async def invoke(self, request: ContractModel) -> ContractModel: ...


class ArtifactRegistry(Protocol):
    def register(self, record: ArtifactRecord) -> None: ...


class WorktreeManager(Protocol):
    def create(
        self, run_id: str, spec: ExperimentSpec, parent_commit: str
    ) -> WorktreeAssignment: ...

    def register_source(
        self, assignment: WorktreeAssignment, allowed_scopes: tuple[str, ...]
    ) -> SourceRegistration: ...

    def remove(self, assignment: WorktreeAssignment) -> None: ...


class Executor(Protocol):
    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class RepositoryReader(Protocol):
    def read(self, relative_path: str, max_characters: int = 20_000) -> str: ...

    def search(self, query: str, max_results: int = 20) -> tuple[str, ...]: ...


class ScopedRepository(RepositoryReader, Protocol):
    def write(self, relative_path: str, content: str) -> None: ...

    def diff(self) -> str: ...

    def run_check(self, command: tuple[str, ...], timeout_seconds: int) -> str: ...


class DataSummaryReader(Protocol):
    def summarize(self, manifest_id: str) -> tuple[ContractModel, ...]: ...


class MemoryReader(Protocol):
    def retrieve(self, query: str, limit: int) -> tuple[ContractModel, ...]: ...


class LiteratureReader(Protocol):
    async def retrieve(self, query: str, limit: int) -> tuple[ContractModel, ...]: ...


class TraceSink(Protocol):
    def record(self, run_id: str, payload: ContractModel) -> Path: ...


# ---------------------------------------------------------------------------
# Phase 3 Lane B: new seam protocols
# ---------------------------------------------------------------------------


class ResourceAccountant(Protocol):
    """Seam for resource reservation, consumption, and state queries."""

    def state(self) -> ResourceState: ...

    def reserve(self, reservation: ContractModel) -> bool: ...

    def consume(self, reservation_id: str, **usage: float | int) -> bool: ...


class PolicyGate(Protocol):
    """Seam for deterministic policy decisions."""

    def check_paths(
        self, changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
    ) -> PolicyDecisionModel: ...

    def can_repair(self, repair_attempts: int) -> PolicyDecisionModel: ...


class RunStore(Protocol):
    """Seam for persisting experiments, evaluations, failures, audit events."""

    def put_experiment(
        self,
        spec: ExperimentSpec,
        status: str,
        run_id: str,
        transition_id: str,
        expected_predecessor: str | None = None,
        audit_event: ContractModel | None = None,
    ) -> None: ...

    def put_evaluation(self, result: EvaluationResult, provenance: ProvenanceRequest) -> None: ...

    def put_failure(self, record: FailureRecord, run_id: str) -> None: ...

    def put_run(self, record: RunRecord, transition_id: str) -> None: ...

    def put_audit_event(self, event: ContractModel) -> None: ...

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None: ...

    def put_worktree_assignment(self, assignment: WorktreeAssignment) -> None: ...

    def get_worktree_assignment(self, experiment_id: str) -> WorktreeAssignment | None: ...

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None: ...

    def list_json(self, kind: str) -> tuple[str, ...]: ...

    def persist_provisional_finalization(
        self, request: ContractModel
    ) -> FinalizationRecord: ...


class ExportService(Protocol):
    """Seam for writing deterministic Markdown and JSONL exports."""

    async def export_run(self, run_id: str, output_dir: Path) -> dict[str, Path]: ...


class FrontierService(Protocol):
    """Seam for updating the experiment frontier after persistence."""

    def update(self, experiment_id: str, score: float) -> str | None: ...


class AgentResultParser(Protocol):
    """Seam for parsing and repairing structured agent responses."""

    async def parse(
        self, client: AgentClient, request: ContractModel, model_type: type
    ) -> ContractModel: ...