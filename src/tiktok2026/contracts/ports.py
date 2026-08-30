from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tiktok2026.contracts.models import (
    ArtifactRecord,
    BaselineCalibrationRecord,
    BlockerResolution,
    ContractModel,
    DatasetManifestIdentity,
    EvaluationRequest,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionRequest,
    ExecutionResult,
    ExperimentRegistrySnapshot,
    ExperimentSpec,
    FailureRecord,
    FinalizationBundleRequest,
    FinalizationRecord,
    PolicyDecisionModel,
    PredictionArtifactRegistration,
    ProvenanceRequest,
    ResourceState,
    RunRecord,
    SourceRegistration,
    ValidationBlocker,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationStage,
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
        self,
        assignment: WorktreeAssignment,
        allowed_scopes: tuple[str, ...],
        previous: SourceRegistration | None = None,
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
    def read_base(self, relative_path: str, max_characters: int = 20_000) -> str: ...

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

    def reconcile(self, reservation_id: str, **usage: float | int) -> bool: ...


class PolicyGate(Protocol):
    """Seam for deterministic policy decisions."""

    def check_paths(
        self, changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
    ) -> PolicyDecisionModel: ...

    def can_repair(self, repair_attempts: int) -> PolicyDecisionModel: ...


class RunStore(Protocol):
    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None: ...

    def load_transition(self, run_id: str, state_version: int) -> dict[str, object] | None: ...

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

    def put_validation_report(
        self,
        report: ValidationReport,
        run_id: str,
        operation: ValidationOperationIdentity,
        subject: dict[str, object],
    ) -> None: ...

    def get_validation_report(self, report_id: str) -> ValidationReport | None: ...

    def get_validation_report_by_operation(
        self, operation_id: str
    ) -> ValidationReport | None: ...

    def get_validation_report_for_attempt(
        self, run_id: str, experiment_id: str, stage: ValidationStage, repair_attempt: int
    ) -> ValidationReport | None: ...

    def get_validation_operation(
        self, operation_id: str
    ) -> ValidationOperationIdentity | None: ...

    def list_validation_reports(
        self, experiment_id: str | None = None
    ) -> tuple[ValidationReport, ...]: ...

    def list_validation_blockers(
        self, experiment_id: str | None = None
    ) -> tuple[ValidationBlocker, ...]: ...

    def get_validation_blocker(self, blocker_id: str) -> ValidationBlocker | None: ...

    def put_blocker_resolution(self, resolution: BlockerResolution, run_id: str) -> None: ...

    def list_blocker_resolutions(
        self, experiment_id: str | None = None
    ) -> tuple[BlockerResolution, ...]: ...

    def get_blocker_resolution(self, resolution_id: str) -> BlockerResolution | None: ...

    def get_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]: ...

    def get_unresolved_blocker_ids(self, experiment_id: str) -> tuple[str, ...]: ...

    def list_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]: ...

    def put_failure(self, record: FailureRecord, run_id: str) -> None: ...

    def put_run(
        self,
        record: RunRecord,
        transition_id: str,
        expected_predecessor: str | None = None,
    ) -> None: ...

    def put_audit_event(self, event: ContractModel) -> None: ...

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None: ...

    def get_source_registration_by_id(
        self, registration_id: str
    ) -> SourceRegistration | None: ...

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None: ...

    def get_experiment_registry(self, limit: int = 50) -> ExperimentRegistrySnapshot: ...

    def put_source_registration(self, registration: SourceRegistration) -> None: ...

    def put_execution_result(self, result: ExecutionResult) -> None: ...

    def get_execution_result(self, execution_id: str) -> ExecutionResult | None: ...

    def get_evaluation_result(self, evaluation_id: str) -> EvaluationResult | None: ...

    def list_evaluation_results(self) -> tuple[EvaluationResult, ...]: ...

    def list_baseline_calibrations(self) -> tuple[BaselineCalibrationRecord, ...]: ...

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None: ...

    def put_artifact(self, record: ArtifactRecord) -> None: ...

    def put_evaluator_identity(self, identity: EvaluatorIdentity) -> None: ...

    def get_evaluator_identity(self, evaluator_id: str) -> EvaluatorIdentity | None: ...

    def put_dataset_manifest_identity(self, identity: DatasetManifestIdentity) -> None: ...

    def get_dataset_manifest_identity(self) -> DatasetManifestIdentity | None: ...

    def get_prediction_artifact(
        self, artifact_id: str
    ) -> PredictionArtifactRegistration | None: ...

    def put_worktree_assignment(self, assignment: WorktreeAssignment) -> None: ...

    def get_worktree_assignment(self, experiment_id: str) -> WorktreeAssignment | None: ...

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None: ...

    def list_json(self, kind: str) -> tuple[str, ...]: ...

    def persist_provisional_finalization(self, request: ContractModel) -> FinalizationRecord: ...

    def get_finalization(self, finalization_id: str) -> FinalizationRecord | None: ...


class TransitionStore(Protocol):
    """Durable CAS store for controller graph transitions."""

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None: ...


class ExportService(Protocol):
    """Seam for writing deterministic Markdown and JSONL exports."""

    async def export_run(self, run_id: str, output_dir: Path) -> dict[str, Path]: ...


class FinalizationBundleService(Protocol):
    """Create a persisted, provenance-bearing finalization bundle."""

    def create(self, request: FinalizationBundleRequest) -> ArtifactRecord: ...


class FrontierService(Protocol):
    """Seam for updating the experiment frontier after persistence."""

    def initialize(self, run_id: str) -> None: ...

    def update(self, experiment_id: str, score: float) -> str | None: ...


class AgentResultParser(Protocol):
    """Seam for parsing and repairing structured agent responses."""

    async def parse(
        self, client: AgentClient, request: ContractModel, model_type: type
    ) -> ContractModel: ...
