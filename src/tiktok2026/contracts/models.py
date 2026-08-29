from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
FullCommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentRole(StrEnum):
    ORCHESTRATION = "orchestration"
    RESEARCH = "research"
    IMPLEMENTOR = "implementor"
    VALIDATOR = "validator"


class Fidelity(StrEnum):
    SMOKE = "smoke"
    PROXY = "proxy"
    FULL = "full"


class DecisionAction(StrEnum):
    RESEARCH = "research"
    IMPLEMENT = "implement"
    VALIDATE = "validate"
    REPAIR = "repair"
    REPLICATE = "replicate"
    INCREASE_FIDELITY = "increase_fidelity"
    REVISIT_BRANCH = "revisit_branch"
    STOP = "stop"


class ValidationStage(StrEnum):
    PROPOSAL = "proposal"
    IMPLEMENTATION = "implementation"
    RESULT = "result"


class ValidationVerdict(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REPAIRABLE = "repairable"
    INCONCLUSIVE = "inconclusive"


class FailureKind(StrEnum):
    SYNTAX_IMPORT = "syntax_import"
    DEPENDENCY_ENVIRONMENT = "dependency_environment"
    MISSING_PATH = "missing_path"
    CUDA_OOM = "cuda_oom"
    CPU_OOM = "cpu_oom"
    NAN_DIVERGENCE = "nan_divergence"
    SCHEMA_MISMATCH = "schema_mismatch"
    EVALUATOR_OUTPUT = "evaluator_output"
    TIMEOUT = "timeout"
    DISK = "disk"
    CORRUPTED_CHECKPOINT = "corrupted_checkpoint"
    UNSTABLE_VALIDATION = "unstable_validation"
    SCIENTIFIC_NON_IMPROVEMENT = "scientific_non_improvement"


class RunPhase(StrEnum):
    BOOTSTRAP = "bootstrap"
    RESEARCH = "research"
    IMPLEMENT = "implement"
    EXECUTE = "execute"
    EVALUATE = "evaluate"
    PERSIST = "persist"
    FINALIZE = "finalize"
    COMPLETE = "complete"


class ArtifactRetention(StrEnum):
    TEMPORARY = "temporary"
    RUN = "run"
    CHAMPION = "champion"
    PROVENANCE = "provenance"


class RuntimePaths(ContractModel):
    root: Path
    application_db: Path
    graph_db: Path
    artifacts: Path
    worktrees: Path
    traces: Path
    exports: Path
    locks: Path
    literature: Path
    temporary: Path

    @classmethod
    def create(cls, repository_root: Path, runtime_root: Path) -> RuntimePaths:
        repository = repository_root.resolve()
        root = runtime_root.resolve()
        if root == repository or repository in root.parents:
            raise ValueError("runtime root must be outside the repository")
        directories = {
            "artifacts": root / "artifacts",
            "worktrees": root / "worktrees",
            "traces": root / "traces",
            "exports": root / "exports",
            "locks": root / "locks",
            "literature": root / "literature",
            "temporary": root / "tmp",
        }
        root.mkdir(parents=True, exist_ok=True)
        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            root=root,
            application_db=root / "application.sqlite3",
            graph_db=root / "graph.sqlite3",
            **directories,
        )


class MetricValue(ContractModel):
    name: Literal["NDCG@10", "Recall@50"]
    value: Annotated[float, Field(ge=0.0, le=1.0)]


class Hypothesis(ContractModel):
    schema_version: Literal["1"] = "1"
    hypothesis_id: str
    statement: str
    mechanism: str
    evidence_refs: tuple[str, ...] = ()


class ExperimentSpec(ContractModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    hypothesis_id: str
    hypothesis: str
    mechanism: str
    motivation: str
    evidence_refs: tuple[str, ...] = ()
    parent_experiment_id: str | None = None
    expected_signal: str
    implementation_scope: tuple[str, ...]
    fidelity: Fidelity
    predicted_gpu_hours: Annotated[float, Field(ge=0.0)] = 0.0
    success_criteria: str
    failure_criteria: str
    leakage_risks: tuple[str, ...] = ()
    source_provenance: tuple[str, ...] = ()


class EvidenceItem(ContractModel):
    evidence_id: str
    kind: str
    summary: str
    source_ref: str
    authorized: bool = True
    contains_test_labels: bool = False


class ResearchDecision(ContractModel):
    request_id: str
    kind: Literal["proposal", "evidence_request", "interpretation"]
    experiment_spec: ExperimentSpec | None = None
    message: str
    evidence_refs: tuple[str, ...] = ()


class OrchestrationDecision(ContractModel):
    schema_version: Literal["1"] = "1"
    decision_id: str
    action: DecisionAction
    target_experiment_id: str | None = None
    fidelity: Fidelity | None = None
    evidence_refs: tuple[str, ...] = ()
    rationale: str


class ImplementationResult(ContractModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    patch_artifact_id: str
    changed_files: tuple[str, ...]
    changed_symbols: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_issues: tuple[str, ...] = ()


class ValidationReport(ContractModel):
    schema_version: Literal["1"] = "1"
    report_id: str
    experiment_id: str
    stage: ValidationStage
    verdict: ValidationVerdict
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    leakage_risk: str
    implementation_fidelity: str | None = None
    scientific_confidence: str | None = None


class ExecutionRequest(ContractModel):
    execution_id: str
    experiment_id: str
    source_commit: CommitSha
    command: tuple[str, ...]
    image: str
    source_path: Path
    dataset_path: Path
    output_path: Path
    timeout_seconds: Annotated[int, Field(gt=0)]
    memory_bytes: Annotated[int, Field(gt=0)]
    cpus: Annotated[float, Field(gt=0)]
    gpu_count: Annotated[int, Field(ge=0)] = 0


class ExecutionResult(ContractModel):
    schema_version: Literal["1"] = "1"
    execution_id: str
    experiment_id: str
    source_commit: str
    command: tuple[str, ...]
    exit_code: int
    elapsed_seconds: Annotated[float, Field(ge=0.0)]
    gpu_hours: Annotated[float, Field(ge=0.0)]
    artifact_ids: tuple[str, ...] = ()
    failure_kind: FailureKind | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> ExecutionResult:
        if self.exit_code == 0 and self.failure_kind is not None:
            raise ValueError("successful execution cannot have a failure kind")
        if self.exit_code != 0 and self.failure_kind is None:
            raise ValueError("failed execution requires a failure kind")
        return self


class EvaluationContext(ContractModel):
    """Controller-owned, immutable context for one offline evaluation."""

    run_id: str
    evaluation_id: str
    experiment_id: str
    checkpoint_id: str
    source_commit: FullCommitSha
    execution_id: str
    dataset_manifest_id: str
    dataset_manifest_sha256: Sha256
    split: Literal["valid", "test"]
    prediction_artifact_id: str
    prediction_sha256: Sha256
    evaluator_id: str
    evaluator_sha256: Sha256
    authorization_claim_id: str | None = None


class FinalTestAuthorizationClaim(ContractModel):
    """Authoritative claim returned by the persistence-backed resolver."""

    claim_id: str
    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    evaluator_id: str
    evaluator_sha256: Sha256 | None = None
    dataset_manifest_id: str
    dataset_manifest_sha256: Sha256
    split: Literal["test"]
    checkpoint_id: str
    execution_id: str
    prediction_artifact_id: str
    prediction_sha256: Sha256


class PredictionArtifactRegistration(ContractModel):
    artifact_id: str
    path: Path
    sha256: Sha256
    checkpoint_id: str
    source_commit: FullCommitSha
    execution_id: str
    dataset_manifest_id: str
    dataset_manifest_sha256: Sha256
    split: Literal["valid", "test"]


class EvaluationRequest(ContractModel):
    evaluation_id: str
    context: EvaluationContext

    @model_validator(mode="after")
    def validate_context_identity(self) -> EvaluationRequest:
        if self.evaluation_id != self.context.evaluation_id:
            raise ValueError("evaluation request/context IDs do not match")
        return self


class PredictionRow(ContractModel):
    row_id: str
    user_id: str
    item_id: str
    score: float
    row_identity: tuple[str, ...]


class FinalTestAuthorizationRequest(ContractModel):
    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    evaluator_id: str
    evaluator_sha256: Sha256 | None = None
    dataset_manifest_id: str | None = None
    dataset_manifest_sha256: Sha256 | None = None
    split: Literal["test"] | None = None
    checkpoint_id: str | None = None
    execution_id: str | None = None
    prediction_artifact_id: str | None = None
    prediction_sha256: Sha256 | None = None


class FinalTestClaim(ContractModel):
    claim_id: str
    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    evaluator_id: str
    evaluator_sha256: Sha256 | None = None
    dataset_manifest_id: str | None = None
    dataset_manifest_sha256: Sha256 | None = None
    split: Literal["test"] | None = None
    checkpoint_id: str | None = None
    execution_id: str | None = None
    prediction_artifact_id: str | None = None
    prediction_sha256: Sha256 | None = None


class ProvisionalFinalizationRequest(ContractModel):
    finalization_id: str
    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    checkpoint_id: str
    evaluation_id: str
    bundle_artifact_id: str
    evaluator_id: str


class FinalTestRequest(ContractModel):
    """Evaluator-side completion request; authorization is mandatory."""

    claim_id: str
    finalization_id: str
    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    checkpoint_id: str
    evaluation_id: str
    bundle_artifact_id: str
    evaluator_id: str


class EvaluationResult(ContractModel):
    schema_version: Literal["1"] = "1"
    evaluation_id: str
    experiment_id: str
    checkpoint_id: str
    metrics: tuple[MetricValue, ...]
    evaluator_artifact_id: str
    evaluator_sha256: Sha256
    prediction_sha256: Sha256
    validity: Literal["provisional", "official", "invalid"]
    dataset_manifest_sha256: Sha256 | None = None
    split: Literal["valid", "test"] | None = None
    run_id: str | None = None
    source_commit: FullCommitSha | None = None
    execution_id: str | None = None
    dataset_manifest_id: str | None = None
    prediction_artifact_id: str | None = None

    @property
    def validation_score(self) -> float:
        values = {metric.name: metric.value for metric in self.metrics}
        if set(values) != {"NDCG@10", "Recall@50"}:
            raise ValueError("both judging metrics are required")
        return (values["NDCG@10"] + values["Recall@50"]) / 2.0


class FailureRecord(ContractModel):
    schema_version: Literal["1"] = "1"
    failure_id: str
    experiment_id: str
    kind: FailureKind
    evidence_refs: tuple[str, ...]
    repair_attempt: Annotated[int, Field(ge=0, le=2)]
    scientific_evidence: bool = False


class ResourceState(ContractModel):
    schema_version: Literal["1"] = "1"
    remaining_gpu_hours: Annotated[float, Field(ge=0.0)]
    accumulated_gpu_hours: Annotated[float, Field(ge=0.0)]
    remaining_wall_seconds: Annotated[float, Field(ge=0.0)]
    used_tokens: Annotated[int, Field(ge=0)]
    remaining_tokens: Annotated[int, Field(ge=0)]
    disk_bytes_available: Annotated[int, Field(ge=0)]
    reserved_final_gpu_hours: Annotated[float, Field(ge=0.0)]
    accumulated_wall_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    used_disk_bytes: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_final_reserve(self) -> ResourceState:
        if self.reserved_final_gpu_hours > self.remaining_gpu_hours:
            raise ValueError("final GPU reserve cannot exceed remaining GPU hours")
        return self


class ResearchRequest(ContractModel):
    request_id: str
    objective: str
    resource_state: ResourceState
    parent_experiment_id: str | None = None
    allowed_paths: tuple[str, ...] = ("src/tiktok2026/experiment",)


class ArtifactRecord(ContractModel):
    artifact_id: str
    run_id: str
    experiment_id: str | None = None
    kind: str
    uri: str
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0)]
    producer: str
    retention: ArtifactRetention
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResourceReservation(ContractModel):
    reservation_id: str
    run_id: str
    experiment_id: str | None = None
    gpu_hours: Annotated[float, Field(ge=0.0)]
    wall_seconds: Annotated[float, Field(ge=0.0)]
    tokens: Annotated[int, Field(ge=0)]
    disk_bytes: Annotated[int, Field(ge=0)]
    purpose: Literal["iteration", "final"] = "iteration"


class ResourceUsage(ContractModel):
    gpu_hours: Annotated[float, Field(ge=0.0)]
    wall_seconds: Annotated[float, Field(ge=0.0)]
    tokens: Annotated[int, Field(ge=0)]
    disk_bytes: Annotated[int, Field(ge=0)]


class RunRecord(ContractModel):
    run_id: str
    status: str


class EvaluatorIdentity(ContractModel):
    evaluator_id: str
    evaluator_sha256: Sha256
    validity: Literal["provisional", "official"]


class WorktreeAssignment(ContractModel):
    worktree_id: str
    run_id: str
    experiment_id: str
    path: Path
    branch: str
    parent_commit: FullCommitSha


class SourceRegistration(ContractModel):
    experiment_id: str
    run_id: str
    parent_commit: FullCommitSha
    source_commit: FullCommitSha
    patch_sha256: Sha256
    patch_artifact_id: str
    patch_artifact_uri: str
    allowed_scopes: tuple[str, ...]
    eligible: bool = False


class ModelUsage(ContractModel):
    role: AgentRole
    model: str
    prompt_tokens: Annotated[int, Field(ge=0)]
    completion_tokens: Annotated[int, Field(ge=0)]


class AgentFailure(ContractModel):
    request_id: str
    role: AgentRole
    kind: Literal["model", "schema", "policy", "capability"]
    message: str
    repair_attempts: Annotated[int, Field(ge=0, le=1)]


class LessonRecord(ContractModel):
    lesson_id: str
    statement: str
    evidence_strength: Literal["weak", "moderate", "strong"]
    experiment_ids: tuple[str, ...]
    tags: tuple[str, ...] = ()


class FrontierCandidate(ContractModel):
    experiment_id: str
    slot: Literal["champion", "alternative", "diagnostic"]
    score: float
    diversity_tags: tuple[str, ...] = ()


class GraphStateReference(ContractModel):
    run_id: str
    phase: RunPhase
    current_experiment_id: str | None = None
    repair_attempts: Annotated[int, Field(ge=0, le=2)] = 0
    terminal_reason: str | None = None


class FinalizationRecord(ContractModel):
    finalization_id: str
    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    checkpoint_id: str
    evaluation_id: str
    validity: Literal["provisional", "official"]
    bundle_artifact_id: str
    consumed_test_access: bool


class AuditEvent(ContractModel):
    schema_version: Literal["1"] = "1"
    event_id: str
    run_id: str
    experiment_id: str | None = None
    event_type: str
    actor_type: Literal["agent", "controller", "human"]
    actor_id: str
    payload: dict[str, object]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
