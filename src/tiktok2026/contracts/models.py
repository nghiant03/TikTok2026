from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
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
    implementation_scope: Annotated[tuple[str, ...], Field(min_length=1)]
    fidelity: Fidelity
    predicted_gpu_hours: Annotated[float, Field(ge=0.0)] = 0.0
    success_criteria: str
    failure_criteria: str
    leakage_risks: tuple[str, ...] = ()
    source_provenance: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_implementation_scope(self) -> ExperimentSpec:
        for scope in self.implementation_scope:
            path = PurePosixPath(scope)
            if (
                not scope
                or scope != scope.strip()
                or "\\" in scope
                or ":" in scope
                or path.is_absolute()
                or path.as_posix() != scope
                or any(part in {".", ".."} for part in path.parts)
            ):
                raise ValueError(
                    "implementation_scope entries must be canonical relative paths without prose"
                )
        if len(set(self.implementation_scope)) != len(self.implementation_scope):
            raise ValueError("implementation_scope entries must be unique")
        return self


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


class ExperimentProposalDecision(ContractModel):
    """Research response required when the controller requests an experiment."""

    request_id: str
    kind: Literal["proposal"]
    experiment_spec: ExperimentSpec
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

    @model_validator(mode="after")
    def validate_action_target(self) -> OrchestrationDecision:
        if self.action == DecisionAction.RESEARCH and self.target_experiment_id is not None:
            raise ValueError("research decisions must not select an experiment identity")
        return self


class ImplementationEdit(ContractModel):
    """One bounded file replacement requested by the implementor."""

    relative_path: str
    content: str


class ImplementationResult(ContractModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    patch_artifact_id: str
    changed_files: tuple[str, ...]
    edits: tuple[ImplementationEdit, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_issues: tuple[str, ...] = ()


class ImplementationSubmission(ContractModel):
    """Production implementor response requiring at least one bounded edit."""

    schema_version: Literal["1"] = "1"
    experiment_id: str
    patch_artifact_id: str
    changed_files: tuple[str, ...]
    edits: Annotated[tuple[ImplementationEdit, ...], Field(min_length=1)]
    changed_symbols: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_issues: tuple[str, ...] = ()


class ImplementationAttemptRecord(ContractModel):
    """Immutable implementor output for one bounded repair attempt."""

    experiment_id: str
    repair_attempt: Annotated[int, Field(ge=0, le=3)]
    result: ImplementationResult

    @model_validator(mode="after")
    def validate_experiment_identity(self) -> ImplementationAttemptRecord:
        if self.experiment_id != self.result.experiment_id:
            raise ValueError("implementation attempt/result experiment IDs do not match")
        return self


class ImplementationRequest(ContractModel):
    """Controller-owned request sent to the implementor role."""

    request_id: str
    experiment_id: str
    experiment_spec: ExperimentSpec
    allowed_scopes: tuple[str, ...]
    capabilities: tuple[str, ...] = ()
    repair_feedback: str | None = None
    source_context: dict[str, str] = {}
    base_source_context: dict[str, str] = {}
    execution_entrypoint: tuple[str, ...] = (
        "python",
        "-m",
        "tiktok2026.experiment.train",
    )
    required_changed_paths: tuple[str, ...] = ("src/tiktok2026/experiment/train.py",)


class ImplementationValidationAuthority(ContractModel):
    """Controller-computed identity available before source registration."""

    evidence_id: str
    worktree_id: str
    parent_commit: FullCommitSha
    diff_sha256: Sha256
    changed_files: tuple[str, ...]
    allowed_scopes: tuple[str, ...]
    path_policy_passed: Literal[True] = True
    execution_entrypoint: tuple[str, ...] = (
        "python",
        "-m",
        "tiktok2026.experiment.train",
    )
    required_changed_paths: tuple[str, ...] = ("src/tiktok2026/experiment/train.py",)
    source_registration_stage: Literal["post_implementation_validation"] = (
        "post_implementation_validation"
    )


class ValidationRequest(ContractModel):
    """Controller-owned request sent to the validator role."""

    request_id: str
    experiment_id: str
    stage: ValidationStage
    subject: dict[str, object] = {}


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
    run_id: str | None = None
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
    checkpoint_id: str | None = None

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


class FinalizationBundleRequest(ContractModel):
    """References that must be materialized before provisional finalization."""

    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    checkpoint_id: str
    evaluation_id: str
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


class DiagnosticMetricValue(ContractModel):
    name: Literal["GAUC", "nDCG@5", "primary"]
    value: float


class BaselineCalibrationRecord(ContractModel):
    """Immutable validation-only Starter Kit calibration."""

    schema_version: Literal["1"] = "1"
    calibration_id: str
    dataset_manifest_id: str
    dataset_manifest_sha256: Sha256
    evaluator_id: str
    evaluator_sha256: Sha256
    baseline_source_sha256: Sha256
    config_sha256: Sha256
    model: Literal["fm"] = "fm"
    seed: Literal[0] = 0
    split: Literal["valid"] = "valid"
    prediction_sha256: Sha256
    prediction_artifact_uri: str
    evaluation: EvaluationResult
    diagnostic_metrics: tuple[DiagnosticMetricValue, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_evaluation_identity(self) -> BaselineCalibrationRecord:
        result = self.evaluation
        if (
            result.evaluator_artifact_id != self.evaluator_id
            or result.evaluator_sha256 != self.evaluator_sha256
            or result.dataset_manifest_id != self.dataset_manifest_id
            or result.dataset_manifest_sha256 != self.dataset_manifest_sha256
            or result.prediction_sha256 != self.prediction_sha256
            or result.split != self.split
            or result.validity != "provisional"
        ):
            raise ValueError("baseline calibration evaluation provenance does not match")
        if {metric.name for metric in self.diagnostic_metrics} != {
            "GAUC",
            "nDCG@5",
            "primary",
        }:
            raise ValueError("baseline calibration requires all diagnostic metrics")
        return self


class FailureRecord(ContractModel):
    schema_version: Literal["1"] = "1"
    failure_id: str
    experiment_id: str | None = None
    kind: FailureKind
    evidence_refs: tuple[str, ...]
    repair_attempt: Annotated[int, Field(ge=0, le=3)]
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


class ExperimentRegistryEntry(ContractModel):
    experiment_id: str
    hypothesis_id: str
    parent_experiment_id: str | None = None
    hypothesis: str
    mechanism: str
    status: str
    evaluation_ids: tuple[str, ...] = ()
    evaluator_sha256s: tuple[Sha256, ...] = ()


class ExperimentRegistrySnapshot(ContractModel):
    evidence_id: str
    entries: tuple[ExperimentRegistryEntry, ...] = ()
    total_experiments: Annotated[int, Field(ge=0)] = 0
    complete: bool = True

    @model_validator(mode="after")
    def validate_coverage(self) -> ExperimentRegistrySnapshot:
        if self.total_experiments < len(self.entries):
            raise ValueError("experiment registry total cannot be smaller than its entries")
        if self.complete != (self.total_experiments == len(self.entries)):
            raise ValueError("experiment registry completeness does not match its entries")
        return self


class ExperimentExecutionContract(ContractModel):
    """Controller-owned experiment interface available to runtime agents."""

    entrypoint_path: Literal["src/tiktok2026/experiment/train.py"] = (
        "src/tiktok2026/experiment/train.py"
    )
    command_module: Literal["tiktok2026.experiment.train"] = "tiktok2026.experiment.train"
    required_arguments: tuple[str, ...] = (
        "--output-dir",
        "--seed",
        "--fidelity",
        "--data-manifest",
        "--source-commit",
        "--execution-id",
        "--data-root",
    )
    available_splits: tuple[Literal["train", "valid"], ...] = ("train", "valid")
    prediction_split: Literal["valid"] = "valid"
    prediction_rows: Literal["exact_valid_manifest_rows_in_manifest_order"] = (
        "exact_valid_manifest_rows_in_manifest_order"
    )
    prediction_fields: tuple[str, ...] = (
        "row_id",
        "row_identity",
        "user_id",
        "item_id",
        "score",
    )
    required_artifacts: tuple[Literal["predictions.json", "checkpoint_bundle.json"], ...] = (
        "predictions.json",
        "checkpoint_bundle.json",
    )
    output_visibility: Literal["private_until_controller_validation"] = (
        "private_until_controller_validation"
    )
    artifact_publication_owner: Literal["controller"] = "controller"
    separate_candidate_input: Literal[False] = False
    valid_labels_may_influence_scores: Literal[False] = False


class ControllerContext(ContractModel):
    """Controller-owned facts agents may reference but must not redefine."""

    schema_version: Literal["1"] = "1"
    dataset_manifest_identity: DatasetManifestIdentity | None = None
    evaluator_identity: EvaluatorIdentity | None = None
    judging_metrics: tuple[Literal["NDCG@10", "Recall@50"], ...] = (
        "NDCG@10",
        "Recall@50",
    )
    metric_and_candidate_semantics_owner: Literal["controller"] = "controller"
    source_commit_stage: Literal["post_implementation"] = "post_implementation"
    dataset_staging_owner: Literal["controller"] = "controller"
    execution_sandbox_owner: Literal["controller"] = "controller"
    test_access_owner: Literal["controller"] = "controller"
    parent_commit: FullCommitSha | None = None
    docker_image: str | None = None
    experiment_registry: ExperimentRegistrySnapshot | None = None
    experiment_execution: ExperimentExecutionContract = ExperimentExecutionContract()


class ResearchRequest(ContractModel):
    request_id: str
    objective: str
    resource_state: ResourceState
    parent_experiment_id: str | None = None
    allowed_paths: tuple[str, ...] = ("src/tiktok2026/experiment",)
    controller_context: ControllerContext | None = None


class OrchestrationRequest(ContractModel):
    """Controller-owned feasible choices and bounded lifecycle evidence."""

    request_id: str
    run_id: str
    phase: RunPhase
    allowed_actions: Annotated[tuple[DecisionAction, ...], Field(min_length=1)]
    resource_state: ResourceState
    current_experiment_id: str | None = None
    latest_evaluation_result_id: str | None = None
    finalization_ready: bool = False
    failure_summary: str | None = None
    controller_context: ControllerContext | None = None


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


class DatasetManifestIdentity(ContractModel):
    manifest_id: str
    manifest_sha256: Sha256


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
    repair_attempts: Annotated[int, Field(ge=0, le=3)] = 0
    terminal_reason: str | None = None


class OperationResult(ContractModel):
    """Typed result returned by bootstrap-owned operator operations."""

    operation: str
    run_id: str | None = None
    phase: RunPhase | None = None
    status: str
    values: dict[str, object] = Field(default_factory=dict)


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


class ProvenanceRequest(ContractModel):
    """Provenance metadata carried alongside an evaluation persistence."""

    run_id: str
    experiment_id: str
    source_commit: FullCommitSha
    execution_id: str
    dataset_manifest_id: str
    dataset_manifest_sha256: Sha256
    evaluator_id: str
    evaluator_sha256: Sha256


class PolicyDecisionModel(ContractModel):
    """Serializable version of a pure policy decision."""

    allowed: bool
    reason: str


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
