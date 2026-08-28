from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


class MetricValue(ContractModel):
    name: Literal["NDCG@10", "Recall@50"]
    value: Annotated[float, Field(ge=0.0, le=1.0)]


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


class EvaluationResult(ContractModel):
    schema_version: Literal["1"] = "1"
    evaluation_id: str
    experiment_id: str
    checkpoint_id: str
    metrics: tuple[MetricValue, ...]
    evaluator_artifact_id: str
    evaluator_sha256: str
    prediction_sha256: str
    validity: Literal["provisional", "official", "invalid"]

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
