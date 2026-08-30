from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
FullCommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


def _populate_source_identity(value: object, field: str) -> object:
    if not isinstance(value, dict):
        return value
    data = dict(cast(dict[str, object], value))
    if field not in data:
        source_commit = data.get("source_commit")
        if isinstance(source_commit, str):
            data[field] = f"source-{source_commit}"
    return data


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


class ImplementationCriterionId(StrEnum):
    """Stable IDs for independently assessable implementation checks.

    The first five values are the original broad criteria.  The remaining values
    split out recurring contract failures so that a repair can target one
    independently verifiable concern without changing the older spellings.
    """

    SCIENTIFIC_FIDELITY = "scientific_fidelity"
    CHANGED_PATH_SCOPE = "changed_path_scope"
    LEAKAGE = "leakage"
    UNRELATED_CHANGES = "unrelated_changes"
    EXECUTION_WIRING = "execution_wiring"
    STATIC_CHECKS = "static_checks"
    CLI_ARTIFACT_CONTRACT = "cli_artifact_contract"
    PROVENANCE = "provenance"
    STRICT_JSON_TYPES = "strict_json_types"
    ROW_COVERAGE_ORDER = "row_coverage_order"
    DETERMINISTIC_RANKING_TIE_POLICY = "deterministic_ranking_tie_policy"
    EXPERIMENT_SPECIFIC_RECONSTRUCTION = "experiment_specific_reconstruction"
    RESOURCE_FEASIBILITY = "resource_feasibility"

    # These spellings are useful to older clients without adding new criteria.
    SCOPE = "changed_path_scope"
    PATH_SCOPE = "changed_path_scope"
    LEAKAGE_SAFETY = "leakage"
    LEAKAGE_FREE = "leakage"
    EXECUTION_ENTRYPOINT = "execution_wiring"
    ENTRYPOINT_WIRING = "execution_wiring"


ImplementationCriterionType = ImplementationCriterionId

# This is intentionally bounded and controller-owned.  Requests may narrow the
# set only when the controller has an explicit reason; an omitted field must not
# silently produce an unvalidated implementation.
DEFAULT_IMPLEMENTATION_CRITERIA: tuple[ImplementationCriterionId, ...] = tuple(
    ImplementationCriterionId(member.value) for member in ImplementationCriterionId
)
IMPLEMENTATION_CRITERIA = DEFAULT_IMPLEMENTATION_CRITERIA


class CriterionAssessmentStatus(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


ImplementationCriterionStatus = CriterionAssessmentStatus


def validation_blocker_id(
    report_id: str,
    stage: ValidationStage,
    text: str,
    criterion_id: ImplementationCriterionId | str | None = None,
    experiment_id: str | None = None,
) -> str:
    """Return a stable blocker identity, using the legacy text form when needed.

    Reports produced before criterion identities existed remain addressable by their
    report/stage/text identity.  New criterion-bearing blockers intentionally omit
    the report and text so that the same failed criterion remains addressable across
    validation reports.
    """
    if criterion_id is not None:
        material = json.dumps(
            {
                "experiment_id": experiment_id or report_id,
                "stage": stage.value,
                "criterion_id": str(criterion_id),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        material = json.dumps(
            {"report_id": report_id, "stage": stage.value, "text": text},
            sort_keys=True,
            separators=(",", ":"),
        )
    return f"blocker-{hashlib.sha256(material.encode()).hexdigest()}"


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


class ImplementationResourceEstimate(ContractModel):
    """Technique-neutral resource estimates supplied with an implementation proposal."""

    predicted_wall_seconds: Annotated[float, Field(ge=0.0)]
    predicted_peak_memory_bytes: Annotated[int, Field(ge=0)]
    predicted_artifact_bytes: Annotated[int, Field(ge=0)]
    dataset_passes: Annotated[int, Field(ge=0, le=16)]
    high_cardinality_nested_scans: bool = False
    duplicate_full_materializations: bool = False


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
    implementation_resource_estimate: ImplementationResourceEstimate | None = None
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

    @model_validator(mode="after")
    def validate_proposal(self) -> ResearchDecision:
        if self.kind == "proposal":
            if self.experiment_spec is None:
                raise ValueError("proposal decisions require experiment_spec")
            if self.experiment_spec.implementation_resource_estimate is None:
                raise ValueError("proposal decisions require implementation_resource_estimate")
        return self


class ExperimentProposalDecision(ContractModel):
    """Research response required when the controller requests an experiment.

    Unlike standalone ``ExperimentSpec`` records, a newly proposed experiment
    must include a resource estimate so the controller can assess feasibility
    before accepting it.
    """

    request_id: str
    kind: Literal["proposal"]
    experiment_spec: ExperimentSpec
    message: str
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_resource_estimate(self) -> ExperimentProposalDecision:
        if self.experiment_spec.implementation_resource_estimate is None:
            raise ValueError("experiment proposals require implementation_resource_estimate")
        return self


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


class ImplementationCriterion(ContractModel):
    """One controller-selected atomic implementation requirement."""

    criterion_id: ImplementationCriterionId
    description: str = ""


class ImplementationCriterionAssessment(ContractModel):
    """Typed validator assessment of one requested implementation criterion."""

    criterion_id: ImplementationCriterionId
    status: CriterionAssessmentStatus
    evidence_refs: tuple[str, ...] = ()
    details: str = ""


class CriterionResolutionClaim(ContractModel):
    """A possibly partial, evidence-backed resolution of a criterion's blockers."""

    criterion_id: ImplementationCriterionId
    status: CriterionAssessmentStatus
    blocker_ids: tuple[str, ...] = ()
    # Singular spelling keeps hand-authored/early agent payloads convenient.
    blocker_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    claim: str = ""

    @model_validator(mode="after")
    def normalize_blocker_ids(self) -> CriterionResolutionClaim:
        if self.status == CriterionAssessmentStatus.FAIL:
            raise ValueError("failed criteria cannot claim resolution")
        if self.status in (
            CriterionAssessmentStatus.PASS,
            CriterionAssessmentStatus.PARTIAL,
        ) and not self.evidence_refs:
            raise ValueError("pass or partial resolution claims require evidence_refs")
        if self.blocker_id is not None and self.blocker_id not in self.blocker_ids:
            return self.model_copy(update={"blocker_ids": (self.blocker_id, *self.blocker_ids)})
        return self


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
    """Production implementor response; edits may be metadata-only."""

    schema_version: Literal["1"] = "1"
    experiment_id: str
    patch_artifact_id: str
    changed_files: tuple[str, ...]
    edits: tuple[ImplementationEdit, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    unresolved_issues: tuple[str, ...] = ()


class ImplementationAttemptRecord(ContractModel):
    """Immutable implementor output for one bounded repair attempt."""

    experiment_id: str
    repair_attempt: Annotated[int, Field(ge=0, le=3)]
    result: ImplementationResult
    prior_diff_sha256: Sha256 | None = None
    result_diff_sha256: Sha256 | None = None

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
    # Write scopes and controller-derived read scopes are intentionally separate.
    read_scopes: tuple[str, ...] = ()
    implementation_criteria: tuple[ImplementationCriterionId, ...] = (
        DEFAULT_IMPLEMENTATION_CRITERIA
    )
    criterion_requirements: tuple[ImplementationCriterion, ...] = ()
    capabilities: tuple[str, ...] = ()
    repair_feedback: str | None = None
    unresolved_blocker_ids: tuple[str, ...] = ()
    unresolved_blockers: tuple[ValidationBlockerContext, ...] = ()
    source_context: dict[str, str] = {}
    base_source_context: dict[str, str] = {}
    execution_entrypoint: tuple[str, ...] = (
        "python",
        "-m",
        "tiktok2026.experiment.train",
    )
    required_changed_paths: tuple[str, ...] = ("src/tiktok2026/experiment/train.py",)
    execution_timeout_seconds: Annotated[int, Field(gt=0)] = 300
    execution_memory_bytes: Annotated[int, Field(gt=0)] = 4 * 1024**3
    execution_cpus: Annotated[float, Field(gt=0)] = 1.0
    execution_gpu_count: Annotated[int, Field(ge=0)] = 0
    prior_diff_sha256: Sha256 | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_read_scope_alias(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(cast(dict[str, object], value))
        if "read_scopes" not in data and "controller_read_scopes" in data:
            data["read_scopes"] = data.pop("controller_read_scopes")
        return data

    @property
    def controller_read_scopes(self) -> tuple[str, ...]:
        return self.read_scopes

    @property
    def required_implementation_criteria(self) -> tuple[ImplementationCriterionId, ...]:
        return self.implementation_criteria


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
    validation_operation: ValidationOperationIdentity
    subject: dict[str, object] = {}


class ValidationBlockerContext(ContractModel):
    """Bound context copied from authority for a repair or validation request."""

    blocker_id: str
    text: Annotated[str, Field(max_length=2_000)]
    criterion_id: ImplementationCriterionId | None = None
    evidence_refs: Annotated[
        tuple[Annotated[str, Field(max_length=256)], ...], Field(max_length=8)
    ] = ()


class ValidationOperationIdentity(ContractModel):
    """Controller-derived identity for one immutable validation operation."""

    operation_id: str
    run_id: str
    experiment_id: str
    stage: ValidationStage
    repair_attempt: Annotated[int, Field(ge=0, le=3)]
    subject_sha256: Sha256
    implementation_diff_sha256: Sha256 | None = None

class ValidationBlocker(ContractModel):
    """A durable, independently addressable validation failure."""

    blocker_id: str
    experiment_id: str
    stage: ValidationStage
    text: str
    report_id: str = ""
    evidence_refs: tuple[str, ...] = ()
    criterion_id: ImplementationCriterionId | None = None
    _supplied_blocker_id: str | None = PrivateAttr(default=None)

    def __init__(self, **data: object) -> None:
        supplied_id = data.get("blocker_id")
        super().__init__(**data)
        if isinstance(supplied_id, str):
            object.__setattr__(self, "_supplied_blocker_id", supplied_id)

    @property
    def supplied_blocker_id(self) -> str | None:
        """Return an input ID retained for report safety checks."""
        return self._supplied_blocker_id

    @model_validator(mode="before")
    @classmethod
    def normalize_criterion_identity(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(cast(dict[str, object], value))
        criterion = data.get("criterion_id")
        if criterion is None:
            return data
        raw_stage = data.get("stage", ValidationStage.RESULT)
        stage = raw_stage if isinstance(raw_stage, ValidationStage) else ValidationStage(raw_stage)
        data["blocker_id"] = validation_blocker_id(
            str(data.get("report_id", "")),
            stage,
            str(data.get("text", data.get("message", ""))),
            criterion_id=str(criterion),
            experiment_id=str(data.get("experiment_id", "")),
        )
        return data

    def __str__(self) -> str:
        # This keeps diagnostics produced by older callers readable while the
        # authoritative representation is now typed.
        return self.text

    def __contains__(self, value: str) -> bool:
        return value in self.text

    @property
    def message(self) -> str:
        """Compatibility spelling for consumers that called blockers messages."""
        return self.text


class BlockerResolution(ContractModel):
    """Immutable evidence-backed resolution of one validation blocker."""

    resolution_id: str
    blocker_id: str
    report_id: str
    experiment_id: str
    evidence_refs: Annotated[tuple[str, ...], Field(min_length=1)]
    validation_operation_id: str = ""
    criterion_id: ImplementationCriterionId | None = None
    status: CriterionAssessmentStatus | None = None


class ValidationReport(ContractModel):
    schema_version: Literal["1"] = "1"
    report_id: str
    experiment_id: str
    stage: ValidationStage
    verdict: ValidationVerdict
    blockers: tuple[ValidationBlocker, ...] = ()
    criterion_assessments: tuple[ImplementationCriterionAssessment, ...] = ()
    resolution_claims: tuple[CriterionResolutionClaim, ...] = ()
    resolves_blocker_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    leakage_risk: str
    implementation_fidelity: str | None = None
    scientific_confidence: str | None = None
    validation_operation_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_blockers(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(cast(dict[str, object], value))
        if "resolution_claims" not in data and "criterion_resolution_claims" in data:
            data["resolution_claims"] = data.pop("criterion_resolution_claims")
        report_id = str(data.get("report_id", ""))
        experiment_id = str(data.get("experiment_id", ""))
        report_evidence = tuple(cast(tuple[str, ...] | list[str], data.get("evidence_refs", ())))
        raw_stage = data.get("stage", ValidationStage.RESULT)
        stage = raw_stage if isinstance(raw_stage, ValidationStage) else ValidationStage(raw_stage)
        resolution_ids = set(
            cast(tuple[str, ...] | list[str], data.get("resolves_blocker_ids", ()))
        )
        raw_claims = data.get("resolution_claims", ())
        for raw_claim in cast(tuple[object, ...] | list[object], raw_claims):
            if isinstance(raw_claim, CriterionResolutionClaim):
                resolution_ids.update(raw_claim.blocker_ids)
            elif isinstance(raw_claim, dict):
                claim = cast(dict[str, object], raw_claim)
                resolution_ids.update(
                    cast(tuple[str, ...] | list[str], claim.get("blocker_ids", ()))
                )
                blocker_id = claim.get("blocker_id")
                if isinstance(blocker_id, str):
                    resolution_ids.add(blocker_id)
        normalized: list[object] = []
        supplied_ids: set[str] = set()
        for raw in cast(tuple[object, ...] | list[object], data.get("blockers", ())):
            if isinstance(raw, str):
                normalized.append(
                    {
                        "blocker_id": validation_blocker_id(report_id, stage, raw),
                        "experiment_id": experiment_id,
                        "stage": stage,
                        "text": raw,
                        "report_id": report_id,
                        "evidence_refs": report_evidence,
                    }
                )
            elif isinstance(raw, ValidationBlocker):
                supplied_id = raw.supplied_blocker_id
                if supplied_id is not None:
                    supplied_ids.add(supplied_id)
                blocker = raw.model_dump(mode="json")
                blocker["report_id"] = blocker.get("report_id") or report_id
                blocker["evidence_refs"] = blocker.get("evidence_refs") or report_evidence
                criterion = blocker.get("criterion_id")
                if criterion is not None:
                    blocker["blocker_id"] = validation_blocker_id(
                        report_id,
                        stage,
                        str(blocker.get("text", "")),
                        criterion_id=str(criterion),
                        experiment_id=experiment_id,
                    )
                normalized.append(blocker)
            elif isinstance(raw, dict):
                blocker = dict(cast(dict[str, object], raw))
                supplied_id = blocker.get("blocker_id")
                if isinstance(supplied_id, str):
                    supplied_ids.add(supplied_id)
                text = str(blocker.get("text", blocker.get("message", "")))
                blocker["text"] = text
                blocker.setdefault("experiment_id", experiment_id)
                blocker.setdefault("stage", stage)
                blocker["report_id"] = blocker.get("report_id") or report_id
                blocker["evidence_refs"] = blocker.get("evidence_refs") or report_evidence
                criterion = blocker.get("criterion_id")
                if criterion is not None:
                    # Criterion identity is authoritative, but the supplied ID
                    # still participates in the introduced-blocker safety check.
                    blocker["blocker_id"] = validation_blocker_id(
                        report_id,
                        stage,
                        text,
                        criterion_id=str(criterion),
                        experiment_id=experiment_id,
                    )
                else:
                    blocker.setdefault("blocker_id", validation_blocker_id(report_id, stage, text))
                normalized.append(blocker)
            else:
                normalized.append(raw)
        if supplied_ids & resolution_ids:
            raise ValueError("a validation report cannot resolve a blocker it introduces")
        data["blockers"] = tuple(normalized)
        return data

    @model_validator(mode="after")
    def validate_blockers(self) -> ValidationReport:
        criterion_ids = tuple(
            assessment.criterion_id for assessment in self.criterion_assessments
        )
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("criterion_assessments must contain unique criterion IDs")
        if len(set(self.resolves_blocker_ids)) != len(self.resolves_blocker_ids):
            raise ValueError("resolves_blocker_ids must be unique")
        if self.resolves_blocker_ids and self.verdict != ValidationVerdict.APPROVED:
            raise ValueError("only approved validation reports may resolve blockers")
        for blocker in self.blockers:
            if (
                blocker.report_id != self.report_id
                or blocker.experiment_id != self.experiment_id
                or blocker.stage != self.stage
            ):
                raise ValueError("validation blocker identity does not match its report")
        claimed_ids = tuple(
            blocker_id
            for claim in self.resolution_claims
            for blocker_id in claim.blocker_ids
        )
        all_resolved_ids = (*self.resolves_blocker_ids, *claimed_ids)
        if len(set(all_resolved_ids)) != len(all_resolved_ids):
            raise ValueError("resolution blocker IDs must be unique")
        if self.resolution_claims and self.verdict not in (
            ValidationVerdict.APPROVED,
            ValidationVerdict.REPAIRABLE,
        ):
            raise ValueError("only approved or repairable reports may claim resolutions")
        if set(all_resolved_ids) & {blocker.blocker_id for blocker in self.blockers}:
            raise ValueError("a validation report cannot resolve a blocker it introduces")
        return self

    @property
    def criterion_resolution_claims(self) -> tuple[CriterionResolutionClaim, ...]:
        return self.resolution_claims


class DatasetViewRow(ContractModel):
    row_id: str
    row_identity: tuple[str, ...]
    user_id: str
    item_id: str


class DatasetViewProvenance(ContractModel):
    manifest_id: str
    manifest_sha256: Sha256
    view_sha256: Sha256
    valid_rows: tuple[DatasetViewRow, ...]


class ExecutionRequest(ContractModel):
    run_id: str | None = None
    execution_id: str
    experiment_id: str
    source_registration_id: str = ""
    source_commit: CommitSha
    command: tuple[str, ...]
    image: str
    source_path: Path
    dataset_path: Path
    dataset_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    output_path: Path
    timeout_seconds: Annotated[int, Field(gt=0)]
    memory_bytes: Annotated[int, Field(gt=0)]
    cpus: Annotated[float, Field(gt=0)]
    gpu_count: Annotated[int, Field(ge=0)] = 0
    execution_kind: Literal["smoke", "full"] = "full"
    dataset_view_sha256: Sha256 | None = None

    @model_validator(mode="before")
    @classmethod
    def populate_source_registration_id(cls, value: object) -> object:
        return _populate_source_identity(value, "source_registration_id")

    @model_validator(mode="after")
    def validate_source_registration_id(self) -> ExecutionRequest:
        if self.source_registration_id != f"source-{self.source_commit}":
            raise ValueError("source registration identity does not match source commit")
        return self


class ExecutionResult(ContractModel):
    schema_version: Literal["1"] = "1"
    execution_id: str
    experiment_id: str
    source_registration_id: str = ""
    source_commit: str
    command: tuple[str, ...]
    exit_code: int
    elapsed_seconds: Annotated[float, Field(ge=0.0)]
    gpu_hours: Annotated[float, Field(ge=0.0)]
    artifact_output_bytes: Annotated[int, Field(ge=0)] = 0
    artifact_ids: tuple[str, ...] = ()
    failure_kind: FailureKind | None = None
    failure_message: str | None = None
    checkpoint_id: str | None = None
    execution_kind: Literal["smoke", "full"] = "full"
    dataset_manifest_id: str | None = None
    dataset_manifest_sha256: Sha256 | None = None
    dataset_view_sha256: Sha256 | None = None
    dataset_valid_rows: tuple[DatasetViewRow, ...] = ()
    measured_peak_memory_bytes: Annotated[int, Field(ge=0)] | None = None
    memory_measurement_status: Literal["measured", "unavailable"] = "unavailable"
    resource_measurement_basis: Literal["docker_stats", "unavailable"] = "unavailable"
    measured_gpu_hours: Annotated[float, Field(ge=0.0)] | None = None
    gpu_telemetry_status: Literal["measured", "unavailable", "not_requested"] = (
        "not_requested"
    )
    smoke_output_valid: bool = False
    scientific_evidence: bool = True

    @model_validator(mode="before")
    @classmethod
    def populate_source_registration_id(cls, value: object) -> object:
        return _populate_source_identity(value, "source_registration_id")

    @model_validator(mode="after")
    def validate_failure(self) -> ExecutionResult:
        if self.source_registration_id != f"source-{self.source_commit}":
            raise ValueError("source registration identity does not match source commit")
        if self.exit_code == 0 and self.failure_kind is not None:
            raise ValueError("successful execution cannot have a failure kind")
        if self.exit_code != 0 and self.failure_kind is None:
            raise ValueError("failed execution requires a failure kind")
        if self.execution_kind == "smoke" and self.scientific_evidence:
            raise ValueError("smoke execution cannot be scientific evidence")
        if self.execution_kind == "smoke" and (
            self.dataset_manifest_id is None
            or self.dataset_manifest_sha256 is None
            or self.dataset_view_sha256 is None
        ):
            raise ValueError("smoke execution requires dataset provenance")
        if self.memory_measurement_status == "measured":
            if self.measured_peak_memory_bytes is None:
                raise ValueError("measured memory status requires a peak memory value")
            if self.resource_measurement_basis != "docker_stats":
                raise ValueError("measured memory requires Docker stats basis")
        elif self.measured_peak_memory_bytes is not None:
            raise ValueError("unavailable memory status cannot carry a measurement")
        if self.gpu_telemetry_status == "measured":
            if self.measured_gpu_hours is None:
                raise ValueError("measured GPU status requires a GPU measurement")
        elif self.measured_gpu_hours is not None:
            raise ValueError("unavailable GPU telemetry cannot carry a measurement")
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
    dataset_view_sha256: Sha256 | None = None
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


class PredictionRowRequirement(ContractModel):
    """Schema and ordering requirements for rows in the prediction envelope."""

    row_type: Literal["prediction"] = "prediction"
    required_fields: tuple[
        Literal["row_id", "row_identity", "user_id", "item_id", "score"], ...
    ] = ("row_id", "row_identity", "user_id", "item_id", "score")
    order: Literal["exact_valid_manifest_rows_in_manifest_order"] = (
        "exact_valid_manifest_rows_in_manifest_order"
    )


class CheckpointRowRequirement(ContractModel):
    """Required metadata fields for the single checkpoint bundle envelope.

    ``dataset_view_sha256`` is a required envelope key.  Its value is nullable
    for legacy/non-smoke records; smoke execution requires a non-null digest.
    """

    row_type: Literal["checkpoint"] = "checkpoint"
    required_fields: tuple[
        Literal[
            "checkpoint_id",
            "data_manifest_id",
            "seed",
            "source_commit",
            "execution_id",
            "fidelity",
            "prediction_artifact_id",
            "prediction_artifact",
            "prediction_sha256",
            "dataset_view_sha256",
        ],
        ...,
    ] = (
        "checkpoint_id",
        "data_manifest_id",
        "seed",
        "source_commit",
        "execution_id",
        "fidelity",
        "prediction_artifact_id",
        "prediction_artifact",
        "prediction_sha256",
        "dataset_view_sha256",
    )


class PredictionArtifactEnvelope(ContractModel):
    """Typed requirements for the JSON prediction artifact envelope.

    ``dataset_view_sha256`` is a required envelope key.  Its value is nullable
    for legacy/non-smoke records; smoke execution requires a non-null digest.
    """

    artifact_name: Literal["predictions.json"] = "predictions.json"
    schema_version: Literal["1"] = "1"
    required_fields: tuple[
        Literal[
            "schema_version",
            "manifest_id",
            "manifest_sha256",
            "dataset_view_sha256",
            "source_commit",
            "execution_id",
            "split",
            "rows",
        ],
        ...,
    ] = (
        "schema_version",
        "manifest_id",
        "manifest_sha256",
        "dataset_view_sha256",
        "source_commit",
        "execution_id",
        "split",
        "rows",
    )
    row_requirements: PredictionRowRequirement = PredictionRowRequirement()


class CheckpointArtifactEnvelope(ContractModel):
    """Typed requirements for the JSON checkpoint bundle envelope.

    ``dataset_view_sha256`` is a required envelope key.  Its value is nullable
    for legacy/non-smoke records; smoke execution requires a non-null digest.
    """

    artifact_name: Literal["checkpoint_bundle.json"] = "checkpoint_bundle.json"
    schema_version: Literal["1"] = "1"
    required_fields: tuple[
        Literal[
            "schema_version",
            "checkpoint_id",
            "data_manifest_id",
            "seed",
            "source_commit",
            "execution_id",
            "fidelity",
            "prediction_artifact_id",
            "prediction_artifact",
            "prediction_sha256",
            "dataset_view_sha256",
        ],
        ...,
    ] = (
        "schema_version",
        "checkpoint_id",
        "data_manifest_id",
        "seed",
        "source_commit",
        "execution_id",
        "fidelity",
        "prediction_artifact_id",
        "prediction_artifact",
        "prediction_sha256",
        "dataset_view_sha256",
    )
    row_requirements: CheckpointRowRequirement = CheckpointRowRequirement()


# Requirement is the more explicit spelling used by newer callers.
PredictionArtifactRequirement = PredictionArtifactEnvelope
CheckpointArtifactRequirement = CheckpointArtifactEnvelope


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
        "--dataset-manifest-sha256",
        "--data-root",
    )
    required_dataset_hash_arguments: tuple[
        Literal["--dataset-manifest-sha256"], ...
    ] = ("--dataset-manifest-sha256",)
    optional_dataset_hash_arguments: tuple[
        Literal["--dataset-view-sha256"], ...
    ] = ("--dataset-view-sha256",)
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
    prediction_artifact_envelope: PredictionArtifactEnvelope = PredictionArtifactEnvelope()
    prediction_row_requirements: PredictionRowRequirement = PredictionRowRequirement()
    checkpoint_artifact_envelope: CheckpointArtifactEnvelope = CheckpointArtifactEnvelope()
    checkpoint_row_requirements: CheckpointRowRequirement = CheckpointRowRequirement()
    output_visibility: Literal["private_until_controller_validation"] = (
        "private_until_controller_validation"
    )
    artifact_publication_owner: Literal["controller"] = "controller"
    separate_candidate_input: Literal[False] = False
    valid_labels_may_influence_scores: Literal[False] = False

    @property
    def required_injected_dataset_hash_arguments(self) -> tuple[str, ...]:
        return self.required_dataset_hash_arguments

    @property
    def optional_injected_dataset_hash_arguments(self) -> tuple[str, ...]:
        return self.optional_dataset_hash_arguments


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
    unresolved_blockers: tuple[ValidationBlockerContext, ...] = ()


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
    registration_id: str = ""
    revision: Annotated[int, Field(ge=0)] = 0
    experiment_id: str
    run_id: str
    parent_commit: FullCommitSha
    source_commit: FullCommitSha
    patch_sha256: Sha256
    patch_artifact_id: str
    patch_artifact_uri: str
    allowed_scopes: tuple[str, ...]
    eligible: bool = False

    @model_validator(mode="before")
    @classmethod
    def populate_registration_id(cls, value: object) -> object:
        return _populate_source_identity(value, "registration_id")

    @model_validator(mode="after")
    def validate_registration_id(self) -> SourceRegistration:
        if self.registration_id != f"source-{self.source_commit}":
            raise ValueError("source registration identity does not match source commit")
        return self


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
