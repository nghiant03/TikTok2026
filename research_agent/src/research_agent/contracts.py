from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_agent.shared_contracts import (
    ExecutionResult,
    ExperimentSpec,
    FailureKind,
    ResourceState,
)

NonEmptyText = Annotated[str, Field(min_length=1)]
NonEmptyStrings = Annotated[tuple[str, ...], Field(min_length=1)]
OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID = (
    "evaluation-protocol:kuairand-pure:v1"
)


class ResearchContractModel(BaseModel):
    """Immutable, strict base model matching the shared contract conventions."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricName(StrEnum):
    GAUC = "GAUC"
    NDCG_5 = "nDCG@5"


class ResearchMetricValue(ResearchContractModel):
    name: MetricName
    value: Annotated[float, Field(ge=0.0, le=1.0)]


class ResearchEvaluationResult(ResearchContractModel):
    """Latest-benchmark evaluation result kept local until shared contracts are updated."""

    schema_version: Literal["1"] = "1"
    evaluation_id: NonEmptyText
    experiment_id: NonEmptyText
    checkpoint_id: NonEmptyText
    metrics: tuple[ResearchMetricValue, ResearchMetricValue]
    evaluator_artifact_id: NonEmptyText
    evaluator_sha256: NonEmptyText
    prediction_sha256: NonEmptyText
    validity: Literal["provisional", "official", "invalid"]

    @model_validator(mode="after")
    def validate_metrics(self) -> ResearchEvaluationResult:
        names = tuple(metric.name for metric in self.metrics)
        if names != (MetricName.GAUC, MetricName.NDCG_5):
            raise ValueError("metrics must be ordered as GAUC, nDCG@5")
        return self

    @property
    def validation_score(self) -> float:
        return (self.metrics[0].value + self.metrics[1].value) / 2.0


class BenchmarkContract(ResearchContractModel):
    """Problem-statement rules that the Research Agent cannot redefine."""

    schema_version: Literal["1"] = "1"
    benchmark_id: Literal["kuairand-pure"] = "kuairand-pure"
    dataset: Literal["KuaiRand-Pure"] = "KuaiRand-Pure"
    task: Literal["within-user ranking over logged impressions"] = (
        "within-user ranking over logged impressions"
    )
    positive_label: Literal["long_view"] = "long_view"
    metrics: tuple[MetricName, MetricName] = (
        MetricName.GAUC,
        MetricName.NDCG_5,
    )
    validation_ranking: Literal["mean(GAUC,nDCG@5)"] = "mean(GAUC,nDCG@5)"
    train_date_range: tuple[Literal[20220408], Literal[20220421]] = (
        20220408,
        20220421,
    )
    validation_date_range: tuple[Literal[20220422], Literal[20220428]] = (
        20220422,
        20220428,
    )
    public_holdout_date_range: tuple[Literal[20220429], Literal[20220508]] = (
        20220429,
        20220508,
    )
    public_holdout_development_allowed: Literal[False] = False
    organizer_hidden_test_locally_available: Literal[False] = False
    evaluator_sha256: Literal[
        "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"
    ] = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"
    data_loader_sha256: Literal[
        "1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541"
    ] = "1bf54f5f3a9f590eab2f87f09a3c27422031867a20a5328d56cbd8c7db36e541"
    baseline_scores_sha256: Literal[
        "950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324"
    ] = "950f98181770c030a68bdddab7be3c0abbf060531f54455a6a6f81a4cb003324"
    official_baseline_validation_gauc: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6674
    official_baseline_validation_ndcg_5: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5357
    official_baseline_validation_primary: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6016
    convergence_epsilon: Annotated[float, Field(gt=0.0)] = 0.002
    convergence_consecutive_iterations: Literal[3] = 3
    maximum_iterations: Literal[50] = 50
    maximum_wall_seconds: Literal[21600] = 21600
    development_splits: tuple[Literal["train"], Literal["validation"]] = (
        "train",
        "validation",
    )
    test_labels_accessible: Literal[False] = False
    external_training_data_allowed: Literal[False] = False
    public_literature_allowed: Literal[True] = True
    compliant_pretrained_weights_allowed: Literal[False] = False
    test_label_trained_weights_allowed: Literal[False] = False

    @model_validator(mode="after")
    def validate_metric_contract(self) -> BenchmarkContract:
        if self.metrics != (MetricName.GAUC, MetricName.NDCG_5):
            raise ValueError("metrics must be ordered as GAUC, nDCG@5")
        baseline_values = (
            self.official_baseline_validation_gauc,
            self.official_baseline_validation_ndcg_5,
            self.official_baseline_validation_primary,
        )
        if baseline_values != (0.6674, 0.5357, 0.6016):
            raise ValueError("official validation baseline values are fixed")
        if self.convergence_epsilon != 0.002:
            raise ValueError("convergence_epsilon is fixed at 0.002")
        return self


class ResearchTaskType(StrEnum):
    PROPOSE_EXPERIMENT = "propose_experiment"
    INTERPRET_RESULT = "interpret_result"


class EvidenceKind(StrEnum):
    BENCHMARK = "benchmark"
    REPOSITORY = "repository"
    DATA = "data"
    HISTORY = "history"
    LITERATURE = "literature"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    VALIDATION = "validation"


class EvidenceItem(ResearchContractModel):
    evidence_id: NonEmptyText
    kind: EvidenceKind
    summary: NonEmptyText
    source_ref: NonEmptyText
    contains_test_labels: bool = False
    authorized: bool = True


class ExperimentOutcome(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    NO_CLEAR_CHANGE = "no_clear_change"
    INVALID_EXECUTION = "invalid_execution"
    INCONCLUSIVE = "inconclusive"


class EvidenceStrength(StrEnum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class BaselineStatus(StrEnum):
    MISSING = "missing"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"


class EvaluationProtocolStatus(StrEnum):
    """How strongly the train/validation split and evaluator are established."""

    UNCONFIRMED = "unconfirmed"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"


class ExperimentHistoryItem(ResearchContractModel):
    experiment_id: NonEmptyText
    normalized_signature: NonEmptyText
    summary: NonEmptyText
    outcome: ExperimentOutcome
    hypothesis_id: str | None = None
    parent_experiment_id: str | None = None
    tags: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class ResearchLesson(ResearchContractModel):
    lesson_id: NonEmptyText
    claim: NonEmptyText
    evidence_strength: EvidenceStrength
    scope: NonEmptyText
    affected_modules: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    supporting_experiment_ids: NonEmptyStrings
    evidence_refs: NonEmptyStrings


class ResearchMemoryQueryResult(ResearchContractModel):
    query: NonEmptyText
    related_experiments: tuple[ExperimentHistoryItem, ...] = ()
    experiment_lineage: tuple[ExperimentHistoryItem, ...] = ()
    retrieved_lessons: tuple[ResearchLesson, ...] = ()

    @model_validator(mode="after")
    def validate_unique_records(self) -> ResearchMemoryQueryResult:
        for name, values, id_field in (
            ("related_experiments", self.related_experiments, "experiment_id"),
            ("experiment_lineage", self.experiment_lineage, "experiment_id"),
            ("retrieved_lessons", self.retrieved_lessons, "lesson_id"),
        ):
            identifiers = [getattr(value, id_field) for value in values]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{name} must contain unique records")
        return self


class ResearchRequest(ResearchContractModel):
    """A single research assignment sent by Orchestration."""

    schema_version: Literal["1"] = "1"
    request_id: NonEmptyText
    task_type: ResearchTaskType
    objective: NonEmptyText
    benchmark: BenchmarkContract = Field(default_factory=BenchmarkContract)
    parent_experiment_id: str | None = None
    current_experiment_id: str | None = None
    execution_result: ExecutionResult | None = None
    evaluation_result: ResearchEvaluationResult | None = None
    baseline_status: BaselineStatus = BaselineStatus.MISSING
    baseline_evidence_refs: tuple[str, ...] = ()
    evaluation_protocol_status: EvaluationProtocolStatus = (
        EvaluationProtocolStatus.UNCONFIRMED
    )
    evaluation_protocol_evidence_refs: tuple[str, ...] = ()
    resource_state: ResourceState
    allowed_implementation_scope: NonEmptyStrings

    @model_validator(mode="after")
    def validate_task_inputs(self) -> ResearchRequest:
        if self.baseline_status is BaselineStatus.MISSING and self.baseline_evidence_refs:
            raise ValueError("missing baseline cannot declare baseline_evidence_refs")
        if self.baseline_status is not BaselineStatus.MISSING and not self.baseline_evidence_refs:
            raise ValueError("provisional/verified baseline requires baseline_evidence_refs")
        if (
            self.evaluation_protocol_status is EvaluationProtocolStatus.UNCONFIRMED
            and self.evaluation_protocol_evidence_refs
        ):
            raise ValueError(
                "unconfirmed evaluation protocol cannot declare "
                "evaluation_protocol_evidence_refs"
            )
        if (
            self.evaluation_protocol_status is not EvaluationProtocolStatus.UNCONFIRMED
            and not self.evaluation_protocol_evidence_refs
        ):
            raise ValueError(
                "provisional/confirmed evaluation protocol requires "
                "evaluation_protocol_evidence_refs"
            )
        if self.task_type is ResearchTaskType.INTERPRET_RESULT:
            if self.current_experiment_id is None:
                raise ValueError("interpret_result requires current_experiment_id")
            if self.execution_result is None and self.evaluation_result is None:
                raise ValueError(
                    "interpret_result requires execution_result or evaluation_result"
                )
            if self.execution_result is not None:
                if self.execution_result.experiment_id != self.current_experiment_id:
                    raise ValueError("execution_result must belong to current_experiment_id")
                if self.execution_result.exit_code != 0 and self.evaluation_result is not None:
                    raise ValueError("failed execution cannot have an evaluation_result")
            if self.evaluation_result is not None:
                if self.evaluation_result.experiment_id != self.current_experiment_id:
                    raise ValueError("evaluation_result must belong to current_experiment_id")
                metric_names = tuple(metric.name for metric in self.evaluation_result.metrics)
                if metric_names != (MetricName.GAUC, MetricName.NDCG_5):
                    raise ValueError("evaluation_result requires exactly GAUC and nDCG@5")
        elif self.execution_result is not None or self.evaluation_result is not None:
            raise ValueError("propose_experiment must not include execution or evaluation results")
        return self


class ResearchContext(ResearchContractModel):
    """Bounded, traceable context constructed from read-only capabilities."""

    schema_version: Literal["1"] = "1"
    request: ResearchRequest
    evidence: tuple[EvidenceItem, ...]
    experiment_history: tuple[ExperimentHistoryItem, ...] = ()
    experiment_lineage: tuple[ExperimentHistoryItem, ...] = ()
    retrieved_lessons: tuple[ResearchLesson, ...] = ()
    max_evidence_items: Annotated[int, Field(ge=1, le=64)] = 24

    @model_validator(mode="after")
    def validate_context_boundary(self) -> ResearchContext:
        if len(self.evidence) > self.max_evidence_items:
            raise ValueError("research context exceeds its evidence bound")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")
        forbidden = [
            item.evidence_id
            for item in self.evidence
            if item.contains_test_labels or not item.authorized
        ]
        if forbidden:
            raise ValueError(f"forbidden evidence in context: {forbidden}")
        missing_baseline_evidence = sorted(
            set(self.request.baseline_evidence_refs) - set(evidence_ids)
        )
        if missing_baseline_evidence:
            raise ValueError(
                f"baseline evidence is absent from research context: {missing_baseline_evidence}"
            )
        missing_protocol_evidence = sorted(
            set(self.request.evaluation_protocol_evidence_refs) - set(evidence_ids)
        )
        if missing_protocol_evidence:
            raise ValueError(
                "evaluation protocol evidence is absent from research context: "
                f"{missing_protocol_evidence}"
            )
        return self

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.evidence)


class EvidenceRequestCategory(StrEnum):
    DATA_SPLIT = "data_split"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    REPOSITORY = "repository"
    DATA_SUMMARY = "data_summary"
    LITERATURE = "literature"
    EXECUTION_DIAGNOSTICS = "execution_diagnostics"


class EvidenceRequest(ResearchContractModel):
    request_id: NonEmptyText
    reason: NonEmptyText
    categories: Annotated[tuple[EvidenceRequestCategory, ...], Field(min_length=1)]
    requested_items: NonEmptyStrings
    blocking: bool = True
    requires_test_labels: Literal[False] = False
    requires_external_training_data: Literal[False] = False


class InterpretationNextStep(StrEnum):
    PROPOSE_EXPERIMENT = "propose_experiment"
    REPLICATE = "replicate"
    INCREASE_FIDELITY = "increase_fidelity"
    ABANDON_DIRECTION = "abandon_direction"
    REQUEST_EVIDENCE = "request_evidence"


class ResearchInterpretation(ResearchContractModel):
    experiment_id: NonEmptyText
    outcome: ExperimentOutcome
    objective_findings: NonEmptyStrings
    execution_failure_kind: FailureKind | None = None
    interpretation: NonEmptyText
    evidence_refs: NonEmptyStrings
    next_step: InterpretationNextStep
    next_step_rationale: NonEmptyText

    @model_validator(mode="after")
    def validate_execution_failure(self) -> ResearchInterpretation:
        is_execution_failure = self.outcome is ExperimentOutcome.INVALID_EXECUTION
        if is_execution_failure != (self.execution_failure_kind is not None):
            raise ValueError(
                "invalid_execution outcome and execution_failure_kind must appear together"
            )
        return self


class ProposalPurpose(StrEnum):
    BASELINE_REPRODUCTION = "baseline_reproduction"
    OPTIMIZATION = "optimization"


class TargetAggregateTrainingStrategy(StrEnum):
    OUT_OF_FOLD = "out_of_fold"
    LEAVE_ONE_OUT = "leave_one_out"
    STRICTLY_PRIOR_EVENTS = "strictly_prior_events"


class AuxiliaryTargetName(StrEnum):
    IS_CLICK = "is_click"
    IS_LIKE = "is_like"
    IS_FOLLOW = "is_follow"
    IS_COMMENT = "is_comment"
    IS_FORWARD = "is_forward"
    IS_HATE = "is_hate"


class TargetDerivedFeatureControl(ResearchContractModel):
    training_strategy: TargetAggregateTrainingStrategy
    validation_strategy: Literal["training_only"] = "training_only"
    excludes_current_row_label: Literal[True] = True
    excludes_validation_labels: Literal[True] = True


class OfficialFMConfig(ResearchContractModel):
    model: Literal["FM"] = "FM"
    feature_fields: tuple[
        Literal["user_id"],
        Literal["video_id"],
        Literal["author_id"],
        Literal["tab"],
        Literal["dur_bucket"],
    ] = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
    latent_dimension: Literal[16] = 16
    learning_rate: Annotated[float, Field(gt=0.0)] = 0.001
    l2_regularization: Annotated[float, Field(ge=0.0)] = 0.000001
    batch_size: Literal[8192] = 8192
    max_epochs: Literal[40] = 40
    patience: Literal[4] = 4
    model_symbol: Literal["baseline.py::FM"] = "baseline.py::FM"
    evaluator_symbol: Literal["evaluate.py::evaluate"] = "evaluate.py::evaluate"
    forbidden_loader_symbol: Literal["data.py::load"] = "data.py::load"
    forbidden_training_symbol: Literal["baseline.py::run_fm"] = "baseline.py::run_fm"

    @model_validator(mode="after")
    def validate_float_config(self) -> OfficialFMConfig:
        if self.learning_rate != 0.001:
            raise ValueError("official FM learning_rate is fixed at 0.001")
        if self.l2_regularization != 0.000001:
            raise ValueError("official FM l2_regularization is fixed at 0.000001")
        return self


class BaselineReproductionControl(ResearchContractModel):
    official_fm_config: OfficialFMConfig
    execution_mode: Literal["safe_train_validation_wrapper"] = (
        "safe_train_validation_wrapper"
    )
    seeds: tuple[Literal[0], Literal[1], Literal[2], Literal[3], Literal[4]] = (
        0,
        1,
        2,
        3,
        4,
    )
    metric_tolerance: Annotated[float, Field(gt=0.0)] = 0.002
    official_reference_files_read_only: Literal[True] = True
    calls_official_data_load: Literal[False] = False
    calls_official_run_fm: Literal[False] = False
    reads_public_holdout: Literal[False] = False
    reads_organizer_hidden_test: Literal[False] = False

    @model_validator(mode="after")
    def validate_tolerance(self) -> BaselineReproductionControl:
        if self.metric_tolerance != 0.002:
            raise ValueError("baseline reproduction metric_tolerance is fixed at 0.002")
        return self


class Hypothesis(ResearchContractModel):
    schema_version: Literal["1"] = "1"
    hypothesis_id: NonEmptyText
    statement: NonEmptyText
    mechanism: NonEmptyText
    motivation: NonEmptyText
    evidence_refs: tuple[str, ...] = ()


class ExperimentProposal(ResearchContractModel):
    """ExperimentSpec plus explicit compliance declarations."""

    spec: ExperimentSpec
    hypothesis: Hypothesis | None = None
    purpose: ProposalPurpose
    benchmark_id: Literal["kuairand-pure"] = "kuairand-pure"
    positive_label: Literal["long_view"] = "long_view"
    metrics: tuple[MetricName, MetricName] = (
        MetricName.GAUC,
        MetricName.NDCG_5,
    )
    training_data_sources: tuple[Literal["KuaiRand-Pure"], ...] = (
        "KuaiRand-Pure",
    )
    requires_test_labels: Literal[False] = False
    uses_public_holdout_during_development: Literal[False] = False
    requires_organizer_hidden_test: Literal[False] = False
    uses_external_training_data: Literal[False] = False
    uses_pretrained_weights: Literal[False] = False
    pretrained_weight_provenance: tuple[str, ...] = ()
    uses_target_derived_features: bool
    target_derived_feature_control: TargetDerivedFeatureControl | None = None
    uses_auxiliary_training_targets: bool = False
    auxiliary_training_targets: tuple[AuxiliaryTargetName, ...] = ()
    baseline_reproduction_control: BaselineReproductionControl | None = None

    @model_validator(mode="before")
    @classmethod
    def populate_hypothesis(cls, data: object) -> object:
        if not isinstance(data, dict) or data.get("hypothesis") is not None:
            return data
        spec = data.get("spec")
        if isinstance(spec, ExperimentSpec):
            values = {
                "hypothesis_id": spec.hypothesis_id,
                "statement": spec.hypothesis,
                "mechanism": spec.mechanism,
                "motivation": spec.motivation,
                "evidence_refs": spec.evidence_refs,
            }
        elif isinstance(spec, dict):
            values = {
                "hypothesis_id": spec.get("hypothesis_id"),
                "statement": spec.get("hypothesis"),
                "mechanism": spec.get("mechanism"),
                "motivation": spec.get("motivation"),
                "evidence_refs": spec.get("evidence_refs", ()),
            }
        else:
            return data
        updated = dict(data)
        updated["hypothesis"] = values
        return updated

    @model_validator(mode="after")
    def validate_problem_statement_compliance(self) -> ExperimentProposal:
        hypothesis = self.hypothesis
        if hypothesis is None:
            raise ValueError("experiment proposal requires a structured hypothesis")
        expected_hypothesis = (
            self.spec.hypothesis_id,
            self.spec.hypothesis,
            self.spec.mechanism,
            self.spec.motivation,
            self.spec.evidence_refs,
        )
        actual_hypothesis = (
            hypothesis.hypothesis_id,
            hypothesis.statement,
            hypothesis.mechanism,
            hypothesis.motivation,
            hypothesis.evidence_refs,
        )
        if actual_hypothesis != expected_hypothesis:
            raise ValueError("structured hypothesis must match ExperimentSpec")
        if self.metrics != (MetricName.GAUC, MetricName.NDCG_5):
            raise ValueError("proposal metrics must be ordered as GAUC, nDCG@5")
        if self.training_data_sources != ("KuaiRand-Pure",):
            raise ValueError("KuaiRand-Pure must be the only training data source")
        if self.pretrained_weight_provenance:
            raise ValueError("pretrained weights and their provenance are prohibited")
        if self.uses_target_derived_features != (self.target_derived_feature_control is not None):
            raise ValueError(
                "uses_target_derived_features and target_derived_feature_control must agree"
            )
        if self.uses_auxiliary_training_targets != bool(self.auxiliary_training_targets):
            raise ValueError(
                "uses_auxiliary_training_targets and auxiliary_training_targets must agree"
            )
        if (self.purpose is ProposalPurpose.BASELINE_REPRODUCTION) != (
            self.baseline_reproduction_control is not None
        ):
            raise ValueError(
                "baseline_reproduction purpose and baseline_reproduction_control must agree"
            )
        return self


class ResearchDecisionKind(StrEnum):
    EXPERIMENT_PROPOSAL = "experiment_proposal"
    EVIDENCE_REQUEST = "evidence_request"
    RESULT_INTERPRETATION = "result_interpretation"


class ResearchDecision(ResearchContractModel):
    """Exactly one structured research decision."""

    schema_version: Literal["1"] = "1"
    response_id: NonEmptyText
    request_id: NonEmptyText
    kind: ResearchDecisionKind
    experiment_proposal: ExperimentProposal | None = None
    evidence_request: EvidenceRequest | None = None
    result_interpretation: ResearchInterpretation | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> ResearchDecision:
        payloads = {
            ResearchDecisionKind.EXPERIMENT_PROPOSAL: self.experiment_proposal,
            ResearchDecisionKind.EVIDENCE_REQUEST: self.evidence_request,
            ResearchDecisionKind.RESULT_INTERPRETATION: self.result_interpretation,
        }
        if payloads[self.kind] is None:
            raise ValueError(f"{self.kind} requires its matching payload")
        unexpected = [
            kind.value
            for kind, payload in payloads.items()
            if kind is not self.kind and payload is not None
        ]
        if unexpected:
            raise ValueError(f"unexpected payloads for {self.kind}: {unexpected}")
        if self.evidence_request and self.evidence_request.request_id != self.request_id:
            raise ValueError("evidence_request must refer to response request_id")
        return self


# Backward-compatible name used by the existing graph and Orchestration integration draft.
ResearchResponse = ResearchDecision


class ResearchFailureKind(StrEnum):
    MODEL = "model"
    SCHEMA = "schema"
    POLICY = "policy"


class ResearchAgentFailure(ResearchContractModel):
    schema_version: Literal["1"] = "1"
    request_id: NonEmptyText
    kind: ResearchFailureKind
    message: NonEmptyText
    repair_attempts: Annotated[int, Field(ge=0, le=1)]
