from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.contracts import (
    AuxiliaryTargetName,
    BaselineReproductionControl,
    BaselineStatus,
    BenchmarkContract,
    EvaluationProtocolStatus,
    EvidenceRequest,
    EvidenceRequestCategory,
    ExperimentProposal,
    MetricName,
    OfficialFMConfig,
    ProposalPurpose,
    ResearchDecision,
    ResearchDecisionKind,
    ResearchEvaluationResult,
    ResearchMetricValue,
    ResearchRequest,
    ResearchResponse,
    ResearchTaskType,
    TargetAggregateTrainingStrategy,
    TargetDerivedFeatureControl,
)
from research_agent.shared_contracts import (
    ExecutionResult,
    ExperimentSpec,
    FailureKind,
    Fidelity,
    ResourceState,
)
from tests.factories import make_proposal_response


def test_benchmark_contract_is_problem_statement_fixed() -> None:
    contract = BenchmarkContract()

    assert contract.positive_label == "long_view"
    assert contract.metrics == (MetricName.GAUC, MetricName.NDCG_5)
    assert contract.validation_ranking == "mean(GAUC,nDCG@5)"
    assert contract.validation_date_range == (20220422, 20220428)
    assert contract.test_labels_accessible is False
    assert contract.external_training_data_allowed is False
    assert contract.compliant_pretrained_weights_allowed is False


def test_proposal_exposes_explicit_decision_and_hypothesis(proposal_request) -> None:
    decision = make_proposal_response(proposal_request)

    assert isinstance(decision, ResearchDecision)
    assert decision.experiment_proposal is not None
    hypothesis = decision.experiment_proposal.hypothesis
    assert hypothesis is not None
    assert hypothesis.hypothesis_id == decision.experiment_proposal.spec.hypothesis_id
    assert hypothesis.statement == decision.experiment_proposal.spec.hypothesis


def test_benchmark_rejects_old_click_label() -> None:
    with pytest.raises(ValidationError):
        BenchmarkContract(positive_label="is_click")  # pyright: ignore[reportArgumentType]


def test_evidence_response_requires_only_matching_payload() -> None:
    response = ResearchResponse(
        response_id="response-1",
        request_id="request-1",
        kind=ResearchDecisionKind.EVIDENCE_REQUEST,
        evidence_request=EvidenceRequest(
            request_id="request-1",
            reason="The current feature pipeline is unknown.",
            categories=(EvidenceRequestCategory.REPOSITORY,),
            requested_items=("repository feature pipeline summary",),
        ),
    )

    assert response.evidence_request is not None
    assert response.experiment_proposal is None


def test_response_rejects_kind_payload_mismatch() -> None:
    with pytest.raises(ValidationError):
        ResearchResponse(
            response_id="response-1",
            request_id="request-1",
            kind=ResearchDecisionKind.EXPERIMENT_PROPOSAL,
            evidence_request=EvidenceRequest(
                request_id="request-1",
                reason="More evidence is required.",
                categories=(EvidenceRequestCategory.DATA_SUMMARY,),
                requested_items=("data summary",),
            ),
        )


def test_interpretation_request_requires_matching_result(resource_state: ResourceState) -> None:
    result = ResearchEvaluationResult(
        evaluation_id="evaluation-1",
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        metrics=(
            ResearchMetricValue(name=MetricName.GAUC, value=0.6674),
            ResearchMetricValue(name=MetricName.NDCG_5, value=0.5357),
        ),
        evaluator_artifact_id="evaluator-1",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
    )

    request = ResearchRequest(
        request_id="request-2",
        task_type=ResearchTaskType.INTERPRET_RESULT,
        objective="Interpret the latest result.",
        current_experiment_id="experiment-1",
        evaluation_result=result,
        resource_state=resource_state,
        allowed_implementation_scope=("experiment/features",),
    )

    assert request.evaluation_result == result


def test_interpretation_request_rejects_missing_result(resource_state: ResourceState) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(
            request_id="request-2",
            task_type=ResearchTaskType.INTERPRET_RESULT,
            objective="Interpret the latest result.",
            current_experiment_id="experiment-1",
            resource_state=resource_state,
            allowed_implementation_scope=("experiment/features",),
        )


def test_experiment_proposal_rejects_non_problem_statement_label() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="A valid hypothesis.",
        mechanism="A valid mechanism.",
        motivation="Traceable motivation.",
        expected_signal="Both metrics improve.",
        implementation_scope=("experiment/features/a.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Both metrics improve.",
        failure_criteria="They do not improve.",
    )

    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            positive_label="is_click",  # pyright: ignore[reportArgumentType]
        )


def test_pretrained_weights_are_forbidden_even_with_provenance() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="A valid hypothesis.",
        mechanism="A valid mechanism.",
        motivation="Traceable motivation.",
        expected_signal="Both metrics improve.",
        implementation_scope=("experiment/models/a.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Both metrics improve.",
        failure_criteria="They do not improve.",
    )

    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.OPTIMIZATION,
            uses_target_derived_features=False,
            uses_pretrained_weights=True,  # pyright: ignore[reportArgumentType]
            pretrained_weight_provenance=("model://public/checkpoint",),
        )

    with pytest.raises(ValidationError, match="pretrained weights"):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.OPTIMIZATION,
            uses_target_derived_features=False,
            pretrained_weight_provenance=("model://public/checkpoint",),
        )


def test_experiment_proposal_rejects_wrong_metrics_or_training_source() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="A valid hypothesis.",
        mechanism="A valid mechanism.",
        motivation="Traceable motivation.",
        expected_signal="Both metrics improve.",
        implementation_scope=("experiment/features/a.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Both metrics improve.",
        failure_criteria="They do not improve.",
    )

    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            metrics=(MetricName.NDCG_5, MetricName.GAUC),
        )
    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            training_data_sources=(),
        )


def test_evidence_request_cannot_require_forbidden_data() -> None:
    with pytest.raises(ValidationError):
        EvidenceRequest(
            request_id="request-1",
            reason="Forbidden evidence.",
            categories=(EvidenceRequestCategory.DATA_SUMMARY,),
            requested_items=("hidden evaluation labels",),
            requires_test_labels=True,  # pyright: ignore[reportArgumentType]
        )


def test_evaluation_protocol_status_requires_traceable_evidence(
    proposal_request,
) -> None:
    payload = proposal_request.model_dump(mode="python")
    payload.update(
        evaluation_protocol_status=EvaluationProtocolStatus.CONFIRMED,
        evaluation_protocol_evidence_refs=(),
    )
    with pytest.raises(ValidationError, match="evaluation protocol requires"):
        ResearchRequest.model_validate(payload)

    payload.update(
        evaluation_protocol_status=EvaluationProtocolStatus.UNCONFIRMED,
        evaluation_protocol_evidence_refs=("protocol-evidence",),
    )
    with pytest.raises(ValidationError, match="unconfirmed evaluation protocol"):
        ResearchRequest.model_validate(payload)


def test_target_derived_features_require_structured_leakage_control(
    proposal_request,
) -> None:
    spec = make_proposal_response(proposal_request).experiment_proposal.spec
    with pytest.raises(ValidationError, match="must agree"):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=True,
        )

    proposal = ExperimentProposal(
        spec=spec,
        purpose=ProposalPurpose.BASELINE_REPRODUCTION,
        uses_target_derived_features=True,
        target_derived_feature_control=TargetDerivedFeatureControl(
            training_strategy=TargetAggregateTrainingStrategy.STRICTLY_PRIOR_EVENTS,
        ),
        baseline_reproduction_control=BaselineReproductionControl(
            official_fm_config=OfficialFMConfig()
        ),
    )
    assert proposal.target_derived_feature_control is not None
    assert proposal.target_derived_feature_control.validation_strategy == "training_only"


def test_auxiliary_training_targets_require_explicit_declaration(
    proposal_request,
) -> None:
    spec = make_proposal_response(proposal_request).experiment_proposal.spec
    with pytest.raises(ValidationError, match="auxiliary_training_targets must agree"):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            auxiliary_training_targets=(AuxiliaryTargetName.IS_CLICK,),
        )

    proposal = ExperimentProposal(
        spec=spec,
        purpose=ProposalPurpose.BASELINE_REPRODUCTION,
        uses_target_derived_features=False,
        uses_auxiliary_training_targets=True,
        auxiliary_training_targets=(AuxiliaryTargetName.IS_CLICK,),
        baseline_reproduction_control=BaselineReproductionControl(
            official_fm_config=OfficialFMConfig()
        ),
    )
    assert proposal.auxiliary_training_targets == (AuxiliaryTargetName.IS_CLICK,)


def test_baseline_reproduction_requires_safe_control(proposal_request) -> None:
    proposal = make_proposal_response(proposal_request).experiment_proposal
    payload = proposal.model_dump(mode="python")
    payload["baseline_reproduction_control"] = None

    with pytest.raises(ValidationError, match="baseline_reproduction_control must agree"):
        ExperimentProposal.model_validate(payload)


def test_baseline_reproduction_control_rejects_unsafe_execution() -> None:
    with pytest.raises(ValidationError):
        BaselineReproductionControl(
            official_fm_config=OfficialFMConfig(),
            calls_official_data_load=True,  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValidationError, match="metric_tolerance"):
        BaselineReproductionControl(
            official_fm_config=OfficialFMConfig(),
            metric_tolerance=0.01,
        )


def test_official_fm_config_rejects_wrong_parameters_or_symbols() -> None:
    with pytest.raises(ValidationError):
        OfficialFMConfig(latent_dimension=32)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValidationError, match="learning_rate"):
        OfficialFMConfig(learning_rate=0.01)
    with pytest.raises(ValidationError):
        OfficialFMConfig(
            evaluator_symbol="baseline.py::evaluate"  # pyright: ignore[reportArgumentType]
        )


def test_baseline_status_requires_traceable_evidence(proposal_request) -> None:
    payload = proposal_request.model_dump(mode="python")
    payload["baseline_status"] = BaselineStatus.PROVISIONAL

    with pytest.raises(ValidationError, match="baseline_evidence_refs"):
        ResearchRequest.model_validate(payload)


def test_proposal_rejects_external_training_data() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-external",
        hypothesis_id="hypothesis-external",
        hypothesis="External data might improve ranking.",
        mechanism="Train with an external dataset.",
        motivation="Invalid external-data proposal.",
        expected_signal="Metrics improve.",
        implementation_scope=("experiment/features/external.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Metrics improve.",
        failure_criteria="Metrics do not improve.",
    )

    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            uses_external_training_data=True,  # pyright: ignore[reportArgumentType]
        )


def test_proposal_rejects_public_holdout_or_organizer_hidden_test_access(
    proposal_request,
) -> None:
    spec = make_proposal_response(proposal_request).experiment_proposal.spec
    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            uses_public_holdout_during_development=True,  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValidationError):
        ExperimentProposal(
            spec=spec,
            purpose=ProposalPurpose.BASELINE_REPRODUCTION,
            uses_target_derived_features=False,
            requires_organizer_hidden_test=True,  # pyright: ignore[reportArgumentType]
        )


def test_interpretation_request_accepts_failed_execution(
    resource_state: ResourceState,
) -> None:
    execution = ExecutionResult(
        execution_id="execution-failed-1",
        experiment_id="experiment-1",
        source_commit="commit-1",
        command=("python", "train.py"),
        exit_code=1,
        elapsed_seconds=12.5,
        gpu_hours=0.0,
        failure_kind=FailureKind.SYNTAX_IMPORT,
    )

    request = ResearchRequest(
        request_id="request-failed-execution-1",
        task_type=ResearchTaskType.INTERPRET_RESULT,
        objective="Interpret the failed execution.",
        current_experiment_id="experiment-1",
        execution_result=execution,
        resource_state=resource_state,
        allowed_implementation_scope=("experiment/features",),
    )

    assert request.execution_result == execution


def test_evaluation_result_rejects_incomplete_metrics() -> None:
    with pytest.raises(ValidationError):
        ResearchEvaluationResult(
            evaluation_id="evaluation-incomplete",
            experiment_id="experiment-1",
            checkpoint_id="checkpoint-1",
            metrics=(  # pyright: ignore[reportArgumentType]
                ResearchMetricValue(name=MetricName.GAUC, value=0.5),
            ),
            evaluator_artifact_id="evaluator-1",
            evaluator_sha256="a" * 64,
            prediction_sha256="b" * 64,
            validity="provisional",
        )
