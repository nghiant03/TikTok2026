import pytest
from pydantic import ValidationError

from tiktok2026.contracts import (
    DEFAULT_IMPLEMENTATION_CRITERIA,
    CheckpointArtifactEnvelope,
    CheckpointRowRequirement,
    ControllerContext,
    CriterionAssessmentStatus,
    CriterionResolutionClaim,
    EvaluationResult,
    ExecutionResult,
    ExperimentProposalDecision,
    ExperimentSpec,
    FailureKind,
    FailureRecord,
    Fidelity,
    GraphStateReference,
    ImplementationAttemptRecord,
    ImplementationCriterionAssessment,
    ImplementationCriterionId,
    ImplementationRequest,
    ImplementationResourceEstimate,
    ImplementationResult,
    ImplementationSubmission,
    MetricValue,
    PredictionArtifactEnvelope,
    ResearchDecision,
    ResourceState,
    RunPhase,
    SourceRegistration,
    ValidationBlocker,
    ValidationBlockerContext,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
    validation_blocker_id,
)


def test_controller_context_exposes_the_concrete_experiment_interface() -> None:
    contract = ControllerContext().experiment_execution

    assert contract.available_splits == ("train", "valid")
    assert contract.prediction_rows == "exact_valid_manifest_rows_in_manifest_order"
    assert contract.separate_candidate_input is False
    assert contract.valid_labels_may_influence_scores is False
    assert contract.required_artifacts == ("predictions.json", "checkpoint_bundle.json")
    assert contract.output_visibility == "private_until_controller_validation"
    assert contract.artifact_publication_owner == "controller"
    assert contract.required_dataset_hash_arguments == ("--dataset-manifest-sha256",)
    assert contract.optional_dataset_hash_arguments == ("--dataset-view-sha256",)
    assert isinstance(contract.checkpoint_artifact_envelope, CheckpointArtifactEnvelope)
    assert contract.prediction_row_requirements.required_fields == (
        "row_id",
        "row_identity",
        "user_id",
        "item_id",
        "score",
    )


@pytest.mark.parametrize(
    "metrics",
    (("nDCG@5", "GAUC"), ("GAUC", "GAUC")),
)
def test_controller_context_rejects_noncanonical_judging_metric_order(
    metrics: tuple[str, str],
) -> None:
    with pytest.raises(ValidationError, match="current order"):
        ControllerContext(judging_metrics=metrics)  # type: ignore[arg-type]


def test_prediction_and_checkpoint_envelopes_require_dataset_view_key() -> None:
    assert "dataset_view_sha256" in PredictionArtifactEnvelope().required_fields
    assert "dataset_view_sha256" in CheckpointArtifactEnvelope().required_fields
    assert "dataset_view_sha256" in CheckpointRowRequirement().required_fields


def test_implementation_contracts_allow_metadata_only_and_typed_criteria() -> None:
    submission = ImplementationSubmission(
        experiment_id="experiment-1",
        patch_artifact_id="patch-1",
        changed_files=(),
    )
    assert submission.edits == ()
    assessment = ImplementationCriterionAssessment(
        criterion_id=ImplementationCriterionId.EXECUTION_WIRING,
        status=CriterionAssessmentStatus.PARTIAL,
    )
    assert assessment.criterion_id == ImplementationCriterionId.EXECUTION_WIRING


def test_implementation_criterion_matrix_has_atomic_recurring_failure_families() -> None:
    assert {
        criterion.value
        for criterion in DEFAULT_IMPLEMENTATION_CRITERIA
    } >= {
        "scientific_fidelity",
        "changed_path_scope",
        "leakage",
        "unrelated_changes",
        "execution_wiring",
        "static_checks",
        "cli_artifact_contract",
        "provenance",
        "strict_json_types",
        "row_coverage_order",
        "deterministic_ranking_tie_policy",
        "experiment_specific_reconstruction",
        "resource_feasibility",
    }


def test_implementation_request_defaults_to_the_bounded_criterion_matrix() -> None:
    request = ImplementationRequest(
        request_id="request-1",
        experiment_id="experiment-1",
        experiment_spec=ExperimentSpec(
            experiment_id="experiment-1",
            hypothesis_id="hypothesis-1",
            hypothesis="hypothesis",
            mechanism="mechanism",
            motivation="motivation",
            expected_signal="signal",
            implementation_scope=("src/tiktok2026/experiment/train.py",),
            fidelity=Fidelity.SMOKE,
            success_criteria="success",
            failure_criteria="failure",
        ),
        allowed_scopes=("src/tiktok2026/experiment",),
    )
    assert request.required_implementation_criteria == DEFAULT_IMPLEMENTATION_CRITERIA


def test_implementation_resource_estimate_validates_and_round_trips() -> None:
    estimate = ImplementationResourceEstimate(
        predicted_wall_seconds=120.5,
        predicted_peak_memory_bytes=2_000_000,
        predicted_artifact_bytes=50_000,
        dataset_passes=2,
        high_cardinality_nested_scans=True,
        duplicate_full_materializations=False,
    )

    assert (
        ImplementationResourceEstimate.model_validate_json(estimate.model_dump_json()) == estimate
    )

    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="hypothesis",
        mechanism="mechanism",
        motivation="motivation",
        expected_signal="signal",
        implementation_scope=("src/tiktok2026/experiment/train.py",),
        fidelity=Fidelity.SMOKE,
        implementation_resource_estimate=estimate,
        success_criteria="success",
        failure_criteria="failure",
    )
    assert spec.implementation_resource_estimate == estimate


@pytest.mark.parametrize(
    "field, value",
    (
        ("predicted_wall_seconds", -1.0),
        ("predicted_peak_memory_bytes", -1),
        ("predicted_artifact_bytes", -1),
        ("dataset_passes", 17),
    ),
)
def test_implementation_resource_estimate_rejects_invalid_bounds(
    field: str, value: float | int
) -> None:
    values: dict[str, object] = {
        "predicted_wall_seconds": 1.0,
        "predicted_peak_memory_bytes": 1,
        "predicted_artifact_bytes": 1,
        "dataset_passes": 1,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        ImplementationResourceEstimate.model_validate(values)


def test_experiment_spec_keeps_resource_estimate_optional() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="hypothesis",
        mechanism="mechanism",
        motivation="motivation",
        expected_signal="signal",
        implementation_scope=("src/tiktok2026/experiment/train.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="success",
        failure_criteria="failure",
    )

    assert spec.implementation_resource_estimate is None


def test_experiment_proposal_requires_resource_estimate() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="hypothesis",
        mechanism="mechanism",
        motivation="motivation",
        expected_signal="signal",
        implementation_scope=("src/tiktok2026/experiment/train.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="success",
        failure_criteria="failure",
    )

    with pytest.raises(ValidationError, match="implementation_resource_estimate"):
        ExperimentProposalDecision(
            request_id="request-1",
            kind="proposal",
            experiment_spec=spec,
            message="proposal",
        )


def test_experiment_spec_legacy_records_remain_parseable_without_estimate() -> None:
    spec = ExperimentSpec.model_validate(
        {
            "experiment_id": "experiment-legacy",
            "hypothesis_id": "hypothesis-1",
            "hypothesis": "hypothesis",
            "mechanism": "mechanism",
            "motivation": "motivation",
            "expected_signal": "signal",
            "implementation_scope": ["src/tiktok2026/experiment/train.py"],
            "fidelity": "smoke",
            "success_criteria": "success",
            "failure_criteria": "failure",
        }
    )

    assert spec.implementation_resource_estimate is None


def test_research_proposal_requires_experiment_spec() -> None:
    with pytest.raises(ValidationError, match="experiment_spec"):
        ResearchDecision(
            request_id="request-1",
            kind="proposal",
            message="proposal",
        )


def test_research_proposal_requires_resource_estimate() -> None:
    spec = ExperimentSpec(
        experiment_id="experiment-1",
        hypothesis_id="hypothesis-1",
        hypothesis="hypothesis",
        mechanism="mechanism",
        motivation="motivation",
        expected_signal="signal",
        implementation_scope=("src/tiktok2026/experiment/train.py",),
        fidelity=Fidelity.SMOKE,
        success_criteria="success",
        failure_criteria="failure",
    )

    with pytest.raises(ValidationError, match="implementation_resource_estimate"):
        ResearchDecision(
            request_id="request-1",
            kind="proposal",
            experiment_spec=spec,
            message="proposal",
        )


def test_non_proposal_research_decisions_keep_optional_experiment_spec() -> None:
    assert ResearchDecision(
        request_id="request-1",
        kind="evidence_request",
        message="request evidence",
    ).experiment_spec is None
    assert ResearchDecision(
        request_id="request-2",
        kind="interpretation",
        experiment_spec=ExperimentSpec(
            experiment_id="experiment-1",
            hypothesis_id="hypothesis-1",
            hypothesis="hypothesis",
            mechanism="mechanism",
            motivation="motivation",
            expected_signal="signal",
            implementation_scope=("src/tiktok2026/experiment/train.py",),
            fidelity=Fidelity.SMOKE,
            success_criteria="success",
            failure_criteria="failure",
        ),
        message="interpretation",
    ).experiment_spec is not None


def test_validation_blocker_context_keeps_criterion_identity() -> None:
    context = ValidationBlockerContext(
        blocker_id="blocker-1",
        criterion_id=ImplementationCriterionId.PROVENANCE,
        text="provenance is missing",
    )
    assert context.criterion_id == ImplementationCriterionId.PROVENANCE


def test_criterion_blocker_identity_is_independent_of_report_and_text() -> None:
    first = ValidationBlocker(
        blocker_id="ignored",
        experiment_id="experiment-1",
        stage=ValidationStage.IMPLEMENTATION,
        text="first wording",
        report_id="report-1",
        criterion_id=ImplementationCriterionId.LEAKAGE,
    )
    second = ValidationBlocker(
        blocker_id="also-ignored",
        experiment_id="experiment-1",
        stage=ValidationStage.IMPLEMENTATION,
        text="reworded",
        report_id="report-2",
        criterion_id=ImplementationCriterionId.LEAKAGE,
    )
    assert first.blocker_id == second.blocker_id == validation_blocker_id(
        "report-1",
        ValidationStage.IMPLEMENTATION,
        "first wording",
        criterion_id=ImplementationCriterionId.LEAKAGE,
        experiment_id="experiment-1",
    )


def test_validation_report_rejects_supplied_id_for_criterion_blocker() -> None:
    with pytest.raises(ValidationError, match="cannot resolve"):
        ValidationReport.model_validate(
            {
                "report_id": "report-1",
                "experiment_id": "experiment-1",
                "stage": ValidationStage.IMPLEMENTATION,
                "verdict": ValidationVerdict.APPROVED,
                "leakage_risk": "none",
                "blockers": (
                    {
                        "blocker_id": "legacy-blocker-id",
                        "criterion_id": ImplementationCriterionId.LEAKAGE,
                        "text": "leakage found",
                    },
                ),
                "resolves_blocker_ids": ("legacy-blocker-id",),
                "evidence_refs": ("evidence-1",),
            }
        )


def test_validation_report_rejects_duplicate_criterion_assessments() -> None:
    assessment = ImplementationCriterionAssessment(
        criterion_id=ImplementationCriterionId.LEAKAGE,
        status=CriterionAssessmentStatus.FAIL,
    )

    with pytest.raises(ValidationError, match="criterion_assessments"):
        ValidationReport(
            report_id="report-1",
            experiment_id="experiment-1",
            stage=ValidationStage.IMPLEMENTATION,
            verdict=ValidationVerdict.REPAIRABLE,
            criterion_assessments=(assessment, assessment),
            leakage_risk="none",
        )


def test_validation_report_rejects_pre_normalization_id_from_blocker_instance() -> None:
    blocker = ValidationBlocker(
        blocker_id="legacy-blocker-id",
        experiment_id="experiment-1",
        stage=ValidationStage.IMPLEMENTATION,
        text="leakage found",
        report_id="report-1",
        criterion_id=ImplementationCriterionId.LEAKAGE,
    )
    assert blocker.blocker_id != "legacy-blocker-id"

    with pytest.raises(ValidationError, match="cannot resolve"):
        ValidationReport(
            report_id="report-1",
            experiment_id="experiment-1",
            stage=ValidationStage.IMPLEMENTATION,
            verdict=ValidationVerdict.APPROVED,
            blockers=(blocker,),
            resolves_blocker_ids=("legacy-blocker-id",),
            evidence_refs=("evidence-1",),
            leakage_risk="none",
        )


def test_validation_report_supports_partial_criterion_resolution() -> None:
    report = ValidationReport(
        report_id="report-1",
        experiment_id="experiment-1",
        stage=ValidationStage.IMPLEMENTATION,
        verdict=ValidationVerdict.REPAIRABLE,
        leakage_risk="none",
        resolution_claims=(
            CriterionResolutionClaim(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PARTIAL,
                blocker_ids=("blocker-old",),
                evidence_refs=("evidence-1",),
            ),
        ),
    )
    assert report.criterion_resolution_claims[0].blocker_ids == ("blocker-old",)


def test_legacy_criterion_resolution_claim_alias_remains_supported() -> None:
    report = ValidationReport.model_validate(
        {
            "report_id": "report-legacy",
            "experiment_id": "experiment-1",
            "stage": ValidationStage.IMPLEMENTATION,
            "verdict": ValidationVerdict.REPAIRABLE,
            "leakage_risk": "none",
            "criterion_resolution_claims": (
                {
                    "criterion_id": "leakage",
                    "status": "partial",
                    "blocker_ids": ("blocker-old",),
                    "evidence_refs": ("evidence-1",),
                },
            ),
        }
    )
    assert report.resolution_claims[0].criterion_id == ImplementationCriterionId.LEAKAGE


@pytest.mark.parametrize(
    "status",
    (CriterionAssessmentStatus.PASS, CriterionAssessmentStatus.PARTIAL),
)
def test_resolution_claims_require_evidence_for_positive_statuses(
    status: CriterionAssessmentStatus,
) -> None:
    with pytest.raises(ValidationError, match="evidence_refs"):
        CriterionResolutionClaim(
            criterion_id=ImplementationCriterionId.LEAKAGE,
            status=status,
        )


def test_failed_criterion_cannot_claim_resolution() -> None:
    with pytest.raises(ValidationError, match="cannot claim resolution"):
        CriterionResolutionClaim(
            criterion_id=ImplementationCriterionId.LEAKAGE,
            status=CriterionAssessmentStatus.FAIL,
            evidence_refs=("evidence-1",),
        )


def test_lifecycle_contracts_allow_three_repairs_but_not_four() -> None:
    result = ImplementationResult(
        experiment_id="experiment-1",
        patch_artifact_id="patch-1",
        changed_files=("src/tiktok2026/experiment/train.py",),
    )
    assert (
        ImplementationAttemptRecord(
            experiment_id="experiment-1", repair_attempt=3, result=result
        ).repair_attempt
        == 3
    )
    assert FailureRecord(
        failure_id="failure-3",
        experiment_id="experiment-1",
        kind=FailureKind.SCHEMA_MISMATCH,
        evidence_refs=("evidence-1",),
        repair_attempt=3,
    ).repair_attempt == 3
    assert GraphStateReference(
        run_id="run-1", phase=RunPhase.IMPLEMENT, repair_attempts=3
    ).repair_attempts == 3
    with pytest.raises(ValidationError):
        GraphStateReference(run_id="run-1", phase=RunPhase.IMPLEMENT, repair_attempts=4)


def test_validation_score_uses_equal_weight_judging_metrics() -> None:
    result = EvaluationResult(
        evaluation_id="evaluation-1",
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        metrics=(
            MetricValue(name="GAUC", value=0.6),
            MetricValue(name="nDCG@5", value=0.8),
        ),
        evaluator_artifact_id="evaluator-1",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
    )

    assert result.validation_score == pytest.approx(0.7)


def test_historical_evaluation_result_json_remains_parseable() -> None:
    result = EvaluationResult(
        evaluation_id="historical-evaluation-1",
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        metrics=(
            MetricValue(name="NDCG@10", value=0.6),
            MetricValue(name="Recall@50", value=0.8),
        ),
        evaluator_artifact_id="provisional-within-user-v1",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
    )

    restored = EvaluationResult.model_validate_json(result.model_dump_json())
    assert restored.validation_score == pytest.approx(0.7)


@pytest.mark.parametrize(
    "metrics",
    (
        (MetricValue(name="GAUC", value=0.5),),
        (
            MetricValue(name="GAUC", value=0.5),
            MetricValue(name="NDCG@10", value=0.5),
        ),
        (
            MetricValue(name="GAUC", value=0.5),
            MetricValue(name="GAUC", value=0.6),
        ),
    ),
)
def test_evaluation_result_rejects_incomplete_or_mixed_metric_pairs(
    metrics: tuple[MetricValue, ...],
) -> None:
    with pytest.raises(ValidationError, match="metric pair"):
        EvaluationResult(
            evaluation_id="evaluation-1",
            experiment_id="experiment-1",
            checkpoint_id="checkpoint-1",
            metrics=metrics,
            evaluator_artifact_id="evaluator-1",
            evaluator_sha256="a" * 64,
            prediction_sha256="b" * 64,
            validity="provisional",
        )


def test_failed_execution_requires_failure_classification() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult(
            execution_id="execution-1",
            experiment_id="experiment-1",
            source_commit="abc123",
            command=("python", "train.py"),
            exit_code=1,
            elapsed_seconds=1.0,
            gpu_hours=0.0,
        )


def test_successful_execution_rejects_failure_classification() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult(
            execution_id="execution-1",
            experiment_id="experiment-1",
            source_commit="abc123",
            command=("python", "train.py"),
            exit_code=0,
            elapsed_seconds=1.0,
            gpu_hours=0.0,
            failure_kind=FailureKind.TIMEOUT,
        )


def test_source_registration_identity_is_bound_to_commit() -> None:
    with pytest.raises(ValidationError, match="identity does not match"):
        SourceRegistration(
            registration_id=f"source-{'b' * 40}",
            experiment_id="experiment-1",
            run_id="run-1",
            parent_commit="a" * 40,
            source_commit="c" * 40,
            patch_sha256="d" * 64,
            patch_artifact_id=f"patch-{'d' * 64}",
            patch_artifact_uri="file:///tmp/patch.diff",
            allowed_scopes=("src/tiktok2026/experiment",),
            eligible=True,
        )


def test_final_gpu_reserve_cannot_exceed_remaining_budget() -> None:
    with pytest.raises(ValidationError):
        ResourceState(
            remaining_gpu_hours=1.0,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=1.0,
            used_tokens=0,
            remaining_tokens=1,
            disk_bytes_available=1,
            reserved_final_gpu_hours=1.1,
        )
