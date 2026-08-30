import pytest
from pydantic import ValidationError

from tiktok2026.contracts import (
    ControllerContext,
    EvaluationResult,
    ExecutionResult,
    FailureKind,
    FailureRecord,
    GraphStateReference,
    ImplementationAttemptRecord,
    ImplementationResult,
    MetricValue,
    ResourceState,
    RunPhase,
    SourceRegistration,
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
            MetricValue(name="NDCG@10", value=0.6),
            MetricValue(name="Recall@50", value=0.8),
        ),
        evaluator_artifact_id="evaluator-1",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
    )

    assert result.validation_score == pytest.approx(0.7)


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
