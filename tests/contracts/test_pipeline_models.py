from pathlib import Path

import pytest
from pydantic import ValidationError

from tiktok2026.contracts import (
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    ContractModel,
    ExperimentHistoryContext,
    ExperimentSpec,
    Fidelity,
    FinalizationRecord,
    OutcomeSummary,
    PredictionArtifactRegistration,
    ResourceReservation,
    RuntimePaths,
    SourceContext,
)


def test_runtime_roles_are_exactly_the_four_authorized_roles() -> None:
    assert {role.value for role in AgentRole} == {
        "orchestration",
        "research",
        "implementor",
        "validator",
    }


def test_registered_artifact_requires_sha256() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord(
            artifact_id="artifact-1",
            run_id="run-1",
            kind="predictions",
            uri="file:///tmp/predictions.csv",
            sha256="bad",
            size_bytes=1,
            producer="controller",
            retention=ArtifactRetention.RUN,
        )


def test_prediction_registration_supports_optional_dataset_view_provenance(
    tmp_path: Path,
) -> None:
    common = {
        "artifact_id": "predictions-1",
        "path": tmp_path / "predictions.json",
        "sha256": "a" * 64,
        "checkpoint_id": "checkpoint-1",
        "source_commit": "b" * 40,
        "execution_id": "execution-1",
        "dataset_manifest_id": "manifest-1",
        "dataset_manifest_sha256": "c" * 64,
        "split": "valid",
    }

    historic = PredictionArtifactRegistration.model_validate(common)
    current = PredictionArtifactRegistration.model_validate(
        {**common, "dataset_view_sha256": "d" * 64}
    )

    assert historic.dataset_view_sha256 is None
    assert current.dataset_view_sha256 == "d" * 64


def test_provisional_finalization_is_explicit() -> None:
    record = FinalizationRecord(
        finalization_id="final-1",
        run_id="run-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        checkpoint_id="checkpoint-1",
        evaluation_id="evaluation-1",
        validity="provisional",
        bundle_artifact_id="bundle-1",
        consumed_test_access=True,
    )
    assert record.validity == "provisional"


def test_runtime_paths_reject_repository_descendant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        RuntimePaths.create(tmp_path, tmp_path / ".runtime")


def test_resource_reservation_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        ResourceReservation(
            reservation_id="reservation-1",
            run_id="run-1",
            experiment_id="exp-1",
            gpu_hours=-1.0,
            wall_seconds=1.0,
            tokens=1,
            disk_bytes=1,
        )


def test_experiment_scope_rejects_prose_appended_to_path() -> None:
    with pytest.raises(ValidationError, match="canonical relative paths without prose"):
        ExperimentSpec(
            experiment_id="exp-1",
            hypothesis_id="hyp-1",
            hypothesis="hypothesis",
            mechanism="mechanism",
            motivation="motivation",
            expected_signal="signal",
            implementation_scope=(
                "src/tiktok2026/experiment/model.py: implement the model",
            ),
            fidelity=Fidelity.SMOKE,
            success_criteria="success",
            failure_criteria="failure",
        )


def test_bounded_contexts_are_immutable_and_report_truncation() -> None:
    source = SourceContext(
        evidence_id="source-context-1",
        source_commit="a" * 40,
        parent_commit="a" * 40,
        entrypoint_sha256="b" * 64,
        source_summary="bounded structure",
        source_excerpt="def train(): pass",
        structural_summary=("functiondef:train",),
    )
    history = ExperimentHistoryContext(
        evidence_id="history-context-1",
        run_id="run-1",
        records=(OutcomeSummary(experiment_id="exp-1", hypothesis="h"),),
        total_records=2,
        truncated=True,
    )

    assert source.source_commit == source.parent_commit
    assert history.truncated is True
    with pytest.raises(ValidationError):
        source.source_commit = "c" * 40  # type: ignore[misc]


def test_history_context_rejects_more_than_fifty_items() -> None:
    records = tuple(
        OutcomeSummary(experiment_id=f"exp-{index}", hypothesis="h")
        for index in range(51)
    )
    with pytest.raises(ValidationError):
        ExperimentHistoryContext(
            evidence_id="history-context-1",
            run_id="run-1",
            records=records,
            total_records=51,
        )


def test_artifact_retention_values_are_stable() -> None:
    assert ArtifactRetention.PROVENANCE.value == "provenance"


def test_contract_models_are_immutable() -> None:
    model = ContractModel()
    with pytest.raises(ValidationError):
        model.extra = "forbidden"  # type: ignore[attr-defined]
