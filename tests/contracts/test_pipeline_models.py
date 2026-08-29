from pathlib import Path

import pytest
from pydantic import ValidationError

from tiktok2026.contracts import (
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    ContractModel,
    FinalizationRecord,
    ResourceReservation,
    RuntimePaths,
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


def test_artifact_retention_values_are_stable() -> None:
    assert ArtifactRetention.PROVENANCE.value == "provenance"


def test_contract_models_are_immutable() -> None:
    model = ContractModel()
    with pytest.raises(ValidationError):
        model.extra = "forbidden"  # type: ignore[attr-defined]
