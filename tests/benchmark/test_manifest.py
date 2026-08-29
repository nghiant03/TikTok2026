import csv
import json
from pathlib import Path

import pytest

from tiktok2026.benchmark.kuaireand_pure.adapter import KuaiRandPureAdapter
from tiktok2026.benchmark.kuaireand_pure.manifest import (
    BenchmarkManifest,
    DatasetFile,
    DatasetManifest,
    DatasetSplit,
    VerifiedDataset,
    authorized_training_view,
    canonical_manifest_sha256,
    verify_dataset_manifest,
    verify_protected_files,
)
from tiktok2026.contracts import PredictionRow


def test_canonical_manifest_uses_judging_metrics() -> None:
    path = Path("src/tiktok2026/benchmark/kuaireand_pure/manifest.json")
    manifest = BenchmarkManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert manifest.judging_metrics == ("NDCG@10", "Recall@50")
    assert manifest.judging_evaluator_status == "provisional"


def test_protected_file_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "evaluate.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="protected file hash mismatch"):
        verify_protected_files(tmp_path, {"baseline/evaluate.py": "0" * 64})


def test_dataset_file_cannot_escape_declared_root(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        manifest_id="fixture-v1",
        data_root_env="UNUSED",
        files=(
            DatasetFile.model_construct(
                path="../outside.csv",
                sha256="0" * 64,
                columns=("row_id", "label"),
                split="train",
            ),
        ),
        splits={"train": DatasetSplit(files=("../outside.csv",), identity_sha256="0" * 64)},
    )
    with pytest.raises(ValueError, match="escapes root"):
        verify_dataset_manifest(manifest, tmp_path, splits={"train"})


def test_submission_preserves_supplied_row_identities(tmp_path: Path) -> None:
    prediction = PredictionRow(
        row_id='["row-42","u1","v1"]',
        row_identity=("row-42", "u1", "v1"),
        user_id="u1",
        item_id="v1",
        score=0.5,
    )
    output = KuaiRandPureAdapter().write_prediction_submission(
        tmp_path / "submission.csv", (prediction,), (prediction,)
    )
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["row_id"] == '["row-42","u1","v1"]'


def test_typed_submission_rejects_substituted_prediction_identity(tmp_path: Path) -> None:
    expected = PredictionRow(
        row_id='["row-42","u1","v1"]',
        row_identity=("row-42", "u1", "v1"),
        user_id="u1",
        item_id="v1",
        score=0.0,
    )
    substituted = expected.model_copy(update={"item_id": "other-video"})
    with pytest.raises(ValueError, match="identity"):
        KuaiRandPureAdapter().write_prediction_submission(
            tmp_path / "submission.csv", (substituted,), (expected,)
        )


def test_training_view_rejects_manifest_misclassified_held_out_file(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        manifest_id="fixture-v1",
        data_root_env="UNUSED",
        files=(),
        splits={
            "train": DatasetSplit(files=("train.csv",), identity_sha256="0" * 64),
            "valid": DatasetSplit(files=("valid.csv",), identity_sha256="1" * 64),
        },
    )
    disguised_test = DatasetFile.model_construct(
        path="train.csv",
        sha256="0" * 64,
        columns=("row_id", "user_id", "item_id", "label"),
        split="test",
    )
    verified = VerifiedDataset(
        manifest=manifest,
        root=tmp_path,
        verified_splits=("train", "valid"),
        verified_files=(disguised_test,),
    )
    with pytest.raises(ValueError, match="held-out files"):
        authorized_training_view(verified)


def test_manifest_hash_is_canonical_and_changes_when_content_changes() -> None:
    manifest = DatasetManifest(
        manifest_id="fixture-v1",
        data_root_env="UNUSED",
        files=(),
        splits={},
    )
    changed = manifest.model_copy(update={"manifest_id": "fixture-v2"})
    assert canonical_manifest_sha256(manifest) != canonical_manifest_sha256(changed)
