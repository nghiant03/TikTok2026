import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from tiktok2026.experiment.train import run_training


class SplitRow(TypedDict):
    row_id: str
    user_id: str
    item_id: str
    feature: str
    label: str


CsvRow = Mapping[Literal["row_id", "user_id", "item_id", "feature", "label"], Any]


def _write_split(root: Path, name: str, rows: list[SplitRow]) -> tuple[str, str]:
    path = root / f"{name}.csv"
    fields = ("row_id", "user_id", "item_id", "feature", "label")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cast(Iterable[CsvRow], rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = hashlib.sha256(
        "".join(
            json.dumps(
                (row["row_id"], row["user_id"], row["item_id"]), separators=(",", ":")
            )
            + "\n"
            for row in rows
        ).encode()
    ).hexdigest()
    return digest, identity


def test_training_writes_versioned_data_driven_bundle_without_test_evaluation(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    train_rows: list[SplitRow] = [
        {"row_id": "tr1", "user_id": "u1", "item_id": "i1", "feature": "0.2", "label": "1"}
    ]
    valid_rows: list[SplitRow] = [
        {"row_id": "va1", "user_id": "u1", "item_id": "i2", "feature": "0.8", "label": "0"}
    ]
    test_rows: list[SplitRow] = [
        {"row_id": "te1", "user_id": "u9", "item_id": "i9", "feature": "999", "label": "1"}
    ]
    train_hash, train_identity = _write_split(data_root, "train", train_rows)
    valid_hash, valid_identity = _write_split(data_root, "valid", valid_rows)
    test_hash, test_identity = _write_split(data_root, "test", test_rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_id": "fixture-v1",
                "data_root_env": "TEST_DATA_ROOT",
                "row_id_columns": ["row_id", "user_id", "item_id"],
                "label_column": "label",
                "feature_columns": ["feature"],
                "files": [
                    {
                        "path": "train.csv",
                        "sha256": train_hash,
                        "schema": ["row_id", "user_id", "item_id", "feature", "label"],
                        "split": "train",
                    },
                    {
                        "path": "valid.csv",
                        "sha256": valid_hash,
                        "schema": ["row_id", "user_id", "item_id", "feature", "label"],
                        "split": "valid",
                    },
                    {
                        "path": "test.csv",
                        "sha256": test_hash,
                        "schema": ["row_id", "user_id", "item_id", "feature", "label"],
                        "split": "test",
                    },
                ],
                "splits": {
                    "train": {"files": ["train.csv"], "identity_sha256": train_identity},
                    "valid": {"files": ["valid.csv"], "identity_sha256": valid_identity},
                    "test": {"files": ["test.csv"], "identity_sha256": test_identity},
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    bundle = run_training(
        output,
        seed=7,
        fidelity="smoke",
        data_manifest=manifest,
        source_commit="a" * 40,
        execution_id="exec-1",
        data_root=data_root,
    )
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["data_manifest_id"] == "fixture-v1"
    assert payload["source_commit"] == "a" * 40
    assert payload["execution_id"] == "exec-1"
    assert predictions["rows"][0]["row_id"] == '["va1","u1","i2"]'
    assert predictions["rows"][0]["row_identity"] == ["va1", "u1", "i2"]
    assert predictions["rows"][0]["score"] == 0.9
    assert "te1" not in {row["row_id"] for row in predictions["rows"]}
    assert not (output / "test_metrics.json").exists()
