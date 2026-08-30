from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    AuthorizedTrainingView,
    DatasetManifest,
    authorized_training_view,
    canonical_manifest_sha256,
    encode_row_identity,
    load_dataset_manifest,
    verify_dataset_manifest,
)
from tiktok2026.contracts import FullCommitSha


def _read_rows(
    manifest: DatasetManifest, view: AuthorizedTrainingView, split_name: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for file in view.files:
        if file.split != split_name:
            continue
        with (view.host_root / file.path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _stable_score(
    row: dict[str, str], feature_columns: tuple[str, ...], identity_columns: tuple[str, ...]
) -> float:
    if feature_columns:
        values = [float(row[column]) for column in feature_columns]
        return sum(values) / len(values)
    key = "\x1f".join(row[column] for column in identity_columns).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def run_training(
    output_dir: Path,
    seed: int,
    fidelity: str,
    data_manifest: Path,
    source_commit: FullCommitSha,
    execution_id: str,
    data_root: Path,
) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a 40-character hexadecimal commit")
    if fidelity not in {"smoke", "proxy", "full"}:
        raise ValueError("unknown fidelity")
    manifest = load_dataset_manifest(data_manifest)
    verified = verify_dataset_manifest(manifest, data_root, splits={"train", "valid"})
    training_view = authorized_training_view(verified)
    train_rows = _read_rows(manifest, training_view, "train")
    valid_rows = _read_rows(manifest, training_view, "valid")
    if not train_rows or not valid_rows:
        raise ValueError("training manifest must contain nonempty train and valid splits")
    manifest_hash = canonical_manifest_sha256(manifest)
    train_label_rate = sum(int(row[manifest.label_column]) for row in train_rows) / len(train_rows)
    required = (
        set(manifest.row_identity_columns)
        | {manifest.label_column, manifest.user_id_column, manifest.item_id_column}
        | set(manifest.non_label_feature_columns)
    )
    for row in (*train_rows, *valid_rows):
        if not required <= row.keys():
            raise ValueError("dataset row does not match manifest schema")
        if int(row[manifest.label_column]) not in (0, 1):
            raise ValueError("labels must be binary")
        if not math.isfinite(
            _stable_score(row, manifest.non_label_feature_columns, manifest.row_identity_columns)
        ):
            raise ValueError("features must produce finite scores")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_payload = {
        "schema_version": "1",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest_hash,
        "source_commit": source_commit,
        "execution_id": execution_id,
        "split": "valid",
        "rows": [
            {
                "row_id": encode_row_identity(
                    tuple(row[column] for column in manifest.row_identity_columns)
                ),
                "row_identity": [row[column] for column in manifest.row_identity_columns],
                "user_id": row[manifest.user_id_column],
                "item_id": row[manifest.item_id_column],
                "score": (
                    _stable_score(
                        row,
                        manifest.non_label_feature_columns,
                        manifest.row_identity_columns,
                    )
                    + train_label_rate
                )
                / 2.0,
            }
            for row in valid_rows
        ],
    }
    prediction_bytes = _canonical_json(prediction_payload)
    predictions = output_dir / "predictions.json"
    predictions.write_bytes(prediction_bytes)
    prediction_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    config_bytes = _canonical_json({"fidelity": fidelity, "seed": seed})
    identity_material = b"\n".join(
        (
            source_commit.encode(),
            execution_id.encode(),
            manifest_hash.encode(),
            config_bytes,
            prediction_bytes,
        )
    )
    content_identity = hashlib.sha256(identity_material).hexdigest()
    prediction_artifact_id = f"predictions-{content_identity}"
    checkpoint_hash = hashlib.sha256(
        (prediction_artifact_id + content_identity).encode()
    ).hexdigest()
    checkpoint_id = f"checkpoint-{checkpoint_hash}"

    bundle_payload = {
        "schema_version": "1",
        "checkpoint_id": checkpoint_id,
        "data_manifest_id": manifest.manifest_id,
        "seed": seed,
        "source_commit": source_commit,
        "execution_id": execution_id,
        "fidelity": fidelity,
        "prediction_artifact_id": prediction_artifact_id,
        "prediction_artifact": predictions.name,
        "prediction_sha256": prediction_sha256,
    }
    bundle = output_dir / "checkpoint_bundle.json"
    bundle.write_bytes(_canonical_json(bundle_payload))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fidelity", choices=("smoke", "proxy", "full"), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    arguments = parser.parse_args()
    run_training(
        arguments.output_dir,
        arguments.seed,
        arguments.fidelity,
        arguments.data_manifest,
        arguments.source_commit,
        arguments.execution_id,
        arguments.data_root,
    )


if __name__ == "__main__":
    main()
