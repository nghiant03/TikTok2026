from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from tiktok2026.benchmark.kuaireand_pure.calibration import calibrate_baseline
from tiktok2026.benchmark.kuaireand_pure.manifest import (
    DatasetFile,
    DatasetManifest,
    DatasetSplit,
    encode_row_identity,
)

COLUMNS = (
    "row_id",
    "item_id",
    "label",
    "author_id",
    "user_id",
    "video_id",
    "date",
    "duration_ms",
    "tab",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(rows: list[dict[str, str]]) -> str:
    payload = "".join(
        encode_row_identity((row["row_id"], row["user_id"], row["item_id"])) + "\n"
        for row in rows
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _dataset(root: Path) -> None:
    train = [
        {
            "row_id": "0",
            "item_id": "10",
            "label": "1",
            "author_id": "100",
            "user_id": "1",
            "video_id": "10",
            "date": "20220410",
            "duration_ms": "1000",
            "tab": "0",
        },
        {
            "row_id": "1",
            "item_id": "11",
            "label": "0",
            "author_id": "101",
            "user_id": "1",
            "video_id": "11",
            "date": "20220411",
            "duration_ms": "2000",
            "tab": "1",
        },
    ]
    valid = [
        {
            "row_id": "0",
            "item_id": "10",
            "label": "1",
            "author_id": "100",
            "user_id": "1",
            "video_id": "10",
            "date": "20220422",
            "duration_ms": "1000",
            "tab": "0",
        },
        {
            "row_id": "1",
            "item_id": "11",
            "label": "0",
            "author_id": "101",
            "user_id": "1",
            "video_id": "11",
            "date": "20220422",
            "duration_ms": "2000",
            "tab": "1",
        },
    ]
    _write_rows(root / "train.csv", train)
    _write_rows(root / "valid.csv", valid)
    manifest = DatasetManifest(
        manifest_id="calibration-fixture",
        data_root_env="UNUSED",
        files=(
            DatasetFile(
                path="train.csv", sha256=_sha256(root / "train.csv"), schema=COLUMNS, split="train"
            ),
            DatasetFile(
                path="valid.csv", sha256=_sha256(root / "valid.csv"), schema=COLUMNS, split="valid"
            ),
        ),
        splits={
            "train": DatasetSplit(files=("train.csv",), identity_sha256=_identity(train)),
            "valid": DatasetSplit(files=("valid.csv",), identity_sha256=_identity(valid)),
        },
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True)), encoding="utf-8"
    )


def test_calibration_is_validation_only_and_identity_cached(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    runtime = tmp_path / "runtime"
    dataset.mkdir()
    _dataset(dataset)

    def runner(_: Path, adapter: Path, output: Path) -> None:
        assert (adapter / "log_standard_4_08_to_4_21_pure.csv").resolve() == (
            dataset / "train.csv"
        )
        assert (adapter / "log_standard_4_22_to_5_08_pure.csv").resolve() == (
            dataset / "valid.csv"
        )
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            writer.writerow((0, 1, 10, 0.9))
            writer.writerow((1, 1, 11, 0.1))

    repository = Path(__file__).parents[2]
    record, created = calibrate_baseline(
        repository, runtime, dataset, submission_runner=runner
    )

    assert created is True
    assert record.split == "valid"
    assert record.evaluation.validation_score == 1.0
    assert {metric.name for metric in record.diagnostic_metrics} == {
        "GAUC",
        "nDCG@5",
        "primary",
    }

    def must_not_run(_: Path, __: Path, ___: Path) -> None:
        raise AssertionError("cached calibration reran the Starter Kit")

    cached, created_again = calibrate_baseline(
        repository,
        runtime,
        dataset,
        existing_records=(record.model_dump_json(),),
        submission_runner=must_not_run,
    )
    assert created_again is False
    assert cached == record
