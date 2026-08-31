from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Literal, cast

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    DatasetManifest,
    canonical_manifest_sha256,
    encode_row_identity,
    load_dataset_manifest,
    read_verified_rows,
    verify_dataset_manifest,
)
from tiktok2026.contracts import (
    CURRENT_EVALUATOR_ID,
    BaselineCalibrationRecord,
    DiagnosticMetricValue,
    EvaluationContext,
    EvaluationRequest,
    PredictionArtifactRegistration,
)
from tiktok2026.evaluation.registry import (
    ProvisionalEvaluator,
    evaluator_implementation_sha256,
)

BASELINE_CONFIG: dict[str, object] = {
    "model": "fm",
    "seed": 0,
    "k": 16,
    "lr": 0.001,
    "batch": 8192,
    "max_epochs": 40,
    "patience": 4,
}
BASELINE_FILES = ("baseline.py", "data.py", "evaluate.py", "submit.py")

SubmissionRunner = Callable[[Path, Path, Path], None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _baseline_source_hash(baseline_root: Path) -> str:
    digest = hashlib.sha256()
    for name in BASELINE_FILES:
        path = baseline_root / name
        if not path.is_file():
            raise ValueError(f"Starter Kit source is missing: baseline/{name}")
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _calibration_id(
    manifest: DatasetManifest,
    evaluator_id: str,
    evaluator_sha256: str,
    source_sha256: str,
    config_sha256: str,
) -> str:
    digest = _canonical_hash(
        {
            "dataset_manifest_id": manifest.manifest_id,
            "dataset_manifest_sha256": canonical_manifest_sha256(manifest),
            "evaluator_id": evaluator_id,
            "evaluator_sha256": evaluator_sha256,
            "baseline_source_sha256": source_sha256,
            "config_sha256": config_sha256,
            "split": "valid",
        }
    )
    return f"baseline-calibration-{digest}"


def _prepare_starter_data(dataset_root: Path, adapter_root: Path) -> None:
    adapter_root.mkdir(parents=True, exist_ok=True)
    links = {
        "log_standard_4_08_to_4_21_pure.csv": dataset_root / "train.csv",
        "log_standard_4_22_to_5_08_pure.csv": dataset_root / "valid.csv",
    }
    for name, target in links.items():
        link = adapter_root / name
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"Starter Kit data adapter changed identity: {link}")
            continue
        link.symlink_to(target.resolve())

    video_path = adapter_root / "video_features_basic_pure.csv"
    if video_path.is_file():
        return
    videos: dict[str, str] = {}
    for source in (dataset_root / "train.csv", dataset_root / "valid.csv"):
        with source.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                video_id = row["video_id"]
                author_id = row["author_id"]
                existing = videos.setdefault(video_id, author_id)
                if existing != author_id:
                    raise ValueError(f"video has inconsistent author identity: {video_id}")
    temporary = video_path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("video_id", "author_id"))
        writer.writerows(
            (video_id, videos[video_id]) for video_id in sorted(videos)
        )
    temporary.replace(video_path)


def _run_starter_submission(baseline_root: Path, adapter_root: Path, output: Path) -> None:
    subprocess.run(
        (
            sys.executable,
            str(baseline_root / "submit.py"),
            "--make",
            "--split",
            "valid",
            "--data_dir",
            str(adapter_root),
            str(output),
        ),
        cwd=baseline_root,
        check=True,
        timeout=1800,
    )


def _read_submission(path: Path, rows: tuple[dict[str, str], ...]) -> tuple[float, ...]:
    scores: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row_id", "user_id", "video_id", "score"]:
            raise ValueError("Starter Kit submission header is invalid")
        for index, (submission, row) in enumerate(zip(reader, rows, strict=True)):
            if (
                submission["row_id"] != str(index)
                or submission["user_id"] != row["user_id"]
                or submission["video_id"] != row["video_id"]
            ):
                raise ValueError(f"Starter Kit submission is misaligned at row {index}")
            score = float(submission["score"])
            if not (-float("inf") < score < float("inf")):
                raise ValueError(f"Starter Kit submission has a non-finite score at row {index}")
            scores.append(score)
    if len(scores) != len(rows):
        raise ValueError("Starter Kit submission row count does not match validation data")
    return tuple(scores)


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tiktok2026_starter_evaluate", path)
    if spec is None or spec.loader is None:
        raise ValueError("Starter Kit evaluator could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic_metrics(
    baseline_root: Path, rows: tuple[dict[str, str], ...], scores: tuple[float, ...]
) -> tuple[DiagnosticMetricValue, ...]:
    evaluator = _load_module(baseline_root / "evaluate.py")
    evaluate = getattr(evaluator, "evaluate", None)
    if not callable(evaluate):
        raise ValueError("Starter Kit evaluator has no evaluate function")
    raw = cast(
        Mapping[str, object],
        evaluate(
            [row["user_id"] for row in rows],
            [int(row["label"]) for row in rows],
            list(scores),
        ),
    )
    names: tuple[Literal["GAUC", "nDCG@5", "primary"], ...] = (
        "GAUC",
        "nDCG@5",
        "primary",
    )
    metrics: list[DiagnosticMetricValue] = []
    for name in names:
        value = raw[name]
        if not isinstance(value, (int, float)):
            raise ValueError(f"Starter Kit diagnostic metric is not numeric: {name}")
        metrics.append(DiagnosticMetricValue(name=name, value=float(value)))
    return tuple(metrics)


def _repository_commit(repository_root: Path) -> str:
    commit = subprocess.run(
        ("git", "rev-parse", "@^{commit}"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("repository commit identity is invalid")
    return commit


def load_calibration(
    records: tuple[str, ...], calibration_id: str
) -> BaselineCalibrationRecord | None:
    for payload in records:
        record = BaselineCalibrationRecord.model_validate_json(payload)
        if record.calibration_id == calibration_id:
            artifact = Path(record.prediction_artifact_uri.removeprefix("file://"))
            if not artifact.is_file() or _sha256(artifact) != record.prediction_sha256:
                raise ValueError("persisted baseline calibration artifact is missing or changed")
            return record
    return None


def calibrate_baseline(
    repository_root: Path,
    runtime_root: Path,
    dataset_root: Path,
    existing_records: tuple[str, ...] = (),
    submission_runner: SubmissionRunner = _run_starter_submission,
) -> tuple[BaselineCalibrationRecord, bool]:
    repository_root = repository_root.resolve()
    runtime_root = runtime_root.resolve()
    dataset_root = dataset_root.resolve()
    baseline_root = repository_root / "baseline"
    manifest = load_dataset_manifest(dataset_root / "manifest.json")
    verified = verify_dataset_manifest(manifest, dataset_root, splits={"train", "valid"})
    evaluator_id = CURRENT_EVALUATOR_ID
    evaluator_sha256 = evaluator_implementation_sha256()
    source_sha256 = _baseline_source_hash(baseline_root)
    config_sha256 = _canonical_hash(BASELINE_CONFIG)
    calibration_id = _calibration_id(
        manifest, evaluator_id, evaluator_sha256, source_sha256, config_sha256
    )
    existing = load_calibration(existing_records, calibration_id)
    if existing is not None:
        return existing, False

    calibration_root = runtime_root / "calibrations" / "baseline" / calibration_id
    calibration_root.mkdir(parents=True, exist_ok=True)
    adapter_root = calibration_root / "starter-data"
    submission_path = calibration_root / "submission-valid.csv"
    _prepare_starter_data(dataset_root, adapter_root)
    submission_runner(baseline_root, adapter_root, submission_path)

    rows = read_verified_rows(verified, "valid")
    scores = _read_submission(submission_path, rows)
    source_commit = _repository_commit(repository_root)
    execution_id = calibration_id
    manifest_sha256 = canonical_manifest_sha256(manifest)
    prediction_payload = {
        "schema_version": "1",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest_sha256,
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
                "score": score,
            }
            for row, score in zip(rows, scores, strict=True)
        ],
    }
    prediction_path = calibration_root / "predictions.json"
    prediction_bytes = (
        json.dumps(prediction_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    prediction_path.write_bytes(prediction_bytes)
    prediction_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    prediction = PredictionArtifactRegistration(
        artifact_id=f"baseline-predictions-{prediction_sha256}",
        path=prediction_path,
        sha256=prediction_sha256,
        checkpoint_id=f"baseline-checkpoint-{config_sha256}",
        source_commit=source_commit,
        execution_id=execution_id,
        dataset_manifest_id=manifest.manifest_id,
        dataset_manifest_sha256=manifest_sha256,
        split="valid",
    )
    evaluator = ProvisionalEvaluator(
        evaluator_id,
        artifacts={prediction.artifact_id: prediction},
        datasets={manifest.manifest_id: verified},
    )
    context = EvaluationContext(
        run_id=calibration_id,
        evaluation_id=f"evaluation-{calibration_id}",
        experiment_id="starter-kit-fm-baseline",
        checkpoint_id=prediction.checkpoint_id,
        source_commit=source_commit,
        execution_id=execution_id,
        dataset_manifest_id=manifest.manifest_id,
        dataset_manifest_sha256=manifest_sha256,
        split="valid",
        prediction_artifact_id=prediction.artifact_id,
        prediction_sha256=prediction_sha256,
        evaluator_id=evaluator_id,
        evaluator_sha256=evaluator_sha256,
    )
    evaluation = evaluator.evaluate(
        EvaluationRequest(evaluation_id=context.evaluation_id, context=context)
    )
    record = BaselineCalibrationRecord(
        calibration_id=calibration_id,
        dataset_manifest_id=manifest.manifest_id,
        dataset_manifest_sha256=manifest_sha256,
        evaluator_id=evaluator_id,
        evaluator_sha256=evaluator_sha256,
        baseline_source_sha256=source_sha256,
        config_sha256=config_sha256,
        prediction_sha256=prediction_sha256,
        prediction_artifact_uri=prediction_path.as_uri(),
        evaluation=evaluation,
        diagnostic_metrics=_diagnostic_metrics(baseline_root, rows, scores),
    )
    return record, True
