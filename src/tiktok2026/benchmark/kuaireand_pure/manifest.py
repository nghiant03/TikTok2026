from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class BenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str
    data_root_env: str
    data_access: Literal["read-only"]
    task: str
    label: str
    splits: dict[str, tuple[int, int]]
    judging_metrics: tuple[Literal["NDCG@10", "Recall@50"], ...]
    judging_evaluator_status: Literal["provisional", "official"]
    validation_ranking: str
    convergence: dict[str, float | int]
    protected_reference_files: dict[str, str]


class DatasetFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    columns: tuple[str, ...] = Field(validation_alias="schema", serialization_alias="schema")
    split: Literal["train", "valid", "test"]


class DatasetSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    files: tuple[str, ...]
    identity_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DatasetManifest(BaseModel):
    """Description of external, immutable tabular benchmark inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    manifest_id: str
    data_root_env: str
    manifest_sha256: str | None = None
    row_identity_encoding: Literal["json-array-v1"] = "json-array-v1"
    row_identity_columns: tuple[str, ...] = Field(
        default=("row_id", "user_id", "item_id"),
        validation_alias=AliasChoices("row_identity_columns", "row_id_columns"),
        serialization_alias="row_identity_columns",
    )
    user_id_column: str = "user_id"
    item_id_column: str = "item_id"
    label_column: str = "label"
    non_label_feature_columns: tuple[str, ...] = Field(
        default=(),
        validation_alias=AliasChoices("non_label_feature_columns", "feature_columns"),
        serialization_alias="non_label_feature_columns",
    )
    files: tuple[DatasetFile, ...]
    splits: dict[str, DatasetSplit]

    @model_validator(mode="after")
    def validate_columns(self) -> DatasetManifest:
        if not self.row_identity_columns:
            raise ValueError("row identity must contain at least one column")
        if self.label_column in self.non_label_feature_columns:
            raise ValueError("label column cannot be a non-label feature")
        if self.label_column in self.row_identity_columns:
            raise ValueError("label column cannot be part of row identity")
        if self.user_id_column == self.label_column or self.item_id_column == self.label_column:
            raise ValueError("user/item columns cannot be the label column")
        if not {self.user_id_column, self.item_id_column} <= set(self.row_identity_columns):
            raise ValueError("row identity must include user and item columns")
        return self


class VerifiedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    manifest: DatasetManifest
    root: Path
    verified_splits: tuple[str, ...]
    verified_files: tuple[DatasetFile, ...] = ()

    @property
    def manifest_sha256(self) -> str:
        return canonical_manifest_sha256(self.manifest)

    def training_view(self) -> AuthorizedTrainingView:
        return authorized_training_view(self)


class AuthorizedTrainingView(BaseModel):
    """Immutable train/valid-only dataset view for a later executor mount."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    manifest_id: str
    manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    host_root: Path
    container_root: Literal["/dataset"] = "/dataset"
    files: tuple[DatasetFile, ...]

    @property
    def container_paths(self) -> tuple[Path, ...]:
        return tuple(Path(self.container_root) / file.path for file in self.files)


def authorized_training_view(verified: VerifiedDataset) -> AuthorizedTrainingView:
    if set(verified.verified_splits) != {"train", "valid"}:
        raise ValueError("training view requires exactly verified train and valid splits")
    files = tuple(verified.verified_files)
    if not files or {file.split for file in files} != {"train", "valid"}:
        raise ValueError("training view contains undisclosed or held-out files")
    declared = {
        path for split in ("train", "valid") for path in verified.manifest.splits[split].files
    }
    if {file.path for file in files} != declared:
        raise ValueError("training view files do not match manifest identity")
    if any(file.split == "test" for file in files):
        raise ValueError("training view cannot contain test files")
    return AuthorizedTrainingView(
        manifest_id=verified.manifest.manifest_id,
        manifest_sha256=verified.manifest_sha256,
        host_root=verified.root,
        files=files,
    )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def canonical_manifest_bytes(manifest: DatasetManifest) -> bytes:
    payload = manifest.model_dump(
        mode="json", by_alias=True, exclude={"manifest_sha256"}
    )
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_manifest_sha256(manifest: DatasetManifest) -> str:
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def encode_row_identity(values: tuple[str, ...]) -> str:
    return json.dumps(values, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_identity(rows: Iterable[dict[str, str]], columns: tuple[str, ...]) -> str:
    payload = "".join(
        encode_row_identity(tuple(row[column] for column in columns))
        + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_dataset_manifest(
    manifest: DatasetManifest,
    data_root: Path | None = None,
    *,
    splits: set[str] | None = None,
) -> VerifiedDataset:
    """Verify declared files and identities without trusting manifest paths."""

    configured_value = (
        str(data_root) if data_root is not None else os.environ.get(manifest.data_root_env)
    )
    if not configured_value:
        raise ValueError(f"dataset root environment variable is not set: {manifest.data_root_env}")
    root = Path(configured_value).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root does not exist: {root}")
    selected = set(manifest.splits) if splits is None else set(splits)
    if manifest.manifest_sha256 and manifest.manifest_sha256 != canonical_manifest_sha256(manifest):
        raise ValueError("dataset manifest hash mismatch")
    unknown = selected - set(manifest.splits)
    if unknown:
        raise ValueError(f"unknown dataset split: {sorted(unknown)[0]}")
    by_path = {file.path: file for file in manifest.files}
    if len(by_path) != len(manifest.files):
        raise ValueError("dataset manifest contains duplicate file paths")
    referenced_paths = {path for split in manifest.splits.values() for path in split.files}
    if referenced_paths != by_path.keys():
        raise ValueError("dataset manifest has unassigned files")
    for split_name, split in manifest.splits.items():
        if not set(split.files) <= by_path.keys():
            raise ValueError(f"split references undeclared file: {split_name}")
        if any(by_path[path].split != split_name for path in split.files):
            raise ValueError(f"file split does not match split declaration: {split_name}")
    split_rows: dict[str, list[dict[str, str]]] = {split: [] for split in selected}
    for file in manifest.files:
        path = (root / file.path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"dataset file escapes root: {file.path}") from error
        if file.split not in selected:
            continue
        if not path.is_file():
            raise ValueError(f"dataset file is missing: {file.path}")
        if _sha256(path) != file.sha256:
            raise ValueError(f"dataset file hash mismatch: {file.path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != file.columns:
                raise ValueError(f"dataset schema mismatch: {file.path}")
            required_columns = (
                set(manifest.row_identity_columns)
                | {manifest.label_column}
                | {manifest.user_id_column, manifest.item_id_column}
                | set(manifest.non_label_feature_columns)
            )
            if not required_columns <= set(file.columns):
                raise ValueError(f"dataset schema lacks required columns: {file.path}")
            split_rows[file.split].extend(reader)
    for split_name, rows in split_rows.items():
        identities = [
            tuple(row[column] for column in manifest.row_identity_columns) for row in rows
        ]
        if len(set(identities)) != len(identities):
            raise ValueError(f"dataset split contains duplicate row identities: {split_name}")
        if (
            _split_identity(rows, manifest.row_identity_columns)
            != manifest.splits[split_name].identity_sha256
        ):
            raise ValueError(f"dataset split identity mismatch: {split_name}")
    verified_files = tuple(file for file in manifest.files if file.split in selected)
    return VerifiedDataset(
        manifest=manifest,
        root=root,
        verified_splits=tuple(sorted(selected)),
        verified_files=verified_files,
    )


def read_verified_rows(verified: VerifiedDataset, split_name: str) -> tuple[dict[str, str], ...]:
    if split_name not in verified.verified_splits:
        raise ValueError(f"split was not verified: {split_name}")
    rows: list[dict[str, str]] = []
    for file in verified.verified_files:
        if file.split != split_name:
            continue
        with (verified.root / file.path).open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return tuple(rows)


def verify_protected_files(repository_root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = repository_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != digest:
            raise ValueError(f"protected file hash mismatch: {relative}")
