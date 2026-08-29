from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


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


def verify_protected_files(repository_root: Path, expected: dict[str, str]) -> None:
    for relative, digest in expected.items():
        path = repository_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != digest:
            raise ValueError(f"protected file hash mismatch: {relative}")
