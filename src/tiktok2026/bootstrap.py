from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    BenchmarkManifest,
    verify_protected_files,
)
from tiktok2026.contracts import RuntimePaths
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository


@dataclass(frozen=True)
class RuntimeServices:
    repository_root: Path
    paths: RuntimePaths
    repository: ApplicationRepository


def initialize_runtime(repository_root: Path, runtime_root: Path) -> RuntimeServices:
    repository_root = repository_root.resolve()
    paths = RuntimePaths.create(repository_root, runtime_root)
    MigrationRunner(paths.application_db, repository_root / "migrations" / "application").apply()
    MigrationRunner(paths.graph_db, repository_root / "migrations" / "graph").apply()
    return RuntimeServices(
        repository_root=repository_root,
        paths=paths,
        repository=ApplicationRepository(paths.application_db),
    )


def verify_manifests(repository_root: Path) -> BenchmarkManifest:
    manifest_path = (
        repository_root / "src" / "tiktok2026" / "benchmark" / "kuaireand_pure" / "manifest.json"
    )
    manifest = BenchmarkManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    verify_protected_files(repository_root, manifest.protected_reference_files)
    return manifest
