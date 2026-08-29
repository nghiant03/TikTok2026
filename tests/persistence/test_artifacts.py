from pathlib import Path

import pytest

from tiktok2026.contracts import ArtifactRetention, RuntimePaths
from tiktok2026.persistence.artifacts import ArtifactStore
from tiktok2026.persistence.repositories import ApplicationRepository


def test_artifact_is_hashed_published_and_registered(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    paths = RuntimePaths.create(repository_root, tmp_path / "runtime")
    repository = ApplicationRepository(paths.application_db)
    repository.initialize()
    store = ArtifactStore(paths, repository)

    record = store.publish_bytes(
        run_id="run-1",
        experiment_id="exp-1",
        kind="predictions",
        filename="predictions.csv",
        content=b"score\n0.5\n",
        producer="executor",
        retention=ArtifactRetention.RUN,
    )

    assert Path(record.uri.removeprefix("file://")).read_bytes() == b"score\n0.5\n"
    assert len(record.sha256) == 64
    assert repository.get_artifact(record.artifact_id) == record


def test_artifact_rejects_traversing_filename(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    paths = RuntimePaths.create(repository_root, tmp_path / "runtime")
    repository = ApplicationRepository(paths.application_db)
    repository.initialize()
    store = ArtifactStore(paths, repository)

    with pytest.raises(ValueError, match="safe filename"):
        store.publish_bytes(
            run_id="run-1",
            experiment_id="exp-1",
            kind="predictions",
            filename="../../escaped.csv",
            content=b"unsafe",
            producer="executor",
            retention=ArtifactRetention.RUN,
        )

    assert not (tmp_path / "runtime" / "artifacts" / "escaped.csv").exists()
