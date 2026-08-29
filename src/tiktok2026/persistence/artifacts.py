from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from tiktok2026.contracts import ArtifactRecord, ArtifactRetention, RuntimePaths
from tiktok2026.persistence.repositories import ApplicationRepository


class ArtifactStore:
    def __init__(self, paths: RuntimePaths, repository: ApplicationRepository) -> None:
        self.paths = paths
        self.repository = repository

    def publish_bytes(
        self,
        run_id: str,
        experiment_id: str | None,
        kind: str,
        filename: str,
        content: bytes,
        producer: str,
        retention: ArtifactRetention,
    ) -> ArtifactRecord:
        if Path(filename).name != filename or filename in {"", ".", ".."}:
            raise ValueError("artifact filename must be a safe filename")
        for identity in (run_id, experiment_id):
            if identity is not None and (
                Path(identity).name != identity or identity in {"", ".", ".."}
            ):
                raise ValueError("artifact identity must be a safe identifier")
        artifact_id = f"artifact-{uuid.uuid4().hex}"
        temporary = self.paths.temporary / f"{artifact_id}.tmp"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(content).hexdigest()
        destination = (
            self.paths.artifacts / run_id / (experiment_id or "run") / artifact_id / filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            run_id=run_id,
            experiment_id=experiment_id,
            kind=kind,
            uri=destination.resolve().as_uri(),
            sha256=digest,
            size_bytes=len(content),
            producer=producer,
            retention=retention,
        )
        self.repository.register_artifact(record)
        return record

    def read(self, record: ArtifactRecord) -> bytes:
        path = Path(record.uri.removeprefix("file://"))
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record.sha256:
            raise ValueError("artifact checksum mismatch")
        return content
