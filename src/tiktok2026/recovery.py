from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RecoveryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    experiment_id: str
    database_source_commit: str
    worktree_source_commit: str
    database_artifact_sha256: str
    artifact_sha256: str
    stale_lock: Path
    stale_reservation_id: str | None = None


class RecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resumable: bool
    reason: str
    released_reservation_id: str | None = None


def reconcile_recovery(
    candidate: RecoveryCandidate, release_reservation: Callable[[str], None]
) -> RecoveryResult:
    if candidate.database_source_commit != candidate.worktree_source_commit:
        return RecoveryResult(resumable=False, reason="source identity mismatch")
    if candidate.database_artifact_sha256 != candidate.artifact_sha256:
        return RecoveryResult(resumable=False, reason="artifact identity mismatch")
    if candidate.stale_reservation_id is not None:
        release_reservation(candidate.stale_reservation_id)
    candidate.stale_lock.unlink(missing_ok=True)
    return RecoveryResult(
        resumable=True,
        reason="identities verified",
        released_reservation_id=candidate.stale_reservation_id,
    )
