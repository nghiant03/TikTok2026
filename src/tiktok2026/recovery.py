from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tiktok2026.contracts import ExecutionResult, SourceRegistration, WorktreeAssignment


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
    worktree_path: Path | None = None
    artifact_uri: Path | None = None


class RecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resumable: bool
    reason: str
    released_reservation_id: str | None = None


def validate_execution_recovery_state(
    state: Mapping[str, object],
    run_id: str,
    execution: ExecutionResult,
    registration: SourceRegistration,
) -> RecoveryResult:
    """Verify that a historical execute checkpoint owns a persisted result."""

    experiment_id = state.get("current_experiment_id")
    state_version = state.get("state_version")
    expected_execution_id = f"execution-{run_id}-{experiment_id}-{state_version}"
    if state.get("pending_route") != "execute" or str(state.get("phase")) not in {
        "execute",
        "RunPhase.EXECUTE",
    }:
        return RecoveryResult(resumable=False, reason="checkpoint is not pending execution")
    if execution.execution_id != expected_execution_id:
        return RecoveryResult(resumable=False, reason="execution checkpoint identity mismatch")
    if (
        execution.experiment_id != experiment_id
        or registration.experiment_id != experiment_id
        or execution.source_registration_id != registration.registration_id
        or execution.source_commit != registration.source_commit
        or registration.run_id != run_id
    ):
        return RecoveryResult(resumable=False, reason="execution source identity mismatch")
    return RecoveryResult(resumable=True, reason="execution result identities verified")


def validate_pre_registration_assignment(
    assignment: WorktreeAssignment,
    runtime_root: Path,
    approved_parent_validator: Callable[[str], bool],
    prior_source_commit: str | None = None,
    allow_pending_commit: bool = False,
) -> RecoveryResult:
    """Validate a pending worktree before source or patch registration exists."""
    expected_path = (
        runtime_root.resolve() / "worktrees" / assignment.run_id / assignment.experiment_id
    )
    if assignment.path.resolve() != expected_path:
        return RecoveryResult(resumable=False, reason="worktree assignment path mismatch")
    if not approved_parent_validator(assignment.parent_commit):
        return RecoveryResult(resumable=False, reason="worktree parent is not approved")
    try:
        top_level = _git_output(assignment.path, ("rev-parse", "--show-toplevel"))
        head = _git_output(assignment.path, ("rev-parse", "HEAD"))
        branch = _git_output(assignment.path, ("rev-parse", "--abbrev-ref", "HEAD"))
        parent = _git_output(
            assignment.path, ("rev-parse", f"{assignment.parent_commit}^{{commit}}")
        )
    except ValueError as error:
        return RecoveryResult(resumable=False, reason=str(error))
    if top_level != str(assignment.path.resolve()):
        return RecoveryResult(resumable=False, reason="assigned worktree root mismatch")
    expected_head = prior_source_commit or parent
    if head != expected_head:
        if not allow_pending_commit:
            return RecoveryResult(resumable=False, reason="worktree HEAD identity mismatch")
        try:
            _git_output(assignment.path, ("merge-base", "--is-ancestor", expected_head, head))
            count = _git_output(
                assignment.path, ("rev-list", "--count", f"{expected_head}..{head}")
            )
            _, clean = _observe_worktree(assignment.path)
        except ValueError as error:
            return RecoveryResult(resumable=False, reason=str(error))
        if count != "1" or not clean:
            return RecoveryResult(
                resumable=False, reason="pending source commit is not recoverable"
            )
    if branch != assignment.branch:
        return RecoveryResult(resumable=False, reason="worktree branch identity mismatch")
    return RecoveryResult(resumable=True, reason="pre-registration identities verified")


def reconcile_recovery(
    candidate: RecoveryCandidate, release_reservation: Callable[[str], bool | None]
) -> RecoveryResult:
    worktree_commit = candidate.worktree_source_commit
    if candidate.worktree_path is not None:
        try:
            worktree_commit, clean = _observe_worktree(candidate.worktree_path)
        except ValueError as error:
            return RecoveryResult(resumable=False, reason=str(error))
        if not clean:
            return RecoveryResult(resumable=False, reason="worktree is dirty")
    if candidate.database_source_commit != worktree_commit:
        return RecoveryResult(resumable=False, reason="source identity mismatch")
    artifact_commit = candidate.artifact_sha256
    if candidate.artifact_uri is not None:
        try:
            artifact_commit = _hash_file(candidate.artifact_uri)
        except ValueError as error:
            return RecoveryResult(resumable=False, reason=str(error))
    if candidate.database_artifact_sha256 != artifact_commit:
        return RecoveryResult(resumable=False, reason="artifact identity mismatch")
    if candidate.stale_lock.is_file():
        try:
            lock_owner = candidate.stale_lock.read_text(encoding="utf-8").strip()
        except OSError:
            return RecoveryResult(resumable=False, reason="runtime lock could not be observed")
        if lock_owner and lock_owner != candidate.run_id:
            return RecoveryResult(resumable=False, reason="runtime lock identity mismatch")
    if candidate.stale_reservation_id is not None:
        try:
            released = release_reservation(candidate.stale_reservation_id)
        except Exception:
            return RecoveryResult(resumable=False, reason="reservation release failed")
        if released is not True:
            return RecoveryResult(resumable=False, reason="reservation was not released")
    candidate.stale_lock.unlink(missing_ok=True)
    return RecoveryResult(
        resumable=True,
        reason="identities verified",
        released_reservation_id=candidate.stale_reservation_id,
    )


def _observe_worktree(path: Path) -> tuple[str, bool]:
    try:
        commit = _git_output(path, ("rev-parse", "@^{commit}"))
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("registered worktree could not be observed") from error
    return commit, not status


def _git_output(path: Path, arguments: tuple[str, ...]) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("registered worktree could not be observed") from error


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError("registered artifact could not be observed") from error
    return digest.hexdigest()
