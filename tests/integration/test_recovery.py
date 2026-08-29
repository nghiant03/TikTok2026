import hashlib
import subprocess
from pathlib import Path

from tiktok2026.contracts import WorktreeAssignment
from tiktok2026.recovery import (
    RecoveryCandidate,
    reconcile_recovery,
    validate_pre_registration_assignment,
)


def test_recovery_releases_stale_state_when_all_identities_agree(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    lock.write_text("run-1", encoding="utf-8")
    candidate = RecoveryCandidate(
        run_id="run-1",
        experiment_id="experiment-1",
        database_source_commit="a" * 40,
        worktree_source_commit="a" * 40,
        database_artifact_sha256="b" * 64,
        artifact_sha256="b" * 64,
        stale_lock=lock,
        stale_reservation_id="reservation-1",
    )
    released: list[str] = []

    result = reconcile_recovery(
        candidate, lambda reservation_id: released.append(reservation_id) or True
    )

    assert result.resumable
    assert result.released_reservation_id == "reservation-1"
    assert not lock.exists()
    assert released == ["reservation-1"]


def test_recovery_preserves_state_and_blocks_on_identity_mismatch(tmp_path: Path) -> None:
    lock = tmp_path / "run.lock"
    lock.write_text("run-1", encoding="utf-8")
    candidate = RecoveryCandidate(
        run_id="run-1",
        experiment_id="experiment-1",
        database_source_commit="a" * 40,
        worktree_source_commit="c" * 40,
        database_artifact_sha256="b" * 64,
        artifact_sha256="b" * 64,
        stale_lock=lock,
        stale_reservation_id="reservation-1",
    )
    released: list[str] = []

    result = reconcile_recovery(
        candidate, lambda reservation_id: released.append(reservation_id) or True
    )

    assert not result.resumable
    assert result.reason == "source identity mismatch"
    assert lock.exists()
    assert released == []


def test_pre_registration_recovery_validates_assignment_without_source_artifact(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "test"), cwd=repository, check=True)
    (repository / "README").write_text("base\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=repository, check=True)
    parent = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    runtime = tmp_path / "runtime"
    worktree = runtime / "worktrees" / "run-1" / "exp-1"
    worktree.parent.mkdir(parents=True)
    subprocess.run(
        ("git", "worktree", "add", "-qb", "experiment/run-1/exp-1", str(worktree), parent),
        cwd=repository,
        check=True,
    )
    assignment = WorktreeAssignment(
        worktree_id="wt-1",
        run_id="run-1",
        experiment_id="exp-1",
        path=worktree,
        branch="experiment/run-1/exp-1",
        parent_commit=parent,
    )

    result = validate_pre_registration_assignment(
        assignment, runtime, lambda value: value == parent
    )

    assert result.resumable
    assert result.reason == "pre-registration identities verified"


def test_recovery_rejects_missing_or_forged_artifact_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "patch.diff"
    artifact.write_text("forged\n", encoding="utf-8")
    candidate = RecoveryCandidate(
        run_id="run-1",
        experiment_id="experiment-1",
        database_source_commit="a" * 40,
        worktree_source_commit="a" * 40,
        database_artifact_sha256="b" * 64,
        artifact_sha256="b" * 64,
        stale_lock=tmp_path / "missing.lock",
        artifact_uri=artifact,
    )

    forged = reconcile_recovery(candidate, lambda _: True)
    assert not forged.resumable
    assert forged.reason == "artifact identity mismatch"

    missing = candidate.model_copy(update={"artifact_uri": tmp_path / "missing.diff"})
    missing_result = reconcile_recovery(missing, lambda _: True)
    assert not missing_result.resumable
    assert missing_result.reason == "registered artifact could not be observed"


def test_recovery_source_mismatch_with_real_worktree_preserves_lock_and_reservation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"), cwd=repository, check=True
    )
    subprocess.run(("git", "config", "user.name", "test"), cwd=repository, check=True)
    (repository / "README").write_text("base\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(("git", "commit", "-qm", "base"), cwd=repository, check=True)
    parent = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    runtime = tmp_path / "runtime"
    worktree = runtime / "worktrees" / "run-1" / "exp-1"
    worktree.parent.mkdir(parents=True)
    subprocess.run(
        ("git", "worktree", "add", "-qb", "experiment/run-1/exp-1", str(worktree), parent),
        cwd=repository,
        check=True,
    )
    (worktree / "README").write_text("changed\n", encoding="utf-8")
    subprocess.run(("git", "add", "README"), cwd=worktree, check=True)
    subprocess.run(("git", "commit", "-qm", "changed"), cwd=worktree, check=True)
    changed_head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=worktree, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert changed_head != parent
    artifact = tmp_path / "patch.diff"
    artifact.write_bytes(b"verified patch")
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lock = runtime / "locks" / "run-1.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("run-1", encoding="utf-8")
    released: list[str] = []
    candidate = RecoveryCandidate(
        run_id="run-1",
        experiment_id="exp-1",
        database_source_commit=parent,
        # Deliberately claim the approved commit.  Only the actual Git
        # observation through worktree_path should discover the changed HEAD.
        worktree_source_commit=parent,
        database_artifact_sha256=artifact_sha256,
        artifact_sha256=artifact_sha256,
        stale_lock=lock,
        stale_reservation_id="reservation-1",
        worktree_path=worktree,
        artifact_uri=artifact,
    )

    result = reconcile_recovery(
        candidate, lambda reservation_id: released.append(reservation_id) or True
    )

    assert not result.resumable
    assert result.reason == "source identity mismatch"
    assert lock.read_text(encoding="utf-8") == "run-1"
    assert released == []
