from pathlib import Path

from tiktok2026.recovery import RecoveryCandidate, reconcile_recovery


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

    result = reconcile_recovery(candidate, released.append)

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

    result = reconcile_recovery(candidate, released.append)

    assert not result.resumable
    assert result.reason == "source identity mismatch"
    assert lock.exists()
    assert released == []
