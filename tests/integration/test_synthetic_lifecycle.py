import json
import sqlite3
from pathlib import Path

import pytest

from tiktok2026.adapters import RepositoryExportService
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.testing import run_synthetic_lifecycle


@pytest.mark.asyncio
async def test_two_cycles_persist_audit_and_provisional_bundle(tmp_path: Path) -> None:
    result = await run_synthetic_lifecycle(2, runtime_root=tmp_path / "runtime")

    assert len(result.experiment_ids) == 2
    assert len(set(result.experiment_ids)) == 2
    assert all(score == pytest.approx(1.0) for score in result.scores)
    assert result.terminal_reason == "plateau"
    assert result.finalization.validity == "provisional"
    assert result.exports.jsonl.exists()
    assert result.exports.markdown.exists()
    assert result.exports.jsonl.read_bytes() == result.exports.jsonl_bytes
    assert result.exports.markdown.read_bytes() == result.exports.markdown_bytes

    repository = ApplicationRepository(result.paths.application_db)
    assert tuple(
        repository.get_experiment(experiment_id) is not None
        for experiment_id in result.experiment_ids
    ) == (True, True)
    assert len(repository.list_audit_events(result.run_id)) >= 20
    assert len(repository.list_json("evaluation")) == 2
    assert repository.get_finalization(result.finalization.finalization_id) == result.finalization
    assert result.finalization.consumed_test_access is False
    assert len(repository.list_json("frontier_observation")) == 2
    assert len(repository.list_json("frontier_decision")) == 2
    with sqlite3.connect(result.paths.application_db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM authority_resource_reservations"
        ).fetchone()[0] == 2
        reservations = connection.execute(
            "SELECT status, reservation_json, settled_usage_json "
            "FROM authority_resource_reservations"
        ).fetchall()
        assert len(reservations) == 2
        assert all(row[0] == "consumed" for row in reservations)
        for status, reservation_json, settled_usage_json in reservations:
            assert status == "consumed"
            assert settled_usage_json is not None
            reservation = json.loads(reservation_json)
            usage = json.loads(settled_usage_json)
            for resource in ("gpu_hours", "wall_seconds", "tokens", "disk_bytes"):
                assert 0 <= usage[resource] <= reservation[resource]
        resource_state = json.loads(
            connection.execute(
                "SELECT payload_json FROM authority_resource_state WHERE id = 1"
            ).fetchone()[0]
        )
        assert resource_state["reserved_final_gpu_hours"] == pytest.approx(10.0)
        assert connection.execute(
            "SELECT final_test_claimed FROM runs WHERE run_id = ?", (result.run_id,)
        ).fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM final_test_claims").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM final_test_completions").fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM audit_events WHERE event_type = 'final_test_completed'"
        ).fetchone()[0] == 0
    with sqlite3.connect(result.paths.graph_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM graph_checkpoints").fetchone()[0] > 2

    rerun = await RepositoryExportService(repository, result.paths.root).export_run(
        result.run_id, result.paths.temporary / "determinism-test" / result.run_id
    )
    rerun_again = await RepositoryExportService(repository, result.paths.root).export_run(
        result.run_id, result.paths.temporary / "determinism-test-again" / result.run_id
    )
    assert rerun["jsonl"].read_bytes() == rerun_again["jsonl"].read_bytes()
    assert rerun["markdown"].read_bytes() == rerun_again["markdown"].read_bytes()
    assert result.exports.jsonl.read_bytes() == result.exports.jsonl_bytes
    assert result.exports.markdown.read_bytes() == result.exports.markdown_bytes
