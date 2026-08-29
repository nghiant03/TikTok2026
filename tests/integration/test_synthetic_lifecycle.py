from pathlib import Path

import pytest

from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.testing import run_synthetic_lifecycle


@pytest.mark.asyncio
async def test_two_cycles_persist_audit_and_provisional_bundle(tmp_path: Path) -> None:
    result = await run_synthetic_lifecycle(2, runtime_root=tmp_path / "runtime")

    assert result.experiment_ids == ("synthetic-1", "synthetic-2")
    assert all(score == pytest.approx(1.0) for score in result.scores)
    assert result.terminal_reason == "synthetic_iteration_limit"
    assert result.finalization.validity == "provisional"
    assert result.exports.jsonl.exists()
    assert result.exports.markdown.exists()

    repository = ApplicationRepository(result.paths.application_db)
    assert tuple(
        repository.get_experiment(experiment_id) is not None
        for experiment_id in result.experiment_ids
    ) == (True, True)
    assert len(repository.list_audit_events(result.run_id)) >= 4
    assert len(repository.list_json("evaluation")) == 2
    assert repository.get_finalization(result.finalization.finalization_id) == result.finalization
