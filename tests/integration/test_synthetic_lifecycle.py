import pytest

from tiktok2026.testing import run_synthetic_lifecycle


@pytest.mark.asyncio
async def test_two_consecutive_synthetic_experiment_cycles() -> None:
    result = await run_synthetic_lifecycle(2)

    assert result["experiment_ids"] == ["synthetic-1", "synthetic-2"]
    assert len(result["scores"]) == 2
    assert all(score == pytest.approx(1.0) for score in result["scores"])
    assert result["terminal_reason"] == "synthetic_iteration_limit"
