import json
from pathlib import Path

from tiktok2026.experiment.train import run_training


def test_training_writes_versioned_bundle_without_test_evaluation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    bundle = run_training(output, seed=7, fidelity="smoke")
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["seed"] == 7
    assert payload["fidelity"] == "smoke"
    assert not (output / "test_metrics.json").exists()
