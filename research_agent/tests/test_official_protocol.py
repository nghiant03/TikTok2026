from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from research_agent.contracts import BenchmarkContract
from research_agent.protocol import (
    ProtocolVerificationError,
    verify_official_starter_kit,
)

STARTER_KIT_ROOT = (
    Path(__file__).resolve().parents[1] / "external" / "kuairand-starter-kit"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_evaluator() -> ModuleType:
    path = STARTER_KIT_ROOT / "evaluate.py"
    spec = importlib.util.spec_from_file_location("_official_kuairand_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_contract_pins_official_starter_kit_artifacts() -> None:
    contract = BenchmarkContract()

    assert _sha256(STARTER_KIT_ROOT / "evaluate.py") == contract.evaluator_sha256
    assert _sha256(STARTER_KIT_ROOT / "data.py") == contract.data_loader_sha256
    assert (
        _sha256(STARTER_KIT_ROOT / "baseline_scores.json")
        == contract.baseline_scores_sha256
    )
    assert contract.train_date_range == (20220408, 20220421)
    assert contract.validation_date_range == (20220422, 20220428)
    assert contract.public_holdout_date_range == (20220429, 20220508)
    assert contract.public_holdout_development_allowed is False
    assert contract.organizer_hidden_test_locally_available is False
    verify_official_starter_kit(STARTER_KIT_ROOT, contract)


def test_protocol_verifier_rejects_tampered_starter_kit(tmp_path: Path) -> None:
    for name in ("data.py", "evaluate.py", "baseline_scores.json"):
        shutil.copy2(STARTER_KIT_ROOT / name, tmp_path / name)
    (tmp_path / "evaluate.py").write_text("tampered", encoding="utf-8")

    with pytest.raises(ProtocolVerificationError, match="hash mismatch"):
        verify_official_starter_kit(tmp_path)


def test_official_evaluator_handles_perfect_ranking() -> None:
    evaluator = _load_evaluator()

    result = evaluator.evaluate(["u1", "u1"], [1, 0], [0.9, 0.1])

    assert result["GAUC"] == pytest.approx(1.0)
    assert result["nDCG@5"] == pytest.approx(1.0)
    assert result["primary"] == pytest.approx(1.0)


def test_official_evaluator_includes_all_negative_users_in_ndcg() -> None:
    evaluator = _load_evaluator()

    result = evaluator.evaluate(
        ["all-negative", "all-negative", "mixed", "mixed"],
        [0, 0, 1, 0],
        [0.8, 0.2, 0.9, 0.1],
    )

    assert result["GAUC"] == pytest.approx(1.0)
    assert result["nDCG@5"] == pytest.approx(0.5)
    assert result["primary"] == pytest.approx(0.75)


def test_official_auc_applies_tie_correction() -> None:
    evaluator = _load_evaluator()

    assert evaluator.auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)
