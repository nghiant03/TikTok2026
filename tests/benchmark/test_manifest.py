import json
from pathlib import Path

import pytest

from tiktok2026.benchmark.kuaireand_pure.manifest import BenchmarkManifest, verify_protected_files


def test_canonical_manifest_uses_judging_metrics() -> None:
    path = Path("src/tiktok2026/benchmark/kuaireand_pure/manifest.json")
    manifest = BenchmarkManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
    assert manifest.judging_metrics == ("NDCG@10", "Recall@50")
    assert manifest.judging_evaluator_status == "provisional"


def test_protected_file_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "baseline").mkdir()
    (tmp_path / "baseline" / "evaluate.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="protected file hash mismatch"):
        verify_protected_files(tmp_path, {"baseline/evaluate.py": "0" * 64})
