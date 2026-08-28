import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "src/tiktok2026/benchmark/kuaireand_pure/manifest.json"


def test_protected_starter_kit_hashes_match_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())

    for relative_path, expected_hash in manifest["protected_reference_files"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
