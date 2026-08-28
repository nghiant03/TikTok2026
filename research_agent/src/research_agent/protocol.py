from __future__ import annotations

import hashlib
from pathlib import Path

from research_agent.contracts import BenchmarkContract


class ProtocolVerificationError(ValueError):
    """The configured official Starter Kit does not match the pinned protocol."""


def verify_official_starter_kit(
    root: Path,
    contract: BenchmarkContract | None = None,
) -> None:
    benchmark = contract or BenchmarkContract()
    expected_hashes = {
        "data.py": benchmark.data_loader_sha256,
        "evaluate.py": benchmark.evaluator_sha256,
        "baseline_scores.json": benchmark.baseline_scores_sha256,
    }
    resolved_root = root.resolve(strict=True)
    for name, expected_hash in expected_hashes.items():
        path = resolved_root / name
        if not path.is_file() or path.is_symlink():
            raise ProtocolVerificationError(
                f"official Starter Kit artifact is missing or unsafe: {path}"
            )
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ProtocolVerificationError(
                f"official Starter Kit artifact hash mismatch: {name}"
            )
