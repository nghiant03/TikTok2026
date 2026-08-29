from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

PROTECTED_PATHS = frozenset(
    {
        "baseline/README.md",
        "baseline/data.py",
        "baseline/evaluate.py",
        "baseline/submit.py",
        "baseline/baseline_scores.json",
    }
)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


def check_changed_paths(
    changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
) -> PolicyDecision:
    raw_paths = tuple(PurePosixPath(path) for path in changed_paths)
    if any(
        path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts
        for path in raw_paths
    ):
        return PolicyDecision(False, "invalid_path")
    normalized = tuple(path.as_posix() for path in raw_paths)
    if any(path in PROTECTED_PATHS or path.startswith("baseline/") for path in normalized):
        return PolicyDecision(False, "protected_path")
    for path in normalized:
        if not any(
            path == scope or path.startswith(f"{scope.rstrip('/')}/") for scope in allowed_scopes
        ):
            return PolicyDecision(False, "outside_implementation_scope")
    return PolicyDecision(True, "allowed")
