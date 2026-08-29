from __future__ import annotations

import hashlib

from tiktok2026.policies.paths import PolicyDecision, check_changed_paths


def validate_diff(
    changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
) -> PolicyDecision:
    return check_changed_paths(changed_paths, allowed_scopes)


def patch_signature(patch: str) -> str:
    normalized = "\n".join(line.rstrip() for line in patch.replace("\r\n", "\n").splitlines())
    return hashlib.sha256(normalized.encode()).hexdigest()
