from __future__ import annotations

import hashlib

from tiktok2026.policies.paths import PolicyDecision, check_changed_paths


def validate_diff(
    changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
) -> PolicyDecision:
    return check_changed_paths(changed_paths, allowed_scopes)


def normalize_patch(patch: str) -> str:
    lines = patch.replace("\r\n", "\n").splitlines()
    normalized = "\n".join(line.rstrip() for line in lines)
    return f"{normalized}\n" if normalized else ""


def patch_signature(patch: str) -> str:
    return hashlib.sha256(normalize_patch(patch).encode()).hexdigest()
