from __future__ import annotations

from tiktok2026.contracts import Fidelity
from tiktok2026.policies.paths import PolicyDecision


def can_repair(repair_attempts: int, maximum: int = 2) -> PolicyDecision:
    return PolicyDecision(
        repair_attempts < maximum, "allowed" if repair_attempts < maximum else "repair_limit"
    )


def valid_fidelity_transition(current: Fidelity, requested: Fidelity) -> PolicyDecision:
    order = {Fidelity.SMOKE: 0, Fidelity.PROXY: 1, Fidelity.FULL: 2}
    allowed = order[requested] >= order[current] and order[requested] - order[current] <= 1
    return PolicyDecision(allowed, "allowed" if allowed else "invalid_fidelity_transition")


def convergence_reason(
    scores: list[float], epsilon: float = 0.002, patience: int = 3
) -> str | None:
    if len(scores) < patience + 1:
        return None
    champion = scores[0]
    insignificant = 0
    for score in scores[1:]:
        if score > champion + epsilon:
            champion = score
            insignificant = 0
        else:
            champion = max(champion, score)
            insignificant += 1
    return "plateau" if insignificant >= patience else None
