from __future__ import annotations

from decimal import Decimal

from tiktok2026.contracts import (
    Fidelity,
    ImplementationResourceEstimate,
    ResourceState,
)
from tiktok2026.policies.paths import PolicyDecision

MAX_IMPLEMENTATION_DATASET_PASSES = 4


def check_implementation_resource_estimate(
    estimate: ImplementationResourceEstimate,
    *,
    execution_timeout_seconds: int,
    execution_memory_bytes: int,
    resource_state: ResourceState,
    max_dataset_passes: int = MAX_IMPLEMENTATION_DATASET_PASSES,
) -> PolicyDecision:
    """Admit a proposal against controller-owned, technique-neutral limits.

    This check intentionally rejects structural scaling hazards even when the
    numeric estimate happens to fit today's budget.  An absent estimate is a
    legacy-spec compatibility concern and is handled by the caller.
    """
    if estimate.predicted_wall_seconds > execution_timeout_seconds:
        return PolicyDecision(False, "implementation_resource_timeout_exceeded")
    if estimate.predicted_wall_seconds > resource_state.remaining_wall_seconds:
        return PolicyDecision(False, "implementation_resource_budget_exceeded")
    if estimate.predicted_peak_memory_bytes > execution_memory_bytes:
        return PolicyDecision(False, "implementation_resource_memory_exceeded")
    if estimate.predicted_artifact_bytes > resource_state.disk_bytes_available:
        return PolicyDecision(False, "implementation_resource_disk_exceeded")
    if estimate.dataset_passes > max_dataset_passes:
        return PolicyDecision(False, "implementation_resource_too_many_dataset_passes")
    if estimate.high_cardinality_nested_scans:
        return PolicyDecision(False, "implementation_resource_nested_scan_risk")
    if estimate.duplicate_full_materializations:
        return PolicyDecision(False, "implementation_resource_duplicate_materialization")
    return PolicyDecision(True, "allowed")


def can_repair(repair_attempts: int, maximum: int = 3) -> PolicyDecision:
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
    """Return plateau only when the cumulative improvement is within epsilon.

    The comparison is against the best score before the final patience-sized
    window, rather than counting consecutive non-improvements.  This prevents
    a sequence of small gains from resetting or prematurely satisfying the
    stopping rule.
    """
    if patience < 1:
        raise ValueError("patience must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if len(scores) < patience + 1:
        return None
    best_now = max(scores)
    best_before_window = max(scores[:-patience])
    # Decimal conversion from the canonical float spellings makes a decimal
    # boundary such as 0.500 -> 0.502 compare as equality without introducing
    # an arbitrary tolerance that could admit a genuinely larger improvement.
    improvement = Decimal(str(best_now)) - Decimal(str(best_before_window))
    return "plateau" if improvement <= Decimal(str(epsilon)) else None
