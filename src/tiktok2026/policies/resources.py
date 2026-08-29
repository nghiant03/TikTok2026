from tiktok2026.contracts import ResourceState
from tiktok2026.policies.paths import PolicyDecision


def can_reserve_iteration(
    state: ResourceState,
    requested_gpu_hours: float,
    requested_wall_seconds: float = 0.0,
    requested_tokens: int = 0,
    requested_disk_bytes: int = 0,
) -> PolicyDecision:
    usable_gpu = state.remaining_gpu_hours - state.reserved_final_gpu_hours
    allowed = (
        requested_gpu_hours <= usable_gpu
        and requested_wall_seconds <= state.remaining_wall_seconds
        and requested_tokens <= state.remaining_tokens
        and requested_disk_bytes <= state.disk_bytes_available
    )
    return PolicyDecision(allowed, "allowed" if allowed else "insufficient_resources")
