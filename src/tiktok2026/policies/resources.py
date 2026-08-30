from tiktok2026.contracts import ExecutionResult, ResourceState
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


def check_smoke_feasibility(
    result: ExecutionResult,
    *,
    memory_limit_bytes: int,
    timeout_seconds: int,
    gpu_requested: bool,
) -> PolicyDecision:
    """Apply controller-owned feasibility limits to measured smoke evidence."""

    if result.execution_kind != "smoke":
        return PolicyDecision(False, "not_smoke_execution")
    if result.exit_code != 0:
        return PolicyDecision(False, "smoke_exit_failed")
    if not result.smoke_output_valid:
        return PolicyDecision(False, "smoke_output_invalid")
    if result.elapsed_seconds > timeout_seconds:
        return PolicyDecision(False, "smoke_timeout_exceeded")
    if result.memory_measurement_status != "measured":
        return PolicyDecision(False, "smoke_memory_telemetry_unavailable")
    if (
        result.measured_peak_memory_bytes is None
        or result.measured_peak_memory_bytes > memory_limit_bytes
    ):
        return PolicyDecision(False, "smoke_memory_limit_exceeded")
    if gpu_requested and result.gpu_hours <= 0.0:
        return PolicyDecision(False, "smoke_gpu_allocation_unavailable")
    return PolicyDecision(True, "smoke_feasible")
