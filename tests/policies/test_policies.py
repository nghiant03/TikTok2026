from tiktok2026.contracts import ExecutionResult, FailureKind, Fidelity, ResourceState
from tiktok2026.policies.lifecycle import (
    can_repair,
    convergence_reason,
    valid_fidelity_transition,
)
from tiktok2026.policies.paths import check_changed_paths
from tiktok2026.policies.resources import can_reserve_iteration, check_smoke_feasibility


def test_protected_baseline_change_is_rejected() -> None:
    decision = check_changed_paths(("baseline/evaluate.py",), ("src/tiktok2026/experiment",))
    assert not decision.allowed
    assert decision.reason == "protected_path"


def test_out_of_scope_change_is_rejected() -> None:
    decision = check_changed_paths(("README.md",), ("src/tiktok2026/experiment",))
    assert not decision.allowed
    assert decision.reason == "outside_implementation_scope"


def test_converges_after_three_insignificant_results() -> None:
    assert convergence_reason([0.50, 0.501, 0.5015, 0.5018], epsilon=0.002) == "plateau"


def test_final_reserve_cannot_fund_iteration() -> None:
    state = ResourceState(
        remaining_gpu_hours=1.0,
        accumulated_gpu_hours=0.0,
        remaining_wall_seconds=100.0,
        used_tokens=0,
        remaining_tokens=100,
        disk_bytes_available=1000,
        reserved_final_gpu_hours=0.75,
    )
    assert not can_reserve_iteration(state, requested_gpu_hours=0.5).allowed


def test_repairs_and_fidelity_are_bounded() -> None:
    assert can_repair(2).allowed
    assert not can_repair(3).allowed
    assert valid_fidelity_transition(Fidelity.SMOKE, Fidelity.PROXY).allowed
    assert not valid_fidelity_transition(Fidelity.FULL, Fidelity.SMOKE).allowed


def test_smoke_feasibility_fails_closed_for_unknown_memory_and_gpu() -> None:
    result = ExecutionResult(
        execution_id="smoke-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        exit_code=0,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        execution_kind="smoke",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        dataset_view_sha256="b" * 64,
        smoke_output_valid=True,
        scientific_evidence=False,
    )
    assert not check_smoke_feasibility(
        result, memory_limit_bytes=100, timeout_seconds=2, gpu_requested=False
    ).allowed
    measured = result.model_copy(
        update={
            "measured_peak_memory_bytes": 50,
            "memory_measurement_status": "measured",
            "resource_measurement_basis": "docker_stats",
        }
    )
    assert not check_smoke_feasibility(
        measured, memory_limit_bytes=100, timeout_seconds=2, gpu_requested=True
    ).allowed


def test_smoke_feasibility_accepts_allocated_gpu_without_gpu_telemetry() -> None:
    result = ExecutionResult(
        execution_id="smoke-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        exit_code=0,
        elapsed_seconds=1.0,
        gpu_hours=0.25,
        execution_kind="smoke",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        dataset_view_sha256="b" * 64,
        measured_peak_memory_bytes=50,
        memory_measurement_status="measured",
        resource_measurement_basis="docker_stats",
        smoke_output_valid=True,
        scientific_evidence=False,
        gpu_telemetry_status="unavailable",
    )

    decision = check_smoke_feasibility(
        result, memory_limit_bytes=100, timeout_seconds=2, gpu_requested=True
    )

    assert decision.allowed


def test_smoke_feasibility_rejects_zero_allocated_gpu_time() -> None:
    result = ExecutionResult(
        execution_id="smoke-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        exit_code=0,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        execution_kind="smoke",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        dataset_view_sha256="b" * 64,
        measured_peak_memory_bytes=50,
        memory_measurement_status="measured",
        resource_measurement_basis="docker_stats",
        smoke_output_valid=True,
        scientific_evidence=False,
        gpu_telemetry_status="unavailable",
    )

    decision = check_smoke_feasibility(
        result, memory_limit_bytes=100, timeout_seconds=2, gpu_requested=True
    )

    assert not decision.allowed
    assert decision.reason == "smoke_gpu_allocation_unavailable"


def test_smoke_feasibility_rejects_failed_exit() -> None:
    result = ExecutionResult(
        execution_id="smoke-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        exit_code=1,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        failure_kind=FailureKind.SCHEMA_MISMATCH,
        execution_kind="smoke",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="a" * 64,
        dataset_view_sha256="b" * 64,
        smoke_output_valid=False,
        scientific_evidence=False,
    )
    assert check_smoke_feasibility(
        result, memory_limit_bytes=100, timeout_seconds=2, gpu_requested=False
    ).reason == "smoke_exit_failed"
