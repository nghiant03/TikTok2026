from tiktok2026.contracts import Fidelity, ResourceState
from tiktok2026.policies.lifecycle import (
    can_repair,
    convergence_reason,
    valid_fidelity_transition,
)
from tiktok2026.policies.paths import check_changed_paths
from tiktok2026.policies.resources import can_reserve_iteration


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
