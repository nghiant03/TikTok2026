from tiktok2026.policies.lifecycle import convergence_reason


def test_convergence_requires_n_plus_one_scores() -> None:
    assert convergence_reason([0.5, 0.501, 0.501], epsilon=0.002, patience=3) is None


def test_convergence_accepts_equality_at_epsilon() -> None:
    assert convergence_reason([0.5, 0.501, 0.502, 0.502], epsilon=0.002) == "plateau"


def test_convergence_rejects_just_over_epsilon() -> None:
    assert convergence_reason([0.5, 0.501, 0.502, 0.5021], epsilon=0.002) is None


def test_convergence_rejects_cumulative_small_gains_over_epsilon() -> None:
    assert convergence_reason([0.5, 0.501, 0.502, 0.503], epsilon=0.002) is None


def test_convergence_uses_historical_best_on_regression() -> None:
    assert convergence_reason([0.5, 0.51, 0.49, 0.48], epsilon=0.002) is None
