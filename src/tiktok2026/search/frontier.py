from tiktok2026.contracts import FrontierCandidate


def select_frontier(
    candidates: tuple[FrontierCandidate, ...], limit: int = 4
) -> tuple[FrontierCandidate, ...]:
    champions = sorted(
        (item for item in candidates if item.slot == "champion"),
        key=lambda item: (-item.score, item.experiment_id),
    )
    alternatives = sorted(
        (item for item in candidates if item.slot == "alternative"),
        key=lambda item: (-item.score, item.experiment_id),
    )
    diagnostics = sorted(
        (item for item in candidates if item.slot == "diagnostic"),
        key=lambda item: (-item.score, item.experiment_id),
    )
    selected = champions[:1] + alternatives[:2] + diagnostics[:1]
    return tuple(selected[:limit])
