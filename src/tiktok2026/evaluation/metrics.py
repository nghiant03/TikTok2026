from __future__ import annotations

import math
from collections.abc import Sequence


class PredictionValidationError(ValueError):
    pass


def evaluate_rankings(
    user_ids: Sequence[str], labels: Sequence[int], scores: Sequence[float]
) -> dict[str, float]:
    if not user_ids or len(user_ids) != len(labels) or len(labels) != len(scores):
        raise PredictionValidationError("prediction arrays must be nonempty and equal length")
    if any(label not in (0, 1) for label in labels):
        raise PredictionValidationError("labels must be binary")
    if any(not math.isfinite(score) for score in scores):
        raise PredictionValidationError("scores must be finite")
    grouped: dict[str, list[tuple[float, int]]] = {}
    for user_id, label, score in zip(user_ids, labels, scores, strict=True):
        grouped.setdefault(user_id, []).append((score, label))

    def auc(values: list[tuple[float, int]]) -> float:
        """Return Mann-Whitney AUC with average ranks for tied scores."""
        pairs = sorted(values)
        ranks = [0.0] * len(pairs)
        start = 0
        while start < len(pairs):
            end = start
            while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[start][0]:
                end += 1
            average_rank = (start + end) / 2.0 + 1.0
            for index in range(start, end + 1):
                ranks[index] = average_rank
            start = end + 1
        positives = sum(label for _, label in pairs)
        negatives = len(pairs) - positives
        if positives == 0 or negatives == 0:
            return 0.5
        positive_rank_sum = sum(
            rank for rank, (_, label) in zip(ranks, pairs, strict=True) if label == 1
        )
        return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
            positives * negatives
        )

    ndcgs: list[float] = []
    gauc_numerator = 0.0
    gauc_denominator = 0
    for values in grouped.values():
        ranked_values = sorted(values, key=lambda item: -item[0])
        ranked = [label for _, label in ranked_values]
        positives = sum(ranked)
        dcg = sum(
            ((2**label) - 1) / math.log2(index + 2)
            for index, label in enumerate(ranked[:5])
        )
        ideal = sum(
            ((2**label) - 1) / math.log2(index + 2)
            for index, label in enumerate(sorted(ranked, reverse=True)[:5])
        )
        ndcgs.append(0.0 if ideal == 0 else dcg / ideal)
        if 0 < positives < len(ranked):
            gauc_numerator += positives * auc(values)
            gauc_denominator += positives
    gauc = gauc_numerator / gauc_denominator if gauc_denominator else 0.5
    ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    return {
        "GAUC": gauc,
        "nDCG@5": ndcg,
    }
