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
    ndcgs: list[float] = []
    recalls: list[float] = []
    for values in grouped.values():
        ranked = [label for _, label in sorted(values, key=lambda item: item[0], reverse=True)]
        positives = sum(ranked)
        dcg = sum(label / math.log2(index + 2) for index, label in enumerate(ranked[:10]))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(positives, 10)))
        ndcgs.append(0.0 if ideal == 0 else dcg / ideal)
        recalls.append(0.0 if positives == 0 else sum(ranked[:50]) / positives)
    return {
        "NDCG@10": sum(ndcgs) / len(ndcgs),
        "Recall@50": sum(recalls) / len(recalls),
    }
