from __future__ import annotations

import math
from dataclasses import dataclass

from tiktok2026.contracts import EvaluationResult, MetricValue


@dataclass(frozen=True)
class RankingRow:
    user_id: str
    label: int
    feature: float


def fixture_rows() -> tuple[RankingRow, ...]:
    return (
        RankingRow("u1", 1, 0.9),
        RankingRow("u1", 0, 0.2),
        RankingRow("u1", 1, 0.7),
        RankingRow("u2", 0, 0.1),
        RankingRow("u2", 1, 0.8),
        RankingRow("u2", 0, 0.3),
    )


def score_rows(rows: tuple[RankingRow, ...], scale: float) -> tuple[float, ...]:
    return tuple(row.feature * scale for row in rows)


def evaluate_fixture(
    experiment_id: str, rows: tuple[RankingRow, ...], scores: tuple[float, ...]
) -> EvaluationResult:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for row, score in zip(rows, scores, strict=True):
        grouped.setdefault(row.user_id, []).append((score, row.label))
    ndcgs: list[float] = []
    recalls: list[float] = []
    for values in grouped.values():
        ranked = [label for _, label in sorted(values, reverse=True)]
        positives = sum(ranked)
        dcg = sum(label / math.log2(index + 2) for index, label in enumerate(ranked[:10]))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(positives, 10)))
        ndcgs.append(0.0 if ideal == 0 else dcg / ideal)
        recalls.append(0.0 if positives == 0 else sum(ranked[:50]) / positives)
    return EvaluationResult(
        evaluation_id=f"eval-{experiment_id}",
        experiment_id=experiment_id,
        checkpoint_id=f"checkpoint-{experiment_id}",
        metrics=(
            MetricValue(name="NDCG@10", value=sum(ndcgs) / len(ndcgs)),
            MetricValue(name="Recall@50", value=sum(recalls) / len(recalls)),
        ),
        evaluator_artifact_id="synthetic-evaluator-v1",
        evaluator_sha256="0" * 64,
        prediction_sha256="1" * 64,
        validity="provisional",
    )
