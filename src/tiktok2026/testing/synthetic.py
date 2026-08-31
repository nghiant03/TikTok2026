from __future__ import annotations

from dataclasses import dataclass

from tiktok2026.contracts import EvaluationRequest, EvaluationResult, MetricValue
from tiktok2026.evaluation.metrics import evaluate_rankings


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
    metrics = evaluate_rankings(
        [row.user_id for row in rows],
        [row.label for row in rows],
        scores,
    )
    return EvaluationResult(
        evaluation_id=f"eval-{experiment_id}",
        experiment_id=experiment_id,
        checkpoint_id=f"checkpoint-{experiment_id}",
        metrics=(
            MetricValue(name="GAUC", value=metrics["GAUC"]),
            MetricValue(name="nDCG@5", value=metrics["nDCG@5"]),
        ),
        evaluator_artifact_id="synthetic-evaluator-v1",
        evaluator_sha256="0" * 64,
        prediction_sha256="1" * 64,
        validity="provisional",
    )


class FixtureEvaluator:
    """Offline evaluator that only scores the bounded validation fixture."""

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        context = request.context
        if context.split != "valid":
            raise ValueError("fixture evaluator cannot access test labels")
        rows = fixture_rows()
        result = evaluate_fixture(context.experiment_id, rows, score_rows(rows, 1.0))
        return result.model_copy(
            update={
                "evaluation_id": request.evaluation_id,
                "checkpoint_id": context.checkpoint_id,
                "evaluator_artifact_id": context.evaluator_id,
                "evaluator_sha256": context.evaluator_sha256,
                "prediction_sha256": context.prediction_sha256,
                "dataset_manifest_sha256": context.dataset_manifest_sha256,
                "split": context.split,
                "run_id": context.run_id,
                "source_commit": context.source_commit,
                "execution_id": context.execution_id,
                "dataset_manifest_id": context.dataset_manifest_id,
                "prediction_artifact_id": context.prediction_artifact_id,
            }
        )
