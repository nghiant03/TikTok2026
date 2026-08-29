from __future__ import annotations

import hashlib
from typing import Protocol

from tiktok2026.contracts import EvaluationRequest, EvaluationResult, MetricValue
from tiktok2026.evaluation.metrics import evaluate_rankings


class EvaluationAdapter(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class ProvisionalEvaluator:
    def __init__(self, evaluator_id: str) -> None:
        self.evaluator_id = evaluator_id

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        metrics = evaluate_rankings(request.user_ids, request.labels, request.scores)
        return EvaluationResult(
            evaluation_id=request.evaluation_id,
            experiment_id=request.experiment_id,
            checkpoint_id=request.checkpoint_id,
            metrics=(
                MetricValue(name="NDCG@10", value=metrics["NDCG@10"]),
                MetricValue(name="Recall@50", value=metrics["Recall@50"]),
            ),
            evaluator_artifact_id=self.evaluator_id,
            evaluator_sha256=hashlib.sha256(self.evaluator_id.encode()).hexdigest(),
            prediction_sha256=request.prediction_sha256,
            validity="provisional",
        )


class EvaluatorRegistry:
    def __init__(self, evaluators: dict[str, EvaluationAdapter]) -> None:
        self.evaluators = dict(evaluators)

    def resolve(self, evaluator_id: str) -> EvaluationAdapter:
        try:
            return self.evaluators[evaluator_id]
        except KeyError as error:
            raise ValueError(f"unknown evaluator: {evaluator_id}") from error
