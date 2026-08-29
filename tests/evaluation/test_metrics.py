import pytest

from tiktok2026.contracts import EvaluationRequest
from tiktok2026.evaluation.metrics import PredictionValidationError, evaluate_rankings
from tiktok2026.evaluation.registry import EvaluatorRegistry, ProvisionalEvaluator


def test_provisional_metrics_match_fixture() -> None:
    result = evaluate_rankings(
        ["1", "1", "1", "2", "2"],
        [1, 0, 1, 0, 1],
        [0.9, 0.1, 0.8, 0.2, 0.7],
    )
    assert result["NDCG@10"] == pytest.approx(1.0)
    assert result["Recall@50"] == pytest.approx(1.0)


def test_invalid_predictions_are_rejected() -> None:
    with pytest.raises(PredictionValidationError):
        evaluate_rankings(["1"], [1], [float("nan")])


def test_registry_returns_explicitly_provisional_result() -> None:
    evaluator = ProvisionalEvaluator("provisional-v1")
    request = EvaluationRequest(
        evaluation_id="eval-1",
        experiment_id="exp-1",
        checkpoint_id="checkpoint-1",
        user_ids=("1", "1"),
        labels=(1, 0),
        scores=(0.9, 0.1),
        prediction_sha256="1" * 64,
    )
    result = (
        EvaluatorRegistry({"provisional-v1": evaluator}).resolve("provisional-v1").evaluate(request)
    )
    assert result.validity == "provisional"
    assert {metric.name for metric in result.metrics} == {"NDCG@10", "Recall@50"}
