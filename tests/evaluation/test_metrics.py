import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

import tiktok2026.evaluation.registry as registry
from tiktok2026.benchmark.kuaireand_pure.manifest import (
    DatasetManifest,
    canonical_manifest_sha256,
    verify_dataset_manifest,
)
from tiktok2026.contracts import (
    EvaluationContext,
    EvaluationRequest,
    FinalTestClaim,
    PredictionArtifactRegistration,
)
from tiktok2026.evaluation.metrics import PredictionValidationError, evaluate_rankings
from tiktok2026.evaluation.registry import (
    EvaluationDataError,
    ProvisionalEvaluator,
    evaluator_implementation_sha256,
)


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


def _evaluation_fixture(tmp_path: Path) -> tuple[EvaluationRequest, ProvisionalEvaluator, Path]:
    data_root = tmp_path / "data"
    data_root.mkdir()
    data_path = data_root / "valid.csv"
    data_path.write_text(
        "row_id,user_id,item_id,feature,label\nr1,u1,i1,0.9,1\nr2,u1,i2,0.1,0\n",
        encoding="utf-8",
    )
    manifest = DatasetManifest.model_validate(
        {
            "manifest_id": "manifest-1",
            "data_root_env": "UNUSED",
            "row_identity_columns": ["row_id", "user_id", "item_id"],
            "feature_columns": ["feature"],
            "files": [
                {
                    "path": "valid.csv",
                    "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    "schema": ["row_id", "user_id", "item_id", "feature", "label"],
                    "split": "valid",
                }
            ],
            "splits": {
                "valid": {
                    "files": ["valid.csv"],
                    "identity_sha256": hashlib.sha256(
                        "".join(
                            json.dumps(
                                (row_id, "u1", "i1" if row_id == "r1" else "i2"),
                                separators=(",", ":"),
                            )
                            + "\n"
                            for row_id in ("r1", "r2")
                        ).encode()
                    ).hexdigest(),
                }
            },
        }
    )
    verified = verify_dataset_manifest(manifest, data_root, splits={"valid"})
    prediction_path = tmp_path / "predictions.json"
    prediction_payload = {
        "schema_version": "1",
        "manifest_sha256": canonical_manifest_sha256(manifest),
        "split": "valid",
        "source_commit": "a" * 40,
        "execution_id": "exec-1",
        "rows": [
            {
                "row_id": "[\"r1\",\"u1\",\"i1\"]",
                "row_identity": ["r1", "u1", "i1"],
                "user_id": "u1",
                "item_id": "i1",
                "score": 0.9,
            },
            {
                "row_id": "[\"r2\",\"u1\",\"i2\"]",
                "row_identity": ["r2", "u1", "i2"],
                "user_id": "u1",
                "item_id": "i2",
                "score": 0.1,
            },
        ],
    }
    prediction_path.write_text(
        json.dumps(prediction_payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    registration = PredictionArtifactRegistration(
        artifact_id="predictions-1",
        path=prediction_path,
        sha256=prediction_sha256,
        checkpoint_id="checkpoint-1",
        source_commit="a" * 40,
        execution_id="exec-1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256=verified.manifest_sha256,
        split="valid",
    )
    context = EvaluationContext(
        run_id="run-1",
        evaluation_id="eval-1",
        experiment_id="exp-1",
        checkpoint_id="checkpoint-1",
        source_commit="a" * 40,
        execution_id="exec-1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256=verified.manifest_sha256,
        split="valid",
        prediction_artifact_id="predictions-1",
        prediction_sha256=prediction_sha256,
        evaluator_id="provisional-v1",
        evaluator_sha256=evaluator_implementation_sha256(),
    )
    request = EvaluationRequest(evaluation_id="eval-1", context=context)
    return (
        request,
        ProvisionalEvaluator(
            "provisional-v1", {"predictions-1": registration}, {"manifest-1": verified}
        ),
        prediction_path,
    )


def test_registry_returns_explicitly_provisional_result(tmp_path: Path) -> None:
    request, evaluator, _ = _evaluation_fixture(tmp_path)
    result = evaluator.evaluate(request)
    assert result.validity == "provisional"
    assert {metric.name for metric in result.metrics} == {"NDCG@10", "Recall@50"}
    assert result.evaluator_sha256 == evaluator_implementation_sha256()


def test_artifact_tampering_and_reordering_are_rejected(tmp_path: Path) -> None:
    request, evaluator, prediction_path = _evaluation_fixture(tmp_path)
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8").replace("0.1", "0.2"), encoding="utf-8"
    )
    with pytest.raises(EvaluationDataError, match="hash mismatch"):
        evaluator.evaluate(request)


def test_prediction_order_is_rejected_after_hash_verification(tmp_path: Path) -> None:
    request, evaluator, prediction_path = _evaluation_fixture(tmp_path)
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    payload["rows"].reverse()
    prediction_path.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    evaluator.artifacts["predictions-1"] = evaluator.artifacts["predictions-1"].model_copy(
        update={"sha256": digest}
    )
    context = request.context.model_copy(update={"prediction_sha256": digest})
    with pytest.raises(EvaluationDataError, match="reordered"):
        evaluator.evaluate(EvaluationRequest(evaluation_id=request.evaluation_id, context=context))


def test_test_evaluation_requires_authorization_claim(tmp_path: Path) -> None:
    request, evaluator, _ = _evaluation_fixture(tmp_path)
    context = request.context.model_copy(update={"split": "test"})
    with pytest.raises(EvaluationDataError, match="authoritative claim"):
        evaluator.evaluate(EvaluationRequest(evaluation_id=request.evaluation_id, context=context))


def test_registered_artifact_provenance_must_match_context(tmp_path: Path) -> None:
    request, evaluator, _ = _evaluation_fixture(tmp_path)
    context = request.context.model_copy(update={"checkpoint_id": "other-checkpoint"})
    with pytest.raises(EvaluationDataError, match="provenance"):
        evaluator.evaluate(EvaluationRequest(evaluation_id=request.evaluation_id, context=context))


class FakeClaimResolver:
    def __init__(self, claim: FinalTestClaim) -> None:
        self.claim = claim

    def resolve(self, claim_id: str) -> FinalTestClaim | None:
        return self.claim if claim_id == self.claim.claim_id else None


def test_forged_test_claim_is_rejected(tmp_path: Path) -> None:
    request, base_evaluator, _ = _evaluation_fixture(tmp_path)
    context = request.context.model_copy(
        update={"split": "test", "authorization_claim_id": "claim-1"}
    )
    claim = FinalTestClaim(
        claim_id="claim-1",
        run_id="different-run",
        experiment_id=context.experiment_id,
        source_commit=context.source_commit,
        evaluator_id=context.evaluator_id,
        dataset_manifest_id=context.dataset_manifest_id,
        dataset_manifest_sha256=context.dataset_manifest_sha256,
        split="test",
        checkpoint_id=context.checkpoint_id,
        execution_id=context.execution_id,
        prediction_artifact_id=context.prediction_artifact_id,
        prediction_sha256=context.prediction_sha256,
        evaluator_sha256=context.evaluator_sha256,
    )
    evaluator = ProvisionalEvaluator(
        base_evaluator.evaluator_id,
        base_evaluator.artifacts,
        base_evaluator.datasets,
        FakeClaimResolver(claim),
    )
    with pytest.raises(EvaluationDataError, match="does not match"):
        evaluator.evaluate(EvaluationRequest(evaluation_id=request.evaluation_id, context=context))


def test_test_claim_cannot_substitute_another_evaluator(tmp_path: Path) -> None:
    request, base_evaluator, _ = _evaluation_fixture(tmp_path)
    context = request.context.model_copy(
        update={"split": "test", "authorization_claim_id": "claim-2"}
    )
    claim = FinalTestClaim(
        claim_id="claim-2",
        run_id=context.run_id,
        experiment_id=context.experiment_id,
        source_commit=context.source_commit,
        evaluator_id="substitute-evaluator",
        evaluator_sha256=context.evaluator_sha256,
        dataset_manifest_id=context.dataset_manifest_id,
        dataset_manifest_sha256=context.dataset_manifest_sha256,
        split="test",
        checkpoint_id=context.checkpoint_id,
        execution_id=context.execution_id,
        prediction_artifact_id=context.prediction_artifact_id,
        prediction_sha256=context.prediction_sha256,
    )
    evaluator = ProvisionalEvaluator(
        base_evaluator.evaluator_id,
        base_evaluator.artifacts,
        base_evaluator.datasets,
        FakeClaimResolver(claim),
    )
    with pytest.raises(EvaluationDataError, match="evaluator instance"):
        evaluator.evaluate(EvaluationRequest(evaluation_id=request.evaluation_id, context=context))


def test_evaluator_identity_changes_when_registry_bytes_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_hash = evaluator_implementation_sha256()
    original_read_bytes: Callable[[Path], bytes] = registry.Path.read_bytes

    def mutated_read_bytes(path: Path) -> bytes:
        content = original_read_bytes(path)
        return content + b"mutation" if path.name == "registry.py" else content

    monkeypatch.setattr(registry.Path, "read_bytes", mutated_read_bytes)
    assert evaluator_implementation_sha256() != original_hash
