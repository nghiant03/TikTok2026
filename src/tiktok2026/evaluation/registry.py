from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    VerifiedDataset,
    encode_row_identity,
    read_verified_rows,
)
from tiktok2026.contracts import (
    EvaluationContext,
    EvaluationRequest,
    EvaluationResult,
    FinalTestClaim,
    MetricValue,
    PredictionArtifactRegistration,
    PredictionRow,
)
from tiktok2026.evaluation.metrics import evaluate_rankings


class EvaluationAdapter(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class EvaluationDataError(ValueError):
    pass


class FinalTestClaimResolver(Protocol):
    """Authority boundary for persisted Phase 1 final-test claims."""

    def resolve(self, claim_id: str) -> FinalTestClaim | None: ...


@dataclass(frozen=True)
class LabeledRow:
    row_id: str
    user_id: str
    item_id: str
    label: int
    row_identity: tuple[str, ...]


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _evaluator_bundle() -> bytes:
    metrics_path = Path(__file__).with_name("metrics.py")
    return _canonical_json(
        {
            "schema_version": "1",
            "evaluator_version": "provisional-v2",
            "registry_implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "metric_implementation_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "metric_parameters": {"ndcg_cutoff": 10, "recall_cutoff": 50},
            "ranking_config": {"validation_ranking": "mean(NDCG@10, Recall@50)"},
        }
    )


def evaluator_implementation_sha256() -> str:
    return hashlib.sha256(_evaluator_bundle()).hexdigest()


def _expected_rows(dataset: VerifiedDataset, split: str) -> tuple[LabeledRow, ...]:
    manifest = dataset.manifest
    result: list[LabeledRow] = []
    for row in read_verified_rows(dataset, split):
        identity = tuple(row[column] for column in manifest.row_identity_columns)
        result.append(
            LabeledRow(
                row_id=encode_row_identity(identity),
                user_id=row[manifest.user_id_column],
                item_id=row[manifest.item_id_column],
                label=int(row[manifest.label_column]),
                row_identity=identity,
            )
        )
    return tuple(result)


def _validate_rows(
    predictions: tuple[PredictionRow, ...], expected: tuple[LabeledRow, ...]
) -> None:
    if not predictions or len(predictions) != len(expected):
        raise EvaluationDataError("predictions must contain exactly the expected rows")
    expected_ids = [row.row_id for row in expected]
    actual_ids = [row.row_id for row in predictions]
    if len(set(expected_ids)) != len(expected_ids):
        raise EvaluationDataError("verified rows contain duplicate row identities")
    if actual_ids != expected_ids:
        raise EvaluationDataError("prediction rows are missing, extra, or reordered")
    for prediction, label in zip(predictions, expected, strict=True):
        if prediction.row_identity != label.row_identity:
            raise EvaluationDataError(f"prediction row identity mismatch: {prediction.row_id}")
        if (prediction.user_id, prediction.item_id) != (label.user_id, label.item_id):
            raise EvaluationDataError(f"prediction user/item mismatch: {prediction.row_id}")


class ProvisionalEvaluator:
    def __init__(
        self,
        evaluator_id: str,
        artifacts: Mapping[str, PredictionArtifactRegistration] | None = None,
        datasets: Mapping[str, VerifiedDataset] | None = None,
        claim_resolver: FinalTestClaimResolver | None = None,
    ) -> None:
        self.evaluator_id = evaluator_id
        self.artifacts = dict(artifacts or {})
        self.datasets = dict(datasets or {})
        self.claim_resolver = claim_resolver

    def _authorize_test(self, context: EvaluationContext) -> None:
        claim_id = context.authorization_claim_id
        if claim_id is None or self.claim_resolver is None:
            raise EvaluationDataError("test evaluation requires an authoritative claim")
        claim = self.claim_resolver.resolve(claim_id)
        if claim is None:
            raise EvaluationDataError("test authorization claim was not found")
        if claim.evaluator_id != self.evaluator_id:
            raise EvaluationDataError(
                "test authorization evaluator does not match evaluator instance"
            )
        if claim.evaluator_sha256 != context.evaluator_sha256:
            raise EvaluationDataError("test authorization evaluator hash does not match context")
        claim_provenance = (
            claim.run_id,
            claim.experiment_id,
            claim.source_commit,
            claim.evaluator_id,
            claim.evaluator_sha256,
            claim.dataset_manifest_id,
            claim.dataset_manifest_sha256,
            claim.split,
            claim.checkpoint_id,
            claim.execution_id,
            claim.prediction_artifact_id,
            claim.prediction_sha256,
        )
        context_provenance = (
            context.run_id,
            context.experiment_id,
            context.source_commit,
            context.evaluator_id,
            context.evaluator_sha256,
            context.dataset_manifest_id,
            context.dataset_manifest_sha256,
            context.split,
            context.checkpoint_id,
            context.execution_id,
            context.prediction_artifact_id,
            context.prediction_sha256,
        )
        if (
            claim.dataset_manifest_id is None
            or claim.dataset_manifest_sha256 is None
            or claim.split is None
            or claim.checkpoint_id is None
            or claim.execution_id is None
            or claim.prediction_artifact_id is None
            or claim.prediction_sha256 is None
            or claim.evaluator_sha256 is None
            or claim.claim_id != claim_id
            or claim_provenance != context_provenance
        ):
            raise EvaluationDataError("test authorization claim does not match evaluation context")

    def _load_predictions(self, context: EvaluationContext) -> tuple[PredictionRow, ...]:
        artifact_id = context.prediction_artifact_id
        try:
            registration = self.artifacts[artifact_id]
        except KeyError as error:
            raise EvaluationDataError(f"unregistered prediction artifact: {artifact_id}") from error
        path = registration.path.resolve()
        if not path.is_file():
            raise EvaluationDataError(f"prediction artifact is missing: {path}")
        payload_bytes = path.read_bytes()
        digest = hashlib.sha256(payload_bytes).hexdigest()
        if digest != registration.sha256 or digest != context.prediction_sha256:
            raise EvaluationDataError("prediction artifact hash mismatch")
        try:
            payload = json.loads(payload_bytes)
            rows = tuple(PredictionRow.model_validate(row) for row in payload["rows"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise EvaluationDataError("prediction artifact has invalid schema") from error
        expected_payload = {
            "schema_version": "1",
            "manifest_sha256": context.dataset_manifest_sha256,
            "split": context.split,
            "source_commit": context.source_commit,
            "execution_id": context.execution_id,
        }
        for key, value in expected_payload.items():
            if payload.get(key) != value:
                raise EvaluationDataError(f"prediction artifact provenance mismatch: {key}")
        return rows

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        context = request.context
        if context.evaluator_id != self.evaluator_id:
            raise EvaluationDataError("evaluation evaluator does not match evaluator instance")
        if context.evaluator_sha256 != evaluator_implementation_sha256():
            raise EvaluationDataError("evaluation evaluator hash does not match evaluator instance")
        if context.split == "test":
            self._authorize_test(context)
        if context.split == "valid" and context.authorization_claim_id is not None:
            raise EvaluationDataError("iterative evaluation cannot carry test authorization")
        try:
            dataset = self.datasets[context.dataset_manifest_id]
        except KeyError as error:
            raise EvaluationDataError("unregistered dataset manifest") from error
        if dataset.manifest_sha256 != context.dataset_manifest_sha256:
            raise EvaluationDataError("dataset manifest hash mismatch")
        registration = self.artifacts.get(context.prediction_artifact_id)
        if registration is None:
            raise EvaluationDataError("unregistered prediction artifact")
        provenance = (
            registration.checkpoint_id,
            registration.source_commit,
            registration.execution_id,
            registration.dataset_manifest_id,
            registration.dataset_manifest_sha256,
            registration.split,
        )
        expected_provenance = (
            context.checkpoint_id,
            context.source_commit,
            context.execution_id,
            context.dataset_manifest_id,
            context.dataset_manifest_sha256,
            context.split,
        )
        if provenance != expected_provenance:
            raise EvaluationDataError("prediction artifact provenance does not match context")
        predictions = self._load_predictions(context)
        expected = _expected_rows(dataset, context.split)
        _validate_rows(predictions, expected)
        metrics = evaluate_rankings(
            [row.user_id for row in expected],
            [row.label for row in expected],
            [row.score for row in predictions],
        )
        return EvaluationResult(
            evaluation_id=context.evaluation_id,
            experiment_id=context.experiment_id,
            checkpoint_id=context.checkpoint_id,
            metrics=(
                MetricValue(name="NDCG@10", value=metrics["NDCG@10"]),
                MetricValue(name="Recall@50", value=metrics["Recall@50"]),
            ),
            evaluator_artifact_id=self.evaluator_id,
            evaluator_sha256=evaluator_implementation_sha256(),
            prediction_sha256=context.prediction_sha256,
            validity="provisional",
            source_commit=context.source_commit,
            execution_id=context.execution_id,
            dataset_manifest_id=context.dataset_manifest_id,
            dataset_manifest_sha256=context.dataset_manifest_sha256,
            prediction_artifact_id=context.prediction_artifact_id,
            split=context.split,
            run_id=context.run_id,
        )


class EvaluatorRegistry:
    def __init__(self, evaluators: dict[str, EvaluationAdapter]) -> None:
        self.evaluators = dict(evaluators)

    def resolve(self, evaluator_id: str) -> EvaluationAdapter:
        try:
            return self.evaluators[evaluator_id]
        except KeyError as error:
            raise ValueError(f"unknown evaluator: {evaluator_id}") from error
