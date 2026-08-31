from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tiktok2026.contracts import (
    FullAttemptClaimRequest,
    FullScientificAttemptClaim,
    RunClosure,
    ScoredObservation,
)


def _claim(**updates: object) -> FullScientificAttemptClaim:
    values: dict[str, object] = {
        "attempt_id": "attempt-1",
        "execution_id": "execution-1",
        "run_id": "run-1",
        "experiment_id": "experiment-1",
        "source_registration_id": "source-" + "a" * 40,
        "source_commit": "a" * 40,
        "attempt_sequence": 1,
        "claimed_at": datetime.now(UTC),
    }
    values.update(updates)
    return FullScientificAttemptClaim.model_validate(values)


@pytest.mark.parametrize("sequence", range(1, 51))
def test_full_attempt_claim_accepts_sequences_one_through_fifty(sequence: int) -> None:
    assert _claim(attempt_sequence=sequence).attempt_sequence == sequence


def test_full_attempt_claim_rejects_sequence_fifty_one() -> None:
    with pytest.raises(ValidationError):
        _claim(attempt_sequence=51)


def test_claim_request_rejects_caller_owned_cap() -> None:
    with pytest.raises(ValidationError):
        FullAttemptClaimRequest(
            attempt_id="attempt-1",
            execution_id="execution-1",
            run_id="run-1",
            experiment_id="experiment-1",
            source_registration_id="source-" + "a" * 40,
            source_commit="a" * 40,
            max_attempts=49,
        )


def test_observation_requires_approved_evidence() -> None:
    with pytest.raises(ValidationError):
        ScoredObservation(
            observation_id="observation-1",
            run_id="run-1",
            experiment_id="experiment-1",
            attempt_id="attempt-1",
            execution_id="execution-1",
            evaluation_id="evaluation-1",
            checkpoint_id="checkpoint-1",
            source_commit="a" * 40,
            evaluator_id="evaluator-1",
            evaluator_sha256="b" * 64,
            dataset_manifest_id="manifest-1",
            dataset_manifest_sha256="c" * 64,
            validity="provisional",
            primary_score=0.5,
            validation_report_id="report-1",
            validation_evidence_refs=(),
        )


def test_attempt_cap_closure_requires_consuming_cap() -> None:
    with pytest.raises(ValidationError):
        RunClosure(
            closure_id="closure-1",
            run_id="run-1",
            reason="attempt_cap",
            attempt_count=49,
            scored_observation_count=0,
        )
