import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from tiktok2026.contracts import (
    CriterionAssessmentStatus,
    CriterionResolutionClaim,
    ExperimentSpec,
    Fidelity,
    ImplementationCriterionAssessment,
    ImplementationCriterionId,
    ImplementationRequest,
    RunPhase,
    ValidationBlockerContext,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
)
from tiktok2026.graph.routes import route_after_validation
from tiktok2026.graph.state import ProductionState
from tiktok2026.persistence.repositories import ApplicationRepository, PersistenceConflictError


def _report(
    report_id: str, verdict: ValidationVerdict, **updates: object
) -> ValidationReport:
    data: dict[str, object] = {
        "report_id": report_id,
        "experiment_id": "experiment-1",
        "stage": ValidationStage.IMPLEMENTATION,
        "verdict": verdict,
        "leakage_risk": "none",
    }
    data.update(updates)
    return ValidationReport.model_validate(data)


def _state() -> ProductionState:
    return {
        "run_id": "run-1",
        "phase": RunPhase.IMPLEMENT,
        "current_experiment_id": "experiment-1",
        "current_hypothesis_id": None,
        "active_worktree_id": None,
        "latest_validation_report_id": None,
        "latest_execution_result_id": None,
        "latest_evaluation_result_id": None,
        "orchestration_decision_id": None,
        "repair_attempts": 0,
        "fidelity": Fidelity.SMOKE,
        "pending_route": None,
        "terminal_reason": None,
        "state_version": 1,
    }


def _subject(report: ValidationReport, diff_sha256: str = "a" * 64) -> dict[str, object]:
    return {
        "report_id": report.report_id,
        "implementation_authority": {
            "evidence_id": f"implementation-diff-{diff_sha256}",
            "worktree_id": "worktree-1",
            "parent_commit": "a" * 40,
            "diff_sha256": diff_sha256,
            "changed_files": ("src/tiktok2026/experiment/train.py",),
            "allowed_scopes": ("src/tiktok2026/experiment",),
        },
    }


def _operation(
    report: ValidationReport, subject: dict[str, object] | None = None
) -> ValidationOperationIdentity:
    subject_data = subject or _subject(report)
    subject_json = json.dumps(
        subject_data, sort_keys=True, separators=(",", ":")
    ).encode()
    authority = subject_data.get("implementation_authority")
    authority_data = cast(dict[str, object], authority) if isinstance(authority, dict) else {}
    claimed_diff = authority_data.get("diff_sha256")
    diff_sha256 = claimed_diff if isinstance(claimed_diff, str) else "a" * 64
    return ValidationOperationIdentity(
        operation_id=f"operation-{report.report_id}",
        run_id="run-1",
        experiment_id=report.experiment_id,
        stage=report.stage,
        repair_attempt=0,
        subject_sha256=hashlib.sha256(subject_json).hexdigest(),
        implementation_diff_sha256=(
            diff_sha256 if report.stage == ValidationStage.IMPLEMENTATION else None
        ),
    )


def test_legacy_blocker_normalization_is_stable() -> None:
    first = _report("report-1", ValidationVerdict.REJECTED, blockers=("missing check",))
    second = _report("report-1", ValidationVerdict.REJECTED, blockers=("missing check",))

    assert first.blockers[0].blocker_id == second.blockers[0].blocker_id
    assert first.blockers[0].text == "missing check"
    assert first.blockers[0].report_id == "report-1"


def test_blocker_context_is_bounded_and_reports_cannot_resolve_introduced_blockers() -> None:
    context = ValidationBlockerContext(
        blocker_id="blocker-1",
        text="x" * 2_000,
        evidence_refs=tuple(str(index) for index in range(8)),
    )
    assert len(context.text) == 2_000
    with pytest.raises(ValidationError):
        ValidationBlockerContext(blocker_id="blocker-1", text="x" * 2_001)

    with pytest.raises(ValidationError, match="cannot resolve"):
        _report(
            "report-1",
            ValidationVerdict.APPROVED,
            blockers=({
                "blocker_id": "blocker-1",
                "experiment_id": "experiment-1",
                "stage": ValidationStage.IMPLEMENTATION,
                "text": "introduced",
            },),
            resolves_blocker_ids=("blocker-1",),
            evidence_refs=("evidence",),
        )


def test_implementation_operation_requires_and_retains_diff_identity() -> None:
    operation = ValidationOperationIdentity(
        operation_id="operation-1",
        run_id="run-1",
        experiment_id="experiment-1",
        stage=ValidationStage.IMPLEMENTATION,
        repair_attempt=1,
        subject_sha256="a" * 64,
        implementation_diff_sha256="b" * 64,
    )
    assert operation.implementation_diff_sha256 == "b" * 64


def test_persistence_rejects_implementation_operation_without_authority(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    report = _report("report-1", ValidationVerdict.APPROVED)
    subject: dict[str, object] = {"report_id": report.report_id}

    with pytest.raises(PersistenceConflictError, match="requires implementation authority"):
        repository.put_validation_report(report, "run-1", _operation(report, subject), subject)


def test_persistence_rejects_arbitrary_implementation_diff_claim(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    report = _report("report-1", ValidationVerdict.APPROVED)
    subject = _subject(report, "b" * 64)
    operation = _operation(report, subject).model_copy(
        update={"implementation_diff_sha256": "a" * 64}
    )

    with pytest.raises(PersistenceConflictError, match="does not match operation"):
        repository.put_validation_report(report, "run-1", operation, subject)


def test_validation_ledger_is_idempotent_and_conflict_checked(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    rejected = _report(
        "report-1", ValidationVerdict.REJECTED, blockers=("missing check",), evidence_refs=("e1",)
    )
    subject = _subject(rejected)
    repository.put_validation_report(rejected, "run-1", _operation(rejected), subject)
    repository.put_validation_report(rejected, "run-1", _operation(rejected), subject)
    assert len(repository.get_unresolved_blockers("experiment-1")) == 1

    with pytest.raises(PersistenceConflictError):
        repository.put_validation_report(
            rejected.model_copy(update={"warnings": ("changed",)}),
            "run-1",
            _operation(rejected),
            subject,
        )

    approved = _report(
        "report-2",
        ValidationVerdict.APPROVED,
        resolves_blocker_ids=(rejected.blockers[0].blocker_id,),
        evidence_refs=("repair-evidence",),
    )
    repository.put_validation_report(
        approved,
        "run-1",
        _operation(approved),
        _subject(approved),
    )
    assert repository.get_unresolved_blocker_ids("experiment-1") == ()
    assert len(repository.list_blocker_resolutions("experiment-1")) == 1

    resolution = repository.list_blocker_resolutions("experiment-1")[0]
    repository.put_blocker_resolution(resolution, "run-1")


def test_criterion_blockers_are_stable_and_occurrences_are_append_only(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    first = _report(
        "report-1",
        ValidationVerdict.REJECTED,
        blockers=({"criterion_id": "leakage", "text": "first finding"},),
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.FAIL,
                evidence_refs=("e1",),
            ),
        ),
    )
    second = _report(
        "report-2",
        ValidationVerdict.REJECTED,
        blockers=({"criterion_id": "leakage", "text": "new finding"},),
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PARTIAL,
                evidence_refs=("e2",),
            ),
        ),
    )
    repository.put_validation_report(first, "run-1", _operation(first), _subject(first))
    repository.put_validation_report(second, "run-1", _operation(second), _subject(second))

    assert first.blockers[0].blocker_id == second.blockers[0].blocker_id
    assert repository.list_validation_blockers("experiment-1")[0].text == "first finding"
    assert repository.get_criterion_repeat_count("experiment-1", "leakage") == 2


def test_repairable_claim_partially_resolves_a_criterion_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    rejected = _report(
        "report-1",
        ValidationVerdict.REJECTED,
        blockers=({"criterion_id": "leakage", "text": "two issues"},),
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.FAIL,
            ),
        ),
    )
    repository.put_validation_report(rejected, "run-1", _operation(rejected), _subject(rejected))
    claim = CriterionResolutionClaim(
        criterion_id=ImplementationCriterionId.LEAKAGE,
        status=CriterionAssessmentStatus.PARTIAL,
        blocker_ids=(rejected.blockers[0].blocker_id,),
        evidence_refs=("repair-evidence",),
        claim="one part repaired",
    )
    repaired = _report(
        "report-2",
        ValidationVerdict.REPAIRABLE,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PARTIAL,
            ),
        ),
        resolution_claims=(claim,),
    )
    repository.put_validation_report(repaired, "run-1", _operation(repaired), _subject(repaired))
    repository.put_validation_report(repaired, "run-1", _operation(repaired), _subject(repaired))

    assert repository.get_unresolved_blocker_ids("experiment-1") == (
        rejected.blockers[0].blocker_id,
    )
    assert repository.list_blocker_resolutions("experiment-1") == ()
    assert repository.get_criterion_repeat_count("experiment-1", "leakage") == 2


def test_criterion_status_reopens_and_resolves_stable_blocker(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    rejected = _report(
        "report-1",
        ValidationVerdict.REJECTED,
        blockers=({"criterion_id": "leakage", "text": "finding"},),
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.FAIL,
            ),
        ),
    )
    repository.put_validation_report(rejected, "run-1", _operation(rejected), _subject(rejected))
    blocker_id = rejected.blockers[0].blocker_id

    partial = _report(
        "report-2",
        ValidationVerdict.REPAIRABLE,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PARTIAL,
            ),
        ),
        resolution_claims=(
            CriterionResolutionClaim(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PARTIAL,
                blocker_ids=(blocker_id,),
                evidence_refs=("partial-evidence",),
            ),
        ),
    )
    repository.put_validation_report(partial, "run-1", _operation(partial), _subject(partial))
    assert repository.get_unresolved_blocker_ids("experiment-1") == (blocker_id,)
    assert repository.list_blocker_resolutions("experiment-1") == ()

    passed = _report(
        "report-3",
        ValidationVerdict.APPROVED,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PASS,
                evidence_refs=("pass-evidence",),
            ),
        ),
        resolution_claims=(
            CriterionResolutionClaim(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PASS,
                blocker_ids=(blocker_id,),
                evidence_refs=("pass-evidence",),
            ),
        ),
    )
    repository.put_validation_report(passed, "run-1", _operation(passed), _subject(passed))
    assert repository.get_unresolved_blocker_ids("experiment-1") == ()
    assert len(repository.list_blocker_resolutions("experiment-1")) == 1

    failed_again = _report(
        "report-4",
        ValidationVerdict.REJECTED,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.FAIL,
            ),
        ),
    )
    repository.put_validation_report(
        failed_again, "run-1", _operation(failed_again), _subject(failed_again)
    )
    assert repository.get_unresolved_blocker_ids("experiment-1") == (blocker_id,)

    passed_again = _report(
        "report-5",
        ValidationVerdict.APPROVED,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PASS,
                evidence_refs=("pass-again-evidence",),
            ),
        ),
        resolution_claims=(
            CriterionResolutionClaim(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.PASS,
                blocker_ids=(blocker_id,),
                evidence_refs=("pass-again-evidence",),
            ),
        ),
    )
    repository.put_validation_report(
        passed_again, "run-1", _operation(passed_again), _subject(passed_again)
    )
    assert repository.get_unresolved_blocker_ids("experiment-1") == ()
    assert len(repository.list_blocker_resolutions("experiment-1")) == 2


def test_proposal_criterion_occurrence_is_not_an_implementation_repeat(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    proposal = _report(
        "proposal-report",
        ValidationVerdict.REJECTED,
        stage=ValidationStage.PROPOSAL,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.RESOURCE_FEASIBILITY,
                status=CriterionAssessmentStatus.FAIL,
            ),
        ),
    )
    repository.put_validation_report(proposal, "run-1", _operation(proposal), _subject(proposal))

    assert (
        repository.get_criterion_repeat_count(
            "experiment-1", ImplementationCriterionId.RESOURCE_FEASIBILITY
        )
        == 0
    )
    assert (
        repository.get_criterion_repeat_count(
            "experiment-1",
            ImplementationCriterionId.RESOURCE_FEASIBILITY,
            ValidationStage.PROPOSAL,
        )
        == 1
    )


def test_criterion_occurrence_conflict_rolls_back_report_replay(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    report = _report(
        "report-1",
        ValidationVerdict.REJECTED,
        criterion_assessments=(
            ImplementationCriterionAssessment(
                criterion_id=ImplementationCriterionId.LEAKAGE,
                status=CriterionAssessmentStatus.FAIL,
            ),
        ),
    )
    repository.put_validation_report(report, "run-1", _operation(report), _subject(report))
    with pytest.raises(PersistenceConflictError):
        repository.put_validation_report(
            report.model_copy(
                update={
                    "criterion_assessments": (
                        ImplementationCriterionAssessment(
                            criterion_id=ImplementationCriterionId.LEAKAGE,
                            status=CriterionAssessmentStatus.PARTIAL,
                        ),
                    )
                }
            ),
            "run-1",
            _operation(report),
            _subject(report),
        )
    assert repository.get_criterion_repeat_count("experiment-1", "leakage") == 1


def test_approval_is_blocked_by_history_until_explicit_resolution() -> None:
    report = _report("report-1", ValidationVerdict.APPROVED)
    unresolved = ("historical-blocker",)
    assert route_after_validation(_state(), report, unresolved) == "repair"
    assert route_after_validation(_state(), report) == "register_source"


def test_implementation_request_carries_blocker_ids() -> None:
    request = ImplementationRequest(
        request_id="request-1",
        experiment_id="experiment-1",
        experiment_spec=ExperimentSpec(
            experiment_id="experiment-1",
            hypothesis_id="hypothesis-1",
            hypothesis="h",
            mechanism="m",
            motivation="m",
            expected_signal="s",
            implementation_scope=("src/tiktok2026/experiment",),
            fidelity=Fidelity.SMOKE,
            success_criteria="s",
            failure_criteria="f",
        ),
        allowed_scopes=("src/tiktok2026/experiment",),
        unresolved_blocker_ids=("blocker-1",),
    )
    assert request.unresolved_blocker_ids == ("blocker-1",)
