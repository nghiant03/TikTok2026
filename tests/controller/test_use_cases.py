from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from tiktok2026 import use_cases
from tiktok2026.adapters import DeterministicPolicyGate
from tiktok2026.contracts import (
    DEFAULT_IMPLEMENTATION_CRITERIA,
    AgentFailure,
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    BaselineCalibrationRecord,
    BlockerResolution,
    ChampionBinding,
    ContractModel,
    CriterionAssessmentStatus,
    CriterionResolutionClaim,
    DatasetContext,
    DatasetFileContext,
    DatasetManifestIdentity,
    DatasetViewProvenance,
    DatasetViewRow,
    DecisionAction,
    DiagnosticMetricValue,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionResult,
    ExperimentRegistryEntry,
    ExperimentRegistrySnapshot,
    ExperimentSpec,
    FailureKind,
    FailureRecord,
    Fidelity,
    FullAttemptClaimRequest,
    FullScientificAttemptClaim,
    ImplementationAttemptRecord,
    ImplementationCriterionAssessment,
    ImplementationCriterionId,
    ImplementationRequest,
    ImplementationResourceEstimate,
    ImplementationResult,
    MetricValue,
    OrchestrationDecision,
    OrchestrationRequest,
    PredictionArtifactRegistration,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    RunClosure,
    RunPhase,
    ScoredObservation,
    ScoredObservationRequest,
    SourceRegistration,
    ValidationBlocker,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    ValidationVerdict,
    WorktreeAssignment,
)
from tiktok2026.controller import (
    ControllerServices,
    ProductionController,
)
from tiktok2026.graph.state import ProductionState
from tiktok2026.repository.inspector import RepositoryInspector
from tiktok2026.use_cases import (
    ModelUnavailableError,
    ServiceTransitions,
    TerminalLifecycleError,
    make_service_transitions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def minimal_state(**overrides: Any) -> ProductionState:
    values: ProductionState = {
        "run_id": "test-run",
        "phase": RunPhase.BOOTSTRAP,
        "current_experiment_id": None,
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
        "state_version": 0,
    }
    for key, value in overrides.items():
        if key in values:
            values[key] = value  # type: ignore[typeddict-item]
    return values


class _FakeStore:
    def __init__(self) -> None:
        self.persisted: list[tuple[str, str, int, dict[str, object]]] = []
        self.experiments: dict[str, ExperimentSpec] = {}
        self.experiment_updates: list[tuple[str, str]] = []
        self.executions: dict[str, ExecutionResult] = {}
        self.sources: dict[str, SourceRegistration] = {}
        self.assignments: dict[str, WorktreeAssignment] = {}
        self.predictions: dict[str, PredictionArtifactRegistration] = {}
        self.artifacts: dict[str, ArtifactRecord] = {}
        self.evaluator: EvaluatorIdentity | None = None
        self.manifest: DatasetManifestIdentity | None = None
        self.evaluations: list[EvaluationResult] = []
        self.baseline_calibrations: list[BaselineCalibrationRecord] = []
        self.failures: list[object] = []
        self.json_records: dict[tuple[str, str], str] = {}
        self.validation_reports: dict[str, ValidationReport] = {}
        self.validation_operations: dict[str, ValidationOperationIdentity] = {}
        self.validation_blockers: dict[str, ValidationBlocker] = {}
        self.validation_resolutions: dict[str, BlockerResolution] = {}
        self.criterion_occurrences: list[tuple[str, str, CriterionAssessmentStatus]] = []
        self.attempts: dict[str, FullScientificAttemptClaim] = {}
        self.observations: dict[str, ScoredObservation] = {}
        self.closure: RunClosure | None = None

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        return self.experiments.get(experiment_id)

    def get_experiment_registry(
        self, limit: int = 50, exclude_experiment_id: str | None = None
    ) -> ExperimentRegistrySnapshot:
        all_specs = tuple(
            spec
            for spec in self.experiments.values()
            if spec.experiment_id != exclude_experiment_id
        )
        specs = all_specs[-limit:]
        entries = tuple(
            ExperimentRegistryEntry(
                experiment_id=spec.experiment_id,
                hypothesis_id=spec.hypothesis_id,
                parent_experiment_id=spec.parent_experiment_id,
                hypothesis=spec.hypothesis,
                mechanism=spec.mechanism,
                status="proposed",
                evaluation_ids=tuple(
                    result.evaluation_id
                    for result in self.evaluations
                    if result.experiment_id == spec.experiment_id
                ),
                evaluator_sha256s=tuple(
                    sorted(
                        {
                            result.evaluator_sha256
                            for result in self.evaluations
                            if result.experiment_id == spec.experiment_id
                        }
                    )
                ),
            )
            for spec in specs
        )
        return ExperimentRegistrySnapshot(
            evidence_id="experiment-registry-test",
            entries=entries,
            total_experiments=len(all_specs),
            complete=len(all_specs) <= limit,
        )

    def put_experiment(
        self,
        spec: ExperimentSpec,
        status: str,
        run_id: str,
        transition_id: str,
        expected_predecessor: str | None = None,
    ) -> None:
        del run_id, transition_id, expected_predecessor
        self.experiments[spec.experiment_id] = spec
        self.experiment_updates.append((spec.experiment_id, status))

    def get_execution_result(self, execution_id: str) -> ExecutionResult | None:
        return self.executions.get(execution_id)

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None:
        return self.sources.get(experiment_id)

    def get_source_registration_by_id(
        self, registration_id: str
    ) -> SourceRegistration | None:
        return next(
            (
                source
                for source in self.sources.values()
                if source.registration_id == registration_id
            ),
            None,
        )

    def get_worktree_assignment(self, experiment_id: str) -> WorktreeAssignment | None:
        return self.assignments.get(experiment_id)

    def get_prediction_artifact(self, artifact_id: str) -> PredictionArtifactRegistration | None:
        return self.predictions.get(artifact_id)

    def get_evaluator_identity(self, evaluator_id: str) -> Any:
        if self.evaluator and self.evaluator.evaluator_id == evaluator_id:
            return self.evaluator
        return None

    def get_dataset_manifest_identity(self) -> DatasetManifestIdentity | None:
        return self.manifest

    def put_evaluation(self, result: EvaluationResult, provenance: Any) -> None:
        del provenance
        self.evaluations.append(result)

    def put_validation_report(
        self,
        report: ValidationReport,
        run_id: str,
        operation: ValidationOperationIdentity | None = None,
        subject: dict[str, object] | None = None,
    ) -> None:
        del run_id, subject
        if operation is None:
            raise ValueError("operation is required")
        self.validation_reports[report.report_id] = report
        self.validation_operations[operation.operation_id] = operation
        self.criterion_occurrences.extend(
            (report.experiment_id, str(assessment.criterion_id), assessment.status)
            for assessment in report.criterion_assessments
            if assessment.status
            in (CriterionAssessmentStatus.FAIL, CriterionAssessmentStatus.PARTIAL)
        )
        self.validation_blockers.update(
            {blocker.blocker_id: blocker for blocker in report.blockers}
        )
        for blocker_id in report.resolves_blocker_ids:
            self.validation_resolutions[blocker_id] = BlockerResolution(
                resolution_id=f"resolution-{operation.operation_id}-{blocker_id}",
                blocker_id=blocker_id,
                report_id=report.report_id,
                experiment_id=report.experiment_id,
                evidence_refs=report.evidence_refs or ("test-evidence",),
                validation_operation_id=operation.operation_id,
            )

    def get_validation_report_by_operation(
        self, operation_id: str
    ) -> ValidationReport | None:
        operation = self.validation_operations.get(operation_id)
        if operation is None:
            return None
        return next(
            (
                report
                for report in self.validation_reports.values()
                if report.validation_operation_id == operation_id
            ),
            None,
        )

    def get_validation_report_for_attempt(
        self, run_id: str, experiment_id: str, stage: ValidationStage, repair_attempt: int
    ) -> ValidationReport | None:
        operation = next(
            (
                operation
                for operation in self.validation_operations.values()
                if operation.run_id == run_id
                and operation.experiment_id == experiment_id
                and operation.stage == stage
                and operation.repair_attempt == repair_attempt
            ),
            None,
        )
        return (
            self.get_validation_report_by_operation(operation.operation_id)
            if operation is not None
            else None
        )

    def get_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        return tuple(
            blocker
            for blocker in self.validation_blockers.values()
            if blocker.experiment_id == experiment_id
            and blocker.blocker_id not in self.validation_resolutions
        )

    def get_criterion_repeat_count(
        self, experiment_id: str, criterion_id: ImplementationCriterionId | str
    ) -> int:
        return sum(
            occurrence_experiment_id == experiment_id
            and occurrence_criterion_id == str(criterion_id)
            and status in (CriterionAssessmentStatus.FAIL, CriterionAssessmentStatus.PARTIAL)
            for occurrence_experiment_id, occurrence_criterion_id, status in (
                self.criterion_occurrences
            )
        )

    def put_execution_result(self, result: ExecutionResult) -> None:
        self.executions[result.execution_id] = result

    def get_evaluation_result(self, evaluation_id: str) -> EvaluationResult | None:
        return next(
            (result for result in self.evaluations if result.evaluation_id == evaluation_id),
            None,
        )

    def list_evaluation_results(self) -> tuple[EvaluationResult, ...]:
        return tuple(self.evaluations)

    def list_baseline_calibrations(self) -> tuple[BaselineCalibrationRecord, ...]:
        return tuple(self.baseline_calibrations)

    def put_failure(self, record: object, run_id: str) -> None:
        del run_id
        self.failures.append(record)

    def list_failure_records(self, run_id: str) -> tuple[FailureRecord, ...]:
        return tuple(
            record
            for record in self.failures
            if isinstance(record, FailureRecord) and record.run_id == run_id
        )

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
        key = (kind, record_id)
        existing = self.json_records.get(key)
        if existing is not None and existing != payload_json:
            raise ValueError(f"record {kind}/{record_id} content changed")
        self.json_records[key] = payload_json

    def list_json(self, kind: str) -> tuple[str, ...]:
        return tuple(
            payload
            for (record_kind, _), payload in sorted(self.json_records.items())
            if record_kind == kind
        )

    def claim_full_attempt(
        self, request: FullAttemptClaimRequest
    ) -> FullScientificAttemptClaim | None:
        existing = next(
            (
                attempt
                for attempt in self.attempts.values()
                if attempt.execution_id == request.execution_id
            ),
            None,
        )
        if existing is not None:
            if any(
                getattr(existing, field) != getattr(request, field)
                for field in (
                    "attempt_id",
                    "execution_id",
                    "run_id",
                    "experiment_id",
                    "source_registration_id",
                    "source_commit",
                )
            ):
                raise ValueError("attempt identity changed")
            return existing
        sequence = len(
            [attempt for attempt in self.attempts.values() if attempt.run_id == request.run_id]
        ) + 1
        if sequence > 50:
            return None
        attempt = FullScientificAttemptClaim(
            **request.model_dump(),
            attempt_sequence=sequence,
            claimed_at=datetime.now(UTC),
        )
        self.attempts[attempt.attempt_id] = attempt
        return attempt

    def list_full_attempt_claims(
        self, run_id: str | None = None
    ) -> tuple[FullScientificAttemptClaim, ...]:
        return tuple(
            sorted(
                (
                    attempt
                    for attempt in self.attempts.values()
                    if run_id is None or attempt.run_id == run_id
                ),
                key=lambda attempt: (attempt.run_id, attempt.attempt_sequence),
            )
        )

    def count_full_attempt_claims(self, run_id: str) -> int:
        return len(self.list_full_attempt_claims(run_id))

    def put_scored_observation(self, request: ScoredObservationRequest) -> ScoredObservation:
        existing = self.observations.get(request.observation_id)
        if existing is not None:
            return existing
        observation = ScoredObservation(**request.model_dump(), scored_at=datetime.now(UTC))
        self.observations[observation.observation_id] = observation
        return observation

    def get_scored_observation(self, observation_id: str) -> ScoredObservation | None:
        return self.observations.get(observation_id)

    def list_scored_observations(
        self, run_id: str | None = None
    ) -> tuple[ScoredObservation, ...]:
        return tuple(
            observation
            for observation in self.observations.values()
            if run_id is None or observation.run_id == run_id
        )

    def close_run(
        self,
        run_id: str,
        reason: str,
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> RunClosure:
        del epsilon, patience
        if self.closure is None:
            self.closure = RunClosure(
                closure_id=f"closure-{run_id}",
                run_id=run_id,
                reason=reason,
                attempt_count=self.count_full_attempt_claims(run_id),
                scored_observation_count=len(self.list_scored_observations(run_id)),
                champion=None,
            )
        return self.closure

    def get_run_closure(self, run_id: str) -> RunClosure | None:
        return self.closure if self.closure is not None and self.closure.run_id == run_id else None

    def get_run_champion(self, run_id: str) -> ChampionBinding | None:
        del run_id
        return None

    def close_run_if_ready(
        self,
        run_id: str,
        *,
        after_failure: bool = False,
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> RunClosure | None:
        del run_id, after_failure, epsilon, patience
        return self.closure

    def get_run_baseline(self, run_id: str) -> Any:
        del run_id
        return None

    def get_baseline_calibration(self, calibration_id: str) -> BaselineCalibrationRecord | None:
        return next(
            (item for item in self.baseline_calibrations if item.calibration_id == calibration_id),
            None,
        )

    def get_artifact(self, artifact_id: str) -> Any:
        return self.artifacts.get(artifact_id)


class _ContentAddressedRegistryStore(_FakeStore):
    def list_experiments_by_status(
        self, run_id: str, status: str
    ) -> tuple[ExperimentSpec, ...]:
        del run_id, status
        return tuple(self.experiments.values())

    def get_experiment_registry(
        self, limit: int = 50, exclude_experiment_id: str | None = None
    ) -> ExperimentRegistrySnapshot:
        snapshot = super().get_experiment_registry(limit, exclude_experiment_id)
        identities = ",".join(entry.experiment_id for entry in snapshot.entries)
        evidence_id = f"experiment-registry-{hashlib.sha256(identities.encode()).hexdigest()}"
        return snapshot.model_copy(update={"evidence_id": evidence_id})


# ---------------------------------------------------------------------------
# Test 1: proposal_policy requires and validates persisted specs
# ---------------------------------------------------------------------------


def _experiment(scope: tuple[str, ...]) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="test hypothesis",
        mechanism="test mechanism",
        motivation="test motivation",
        expected_signal="test signal",
        implementation_scope=scope,
        fidelity=Fidelity.SMOKE,
        success_criteria="test success",
        failure_criteria="test failure",
    )


def _estimated_experiment(scope: tuple[str, ...]) -> ExperimentSpec:
    return _experiment(scope).model_copy(
        update={
            "implementation_resource_estimate": ImplementationResourceEstimate(
                predicted_wall_seconds=1.0,
                predicted_peak_memory_bytes=1 << 20,
                predicted_artifact_bytes=1 << 20,
                dataset_passes=1,
            )
        }
    )


def _passing_implementation_assessments() -> tuple[ImplementationCriterionAssessment, ...]:
    return tuple(
        ImplementationCriterionAssessment(
            criterion_id=criterion,
            status=CriterionAssessmentStatus.PASS,
        )
        for criterion in DEFAULT_IMPLEMENTATION_CRITERIA
    )


async def test_proposal_policy_allows_seeded_spec() -> None:
    """proposal_policy allows a persisted spec with an allowed scope."""
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    services = _make_services(store=store, policy_gate=DeterministicPolicyGate())
    controller = ProductionController(services)

    state = minimal_state(
        current_experiment_id="exp-1",
        current_hypothesis_id="hyp-1",
    )

    result = await controller.proposal_policy(state)
    assert result["pending_route"] == "proposal_validation"
    assert len(store.persisted) == 1


async def test_proposal_policy_rejects_protected_path() -> None:
    """proposal_policy rejects a persisted spec whose scope includes protected paths."""
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("baseline/data.py",))
    controller = ProductionController(
        _make_services(store=store, policy_gate=DeterministicPolicyGate())
    )

    result = await controller.proposal_policy(
        minimal_state(current_experiment_id="exp-1", current_hypothesis_id="hyp-1")
    )

    assert result["pending_route"] == "persist_failure"
    assert len(store.persisted) == 1


async def test_proposal_policy_rejects_scope_outside_experiment_root() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/controller.py",))
    controller = ProductionController(
        _make_services(store=store, policy_gate=DeterministicPolicyGate())
    )

    result = await controller.proposal_policy(
        minimal_state(current_experiment_id="exp-1", current_hypothesis_id="hyp-1")
    )

    assert result["pending_route"] == "persist_failure"
    assert "outside_implementation_scope" in str(result["terminal_reason"])


async def test_proposal_policy_rejects_infeasible_implementation_resources() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",)).model_copy(
        update={
            "implementation_resource_estimate": ImplementationResourceEstimate(
                predicted_wall_seconds=301,
                predicted_peak_memory_bytes=1,
                predicted_artifact_bytes=1,
                dataset_passes=1,
            )
        }
    )
    controller = ProductionController(
        _make_services(store=store, resource_accountant=_FakeResourceAccountant())
    )

    result = await controller.proposal_policy(
        minimal_state(current_experiment_id="exp-1")
    )

    assert result["pending_route"] == "persist_failure"
    assert "implementation_resource_timeout_exceeded" in str(result["terminal_reason"])


def test_missing_implementation_criterion_creates_a_blocker() -> None:
    from tiktok2026 import use_cases

    missing = ImplementationCriterionId.RESOURCE_FEASIBILITY
    report = ValidationReport(
        report_id="implementation-report",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        verdict=ValidationVerdict.APPROVED,
        criterion_assessments=tuple(
            assessment
            for assessment in _passing_implementation_assessments()
            if assessment.criterion_id != missing
        ),
        leakage_risk="none",
    )

    completed = use_cases._complete_implementation_validation(report, ())

    assessment = next(
        item for item in completed.criterion_assessments if item.criterion_id == missing
    )
    assert assessment.status == CriterionAssessmentStatus.FAIL
    assert any(blocker.criterion_id == missing for blocker in completed.blockers)


def test_partial_criterion_resolution_does_not_create_a_duplicate_blocker() -> None:
    from tiktok2026 import use_cases

    criterion = ImplementationCriterionId.LEAKAGE
    prior = ValidationBlocker(
        blocker_id="prior-resource-blocker",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        text="prior leakage issue",
        report_id="prior-report",
        criterion_id=criterion,
    )
    claim = CriterionResolutionClaim(
        criterion_id=criterion,
        status=CriterionAssessmentStatus.PARTIAL,
        blocker_ids=(prior.blocker_id,),
        evidence_refs=("repair-evidence",),
    )
    report = ValidationReport(
        report_id="repair-report",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        verdict=ValidationVerdict.REPAIRABLE,
        criterion_assessments=tuple(
            assessment.model_copy(
                update={"status": CriterionAssessmentStatus.PARTIAL}
            )
            if assessment.criterion_id == criterion
            else assessment
            for assessment in _passing_implementation_assessments()
        ),
        resolution_claims=(claim,),
        leakage_risk="none",
    )

    completed = use_cases._complete_implementation_validation(report, (prior,))

    assert not any(blocker.criterion_id == criterion for blocker in completed.blockers)


def test_second_resource_feasibility_failure_escalates_to_orchestration() -> None:
    from tiktok2026 import use_cases

    store = _FakeStore()
    criterion = ImplementationCriterionId.RESOURCE_FEASIBILITY
    blocker = ValidationBlocker(
        blocker_id="resource-blocker",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        text="resource estimate remains infeasible",
        report_id="implementation-report",
        criterion_id=criterion,
    )
    store.validation_blockers[blocker.blocker_id] = blocker
    store.criterion_occurrences = [
        ("exp-1", str(criterion), CriterionAssessmentStatus.FAIL),
        ("exp-1", str(criterion), CriterionAssessmentStatus.FAIL),
    ]
    report = ValidationReport(
        report_id="implementation-report",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        verdict=ValidationVerdict.APPROVED,
        blockers=(blocker,),
        leakage_risk="none",
    )

    updates = use_cases._replayed_validation_updates(
        ServiceTransitions(run_store=store),
        minimal_state(phase=RunPhase.IMPLEMENT, current_experiment_id="exp-1"),
        report,
    )

    assert updates["pending_route"] == "orchestrate"


async def test_rejected_validation_carries_blockers_into_failure() -> None:
    store = _FakeStore()
    store.manifest = DatasetManifestIdentity(
        manifest_id="manifest-1", manifest_sha256="a" * 64
    )
    store.evaluator = EvaluatorIdentity(
        evaluator_id="evaluator-1", evaluator_sha256="b" * 64, validity="provisional"
    )
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="report-1",
                experiment_id="exp-1",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.REJECTED,
                blockers=("success criterion is not measurable",),
                evidence_refs=("evidence-1",),
                leakage_risk="none",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))

    result = await controller.proposal_validation(
        minimal_state(current_experiment_id="exp-1")
    )

    assert result["pending_route"] == "persist_failure"
    assert "success criterion is not measurable" in str(result["terminal_reason"])
    assert str(result["latest_validation_report_id"]).startswith("validation-report-")
    request = agent.calls[0]
    assert isinstance(request, ValidationRequest)
    assert request.subject["experiment_spec"] == store.experiments["exp-1"].model_dump(
        mode="json"
    )
    context = request.subject["controller_context"]
    assert isinstance(context, dict)
    assert context["dataset_manifest_identity"] == store.manifest.model_dump(mode="json")
    assert context["evaluator_identity"] == store.evaluator.model_dump(mode="json")
    assert context["source_commit_stage"] == "post_implementation"
    assert context["dataset_staging_owner"] == "controller"
    assert context["execution_sandbox_owner"] == "controller"
    assert context["test_access_owner"] == "controller"
    registry = context["experiment_registry"]
    assert isinstance(registry, dict)
    assert registry["evidence_id"] == "experiment-registry-test"
    assert registry["complete"] is True
    assert registry["total_experiments"] == 0


async def test_proposal_validation_reuses_research_registry_identity() -> None:
    store = _ContentAddressedRegistryStore()
    proposal = _estimated_experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ResearchDecision(
                request_id="research-1",
                kind="proposal",
                experiment_spec=proposal,
                message="bounded proposal",
            ),
            ValidationReport(
                report_id="report-1",
                experiment_id=proposal.experiment_id,
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.APPROVED,
                leakage_risk="none",
            ),
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    state = minimal_state(phase=RunPhase.RESEARCH)

    research = await controller.research(state)
    await controller.proposal_validation(state | research)  # type: ignore[arg-type]

    research_request, validation_request = agent.calls
    assert isinstance(research_request, ResearchRequest)
    assert isinstance(validation_request, ValidationRequest)
    assert research_request.controller_context is not None
    research_registry = research_request.controller_context.experiment_registry
    validation_context = validation_request.subject["controller_context"]
    assert research_registry is not None
    assert isinstance(validation_context, dict)
    validation_registry = validation_context["experiment_registry"]
    assert isinstance(validation_registry, dict)
    assert validation_registry["evidence_id"] == research_registry.evidence_id
    assert research_request.experiment_history is not None
    validation_history = validation_request.subject["experiment_history"]
    assert isinstance(validation_history, dict)
    assert validation_history["evidence_id"] == research_request.experiment_history.evidence_id


async def test_rejected_proposal_returns_to_research_with_feedback() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="report-1",
                experiment_id="exp-1",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.REJECTED,
                blockers=("tighten the success criterion",),
                leakage_risk="none",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    state = minimal_state(phase=RunPhase.RESEARCH, current_experiment_id="exp-1")

    validation = await controller.proposal_validation(state)
    persisted = await controller.persist_failure(state | validation)  # type: ignore[arg-type]
    repaired = await controller.repair(state | validation | persisted)  # type: ignore[arg-type]

    assert validation["pending_route"] == "persist_failure"
    assert "tighten the success criterion" in str(validation["terminal_reason"])
    assert persisted["pending_route"] == "repair"
    assert repaired["pending_route"] == "research"


async def test_proposal_repair_passes_authoritative_blocker_context_to_research() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    validator = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="report-1",
                experiment_id="exp-1",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.REPAIRABLE,
                blockers=("tighten the success criterion",),
                evidence_refs=("evidence-1",),
                leakage_risk="none",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=validator))
    state = minimal_state(phase=RunPhase.RESEARCH, current_experiment_id="exp-1")
    validation = await controller.proposal_validation(state)
    persisted = await controller.persist_failure(state | validation)  # type: ignore[arg-type]
    repaired = await controller.repair(state | validation | persisted)  # type: ignore[arg-type]

    researcher = _ScriptedAgentClient(
        [
            ResearchDecision(
                request_id="research-1",
                kind="proposal",
                experiment_spec=_estimated_experiment(
                    ("src/tiktok2026/experiment",)
                ).model_copy(
                    update={"experiment_id": "exp-2", "parent_experiment_id": "exp-1"}
                ),
                message="repaired proposal",
            )
        ]
    )
    research_controller = ProductionController(
        _make_services(store=store, agent_client=researcher)
    )
    result = await research_controller.research(  # type: ignore[arg-type]
        state | validation | persisted | repaired
    )
    request = researcher.calls[0]
    assert isinstance(request, ResearchRequest)
    assert request.parent_experiment_id == "exp-1"
    assert request.proposal_mode == "selected_refinement"
    assert request.proposals_needed == 1
    assert request.selected_experiment == store.experiments["exp-1"]
    assert len(request.unresolved_blockers) == 1
    assert request.unresolved_blockers[0].text == "tighten the success criterion"
    assert request.unresolved_blockers[0].evidence_refs == ("evidence-1",)
    assert result["current_experiment_id"] == "exp-2"
    assert result["repair_attempts"] == 1
    assert result["pending_route"] == "proposal_policy"


async def test_proposal_repair_rejects_changed_content_under_same_identity() -> None:
    store = _FakeStore()
    original = _estimated_experiment(("src/tiktok2026/experiment",))
    store.experiments["exp-1"] = original
    revised = original.model_copy(update={"mechanism": "revised mechanism"})
    researcher = _ScriptedAgentClient(
        [
            ResearchDecision(
                request_id="research-1",
                kind="proposal",
                experiment_spec=revised,
                message="repaired proposal",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=researcher))

    result = await controller.research(
        minimal_state(
            phase=RunPhase.RESEARCH,
            current_experiment_id="exp-1",
            repair_attempts=1,
            terminal_reason="proposal validation failed",
        )
    )

    assert result["pending_route"] == "persist_failure"
    assert "experiment identities are immutable" in str(result["terminal_reason"])
    assert store.experiments["exp-1"] == original


async def test_validation_replay_reuses_bound_report_without_invoking_validator() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="first",
                experiment_id="exp-1",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.REJECTED,
                leakage_risk="none",
            ),
            ValidationReport(
                report_id="divergent",
                experiment_id="exp-1",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.REJECTED,
                leakage_risk="none",
            ),
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    state = minimal_state(current_experiment_id="exp-1")
    first = await controller.proposal_validation(state)
    second = await controller.proposal_validation(state)

    assert len(agent.calls) == 1
    assert second["latest_validation_report_id"] == first["latest_validation_report_id"]


async def test_validation_fails_closed_without_typed_ledger_authority() -> None:
    agent = _ScriptedAgentClient([])
    controller = ProductionController(_make_services(store=None, agent_client=agent))

    result = await controller.proposal_validation(
        minimal_state(current_experiment_id="exp-1")
    )

    assert result["pending_route"] == "persist_failure"
    assert not agent.calls


async def test_execution_repair_returns_to_implement_for_same_experiment() -> None:
    controller = ProductionController(_make_services())

    repaired = await controller.repair(
        minimal_state(
            phase=RunPhase.EXECUTE,
            current_experiment_id="exp-1",
            repair_attempts=0,
        )
    )

    assert repaired["pending_route"] == "implement"
    assert repaired["repair_attempts"] == 1


async def test_repair_at_limit_routes_to_persist_failure() -> None:
    controller = ProductionController(_make_services())

    result = await controller.repair(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=3,
        )
    )

    assert result["pending_route"] == "persist_failure"


async def test_implementation_repairs_use_immutable_attempt_records() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ImplementationResult(
                experiment_id="exp-1",
                patch_artifact_id="patch-0",
                changed_files=("src/tiktok2026/experiment/model.py",),
            ),
            ImplementationResult(
                experiment_id="exp-1",
                patch_artifact_id="patch-1",
                changed_files=("src/tiktok2026/experiment/model.py",),
            ),
        ],
        diffs_after_invoke=(
            "diff --git a/train.py b/train.py\n+attempt 0\n",
            "diff --git a/train.py b/train.py\n+attempt 1\n",
        ),
    )
    agent.scoped_repository = _FakeScopedRepository(
        "diff --git a/train.py b/train.py\n",
        ("src/tiktok2026/experiment/train.py",),
        source="unchanged entrypoint\n",
        base_source="unchanged entrypoint\n",
    )
    controller = ProductionController(
        _make_services(
            store=store,
            agent_client=agent,
            default_timeout_seconds=900,
            default_memory_bytes=4 * 1024**3,
            default_cpus=1.0,
            default_gpu_count=1,
        )
    )

    first = await controller.implement(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=0,
        )
    )
    second = await controller.implement(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=1,
        )
    )

    assert first["pending_route"] == "diff_policy"
    assert second["pending_route"] == "diff_policy"
    assert ("implementation", "exp-1:attempt:0") in store.json_records
    assert ("implementation", "exp-1:attempt:1") in store.json_records
    first_request = agent.calls[0]
    assert isinstance(first_request, ImplementationRequest)
    assert first_request.source_context == {}
    assert first_request.base_source_context == {}
    assert first_request.read_scopes == (
        "src/tiktok2026/contracts",
        "src/tiktok2026/benchmark/kuaireand_pure/manifest.py",
        "tests/experiment/test_training_contract.py",
        "src/tiktok2026/experiment/train.py",
    )
    assert first_request.capabilities == ("scoped_read", "scoped_write", "diff", "checks")
    assert first_request.execution_timeout_seconds == 900
    assert first_request.execution_memory_bytes == 4 * 1024**3
    assert first_request.execution_cpus == 1.0
    assert first_request.execution_gpu_count == 1
    first_record = ImplementationAttemptRecord.model_validate_json(
        store.json_records[("implementation", "exp-1:attempt:0")]
    )
    second_record = ImplementationAttemptRecord.model_validate_json(
        store.json_records[("implementation", "exp-1:attempt:1")]
    )
    assert first_record.prior_diff_sha256 is not None
    assert first_record.result_diff_sha256 is not None
    assert second_record.prior_diff_sha256 == first_record.result_diff_sha256
    assert second_record.result_diff_sha256 != second_record.prior_diff_sha256


async def test_implementation_repair_receives_failure_feedback() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ImplementationResult(
                experiment_id="exp-1",
                patch_artifact_id="patch-1",
                changed_files=("src/tiktok2026/experiment/model.py",),
            )
        ],
        diffs_after_invoke=("diff --git a/train.py b/train.py\n+repaired\n",),
    )
    agent.scoped_repository = _FakeScopedRepository(
        "diff --git a/train.py b/train.py\n",
        ("src/tiktok2026/experiment/train.py",),
        source="current repaired source\n",
        base_source="committed controller entrypoint\n",
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.implement(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=1,
            terminal_reason=(
                'failure:{"evidence": [], "kind": "schema_mismatch", '
                '"message": "path is outside the approved implementation scope"}'
            ),
        )
    )

    request = agent.calls[0]
    assert isinstance(request, ImplementationRequest)
    assert request.repair_feedback == "path is outside the approved implementation scope"
    assert request.source_context == {}
    assert request.base_source_context == {}
    assert request.read_scopes
    assert request.capabilities == ("scoped_read", "scoped_write", "diff", "checks")
    assert result["terminal_reason"] is None


async def test_implementation_repair_requires_a_new_authoritative_diff() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ImplementationResult(
                experiment_id="exp-1",
                patch_artifact_id="unchanged-repair",
                changed_files=("src/tiktok2026/experiment/train.py",),
            )
        ]
    )
    agent.scoped_repository = _FakeScopedRepository(
        "diff --git a/train.py b/train.py\n+existing implementation\n",
        ("src/tiktok2026/experiment/train.py",),
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.implement(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=1,
            terminal_reason=(
                'failure:{"evidence": [], "kind": "schema_mismatch", '
                '"message": "repair this"}'
            ),
        )
    )

    assert result["pending_route"] == "persist_failure"
    assert "did not change the authoritative implementation diff" in str(
        result["terminal_reason"]
    )
    assert ("implementation", "exp-1:attempt:1") not in store.json_records


# ---------------------------------------------------------------------------
# Test 2: research transition repairs one bad structured response
# ---------------------------------------------------------------------------


class _ScriptedAgentClient:
    """AgentClient that returns scripted responses, failing once then succeeding."""

    def __init__(
        self,
        responses: list[ContractModel],
        diffs_after_invoke: tuple[str, ...] = (),
    ) -> None:
        self.responses = list(responses)
        self.diffs_after_invoke = list(diffs_after_invoke)
        self.calls: list[ContractModel] = []
        self.scoped_repository: _FakeScopedRepository | None = None

    async def invoke(self, request: ContractModel) -> ContractModel:
        self.calls.append(request)
        if self.diffs_after_invoke and self.scoped_repository is not None:
            self.scoped_repository.set_diff(self.diffs_after_invoke.pop(0))
        if self.responses:
            return self.responses.pop(0)
        return AgentFailure(
            request_id=getattr(request, "request_id", "unknown"),
            role=AgentRole.RESEARCH,
            kind="model",
            message="no more responses",
            repair_attempts=0,
        )


class _FakeScopedRepository:
    def __init__(
        self,
        diff: str,
        changed_files: tuple[str, ...],
        source: str = "current source\n",
        base_source: str = "base source\n",
    ) -> None:
        self._diff = diff
        self._changed_files = changed_files
        self._source = source
        self._base_source = base_source

    def read(self, relative_path: str, max_characters: int = 20_000) -> str:
        del relative_path, max_characters
        return self._source

    def read_base(self, relative_path: str, max_characters: int = 20_000) -> str:
        del relative_path, max_characters
        return self._base_source

    def diff(self) -> str:
        return self._diff

    def set_diff(self, diff: str) -> None:
        self._diff = diff

    def changed_files(self) -> tuple[str, ...]:
        return self._changed_files


async def test_implementation_validation_receives_pre_registration_authority() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    store.assignments["exp-1"] = WorktreeAssignment(
        worktree_id="worktree-exp-1",
        run_id="test-run",
        experiment_id="exp-1",
        path=Path("/tmp/worktree-exp-1"),
        branch="experiment/test-run/exp-1",
        parent_commit="a" * 40,
    )
    implementation = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="inline-patch-label",
        changed_files=("src/tiktok2026/experiment/model.py",),
    )
    store.put_json(
        "implementation",
        "exp-1:attempt:0",
        ImplementationAttemptRecord(
            experiment_id="exp-1", repair_attempt=0, result=implementation
        ).model_dump_json(),
    )
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="implementation-report-1",
                experiment_id="exp-1",
                stage=ValidationStage.IMPLEMENTATION,
                verdict=ValidationVerdict.APPROVED,
                criterion_assessments=_passing_implementation_assessments(),
                leakage_risk="none",
            )
        ]
    )
    diff = "diff --git a/model.py b/model.py\n+VALUE = 1\n"
    agent.scoped_repository = _FakeScopedRepository(
        diff, ("src/tiktok2026/experiment/model.py",)
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.implementation_validation(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=0,
        )
    )

    assert result["pending_route"] == "register_source"
    request = agent.calls[0]
    assert isinstance(request, ValidationRequest)
    authority = request.subject["implementation_authority"]
    assert isinstance(authority, dict)
    assert authority["diff_sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert authority["parent_commit"] == "a" * 40
    assert authority["path_policy_passed"] is True
    assert authority["source_registration_stage"] == "post_implementation_validation"
    assert "controller_context" in request.subject
    assert request.subject["execution_resources"] == {
        "timeout_seconds": 300,
        "memory_bytes": 1024**3,
        "cpus": 1.0,
        "gpu_count": 0,
    }


async def test_changed_implementation_diff_does_not_replay_stale_approval() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    store.assignments["exp-1"] = WorktreeAssignment(
        worktree_id="worktree-exp-1",
        run_id="test-run",
        experiment_id="exp-1",
        path=Path("/tmp/worktree-exp-1"),
        branch="experiment/test-run/exp-1",
        parent_commit="a" * 40,
    )
    implementation = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="inline-patch-label",
        changed_files=("src/tiktok2026/experiment/model.py",),
    )
    store.put_json(
        "implementation",
        "exp-1:attempt:0",
        ImplementationAttemptRecord(
            experiment_id="exp-1", repair_attempt=0, result=implementation
        ).model_dump_json(),
    )
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="implementation-report-1",
                experiment_id="exp-1",
                stage=ValidationStage.IMPLEMENTATION,
                verdict=ValidationVerdict.APPROVED,
                criterion_assessments=_passing_implementation_assessments(),
                leakage_risk="none",
            ),
            ValidationReport(
                report_id="implementation-report-2",
                experiment_id="exp-1",
                stage=ValidationStage.IMPLEMENTATION,
                verdict=ValidationVerdict.APPROVED,
                criterion_assessments=_passing_implementation_assessments(),
                leakage_risk="none",
            ),
        ]
    )
    first_diff = "diff --git a/model.py b/model.py\n+VALUE = 1\n"
    second_diff = "diff --git a/model.py b/model.py\n+VALUE = 2\n"
    agent.scoped_repository = _FakeScopedRepository(
        first_diff, ("src/tiktok2026/experiment/model.py",)
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    state = minimal_state(
        phase=RunPhase.IMPLEMENT,
        current_experiment_id="exp-1",
        repair_attempts=0,
    )

    first = await controller.implementation_validation(state)
    agent.scoped_repository.set_diff(second_diff)
    second = await controller.implementation_validation(state)

    assert first["pending_route"] == "register_source"
    assert second["pending_route"] == "register_source"
    assert len(agent.calls) == 2
    first_request = agent.calls[0]
    second_request = agent.calls[1]
    assert isinstance(first_request, ValidationRequest)
    assert isinstance(second_request, ValidationRequest)
    assert first_request.validation_operation.operation_id != (
        second_request.validation_operation.operation_id
    )
    assert first_request.validation_operation.implementation_diff_sha256 != (
        second_request.validation_operation.implementation_diff_sha256
    )


async def test_diff_policy_requires_execution_entrypoint_integration() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    implementation = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="inline-patch-label",
        changed_files=("src/tiktok2026/experiment/model.py",),
    )
    store.put_json(
        "implementation",
        "exp-1:attempt:0",
        ImplementationAttemptRecord(
            experiment_id="exp-1", repair_attempt=0, result=implementation
        ).model_dump_json(),
    )
    agent = _ScriptedAgentClient([])
    agent.scoped_repository = _FakeScopedRepository(
        "diff\n", ("src/tiktok2026/experiment/model.py",)
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.diff_policy(
        minimal_state(
            phase=RunPhase.IMPLEMENT,
            current_experiment_id="exp-1",
            repair_attempts=0,
        )
    )

    assert result["pending_route"] == "persist_failure"
    assert "experiment/train.py" in str(result["terminal_reason"])

class _MultiProposalStore(_FakeStore):
    """FakeStore with the two read methods used by multi-proposal orchestration."""

    def list_experiments_by_status(
        self, run_id: str, status: str
    ) -> tuple[ExperimentSpec, ...]:
        del run_id, status
        return tuple(self.experiments.values())

    def list_scored_observations(self, run_id: str) -> tuple[ScoredObservation, ...]:
        del run_id
        return tuple(self.observations.values())


def _proposal(experiment_id: str, hypothesis: str) -> ExperimentSpec:
    return _estimated_experiment(("src/tiktok2026/experiment",)).model_copy(
        update={
            "experiment_id": experiment_id,
            "hypothesis_id": f"hyp-{experiment_id}",
            "hypothesis": hypothesis,
        }
    )


async def test_research_accumulates_three_proposals_before_orchestration_selection() -> None:
    store = _MultiProposalStore()
    proposals = tuple(
        _estimated_experiment(("src/tiktok2026/experiment",)).model_copy(
            update={
                "experiment_id": f"exp-{index}",
                "hypothesis_id": f"hyp-{index}",
                "hypothesis": f"distinct hypothesis {index}",
            }
        )
        for index in range(1, 4)
    )
    agent = _ScriptedAgentClient(
        [
            *(
                ResearchDecision(
                    request_id=f"research-{index}",
                    kind="proposal",
                    experiment_spec=proposal,
                    message="fresh candidate",
                )
                for index, proposal in enumerate(proposals, 1)
            ),
            OrchestrationDecision(
                decision_id="select-exp-2",
                action=DecisionAction.IMPLEMENT,
                target_experiment_id="exp-2",
                rationale="best bounded candidate",
            ),
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    state = minimal_state(phase=RunPhase.RESEARCH)

    first = await controller.research(state)
    second_state = cast(ProductionState, state | first)
    second = await controller.research(second_state)
    third_state = cast(ProductionState, second_state | second)
    third = await controller.research(third_state)
    selection_state = cast(ProductionState, third_state | third)
    selection = await controller.orchestrate(selection_state)

    assert first["pending_route"] == "research"
    assert second["pending_route"] == "research"
    assert third["pending_route"] == "orchestrate"
    assert selection["pending_route"] == "proposal_policy"
    assert selection["current_experiment_id"] == "exp-2"
    research_requests = agent.calls[:3]
    assert all(isinstance(request, ResearchRequest) for request in research_requests)
    assert [request.proposals_needed for request in research_requests] == [3, 2, 1]
    orchestration_request = agent.calls[3]
    assert isinstance(orchestration_request, OrchestrationRequest)
    assert orchestration_request.allowed_actions == (DecisionAction.IMPLEMENT,)
    assert orchestration_request.experiment_history is not None
    assert orchestration_request.experiment_history.total_pending_proposals == 3


async def test_orchestration_cannot_select_before_three_proposals() -> None:
    store = _MultiProposalStore()
    store.experiments["exp-a"] = _proposal("exp-a", "pairwise loss")
    store.experiments["exp-b"] = _proposal("exp-b", "bigger embeddings")
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="request-third",
                action=DecisionAction.RESEARCH,
                rationale="candidate batch is incomplete",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.orchestrate(minimal_state(phase=RunPhase.RESEARCH))

    assert result["pending_route"] == "research"
    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert request.allowed_actions == (DecisionAction.RESEARCH,)


async def test_orchestration_history_preserves_both_judging_metrics() -> None:
    store = _MultiProposalStore()
    store.experiments["exp-scored"] = _proposal("exp-scored", "scored hypothesis")
    store.evaluations.append(
        EvaluationResult(
            evaluation_id="eval-scored",
            run_id="test-run",
            experiment_id="exp-scored",
            checkpoint_id="checkpoint-scored",
            metrics=(
                MetricValue(name="GAUC", value=0.61),
                MetricValue(name="nDCG@5", value=0.73),
            ),
            evaluator_artifact_id="evaluator-1",
            evaluator_sha256="a" * 64,
            prediction_sha256="b" * 64,
            dataset_manifest_id="manifest-1",
            dataset_manifest_sha256="c" * 64,
            validity="provisional",
        )
    )
    store.observations["observation-scored"] = ScoredObservation(
        observation_id="observation-scored",
        run_id="test-run",
        experiment_id="exp-scored",
        attempt_id="attempt-scored",
        execution_id="execution-scored",
        evaluation_id="eval-scored",
        checkpoint_id="checkpoint-scored",
        source_commit="d" * 40,
        evaluator_id="evaluator-1",
        evaluator_sha256="a" * 64,
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="c" * 64,
        validity="provisional",
        primary_score=0.67,
        validation_report_id="validation-scored",
        validation_evidence_refs=("validation-evidence",),
    )
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="research-after-score",
                action=DecisionAction.RESEARCH,
                rationale="form a new candidate batch",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    await controller.orchestrate(minimal_state(phase=RunPhase.PERSIST))

    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert request.experiment_history is not None
    record = request.experiment_history.records[0]
    assert record.gauc == 0.61
    assert record.ndcg_at_5 == 0.73
    assert record.primary_score == 0.67


async def test_orchestration_selects_among_pending_proposals() -> None:
    """With pending proposals, implement is allowed and the chosen proposal becomes current."""
    store = _MultiProposalStore()
    store.experiments["exp-a"] = _proposal("exp-a", "pairwise loss")
    store.experiments["exp-b"] = _proposal("exp-b", "bigger embeddings")
    store.experiments["exp-c"] = _proposal("exp-c", "temporal features")
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="pick-exp-a",
                action=DecisionAction.IMPLEMENT,
                target_experiment_id="exp-a",
                rationale="History favors loss-function experiments",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.orchestrate(minimal_state(phase=RunPhase.RESEARCH))

    assert result["pending_route"] == "proposal_policy"
    assert result["current_experiment_id"] == "exp-a"
    assert result["current_hypothesis_id"] == "hyp-exp-a"
    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert DecisionAction.IMPLEMENT in request.allowed_actions
    assert request.experiment_history is not None
    assert request.experiment_history.pending_proposals[0].expected_signal
    assert (
        request.experiment_history.pending_proposals[0].implementation_resource_estimate
        is not None
    )
    assert {p.experiment_id for p in request.experiment_history.pending_proposals} == {
        "exp-a",
        "exp-b",
        "exp-c",
    }


async def test_orchestration_rejects_unknown_proposal_target() -> None:
    """Implement with an id outside pending proposals is a schema failure."""
    store = _MultiProposalStore()
    store.experiments["exp-a"] = _proposal("exp-a", "pairwise loss")
    store.experiments["exp-b"] = _proposal("exp-b", "bigger embeddings")
    store.experiments["exp-c"] = _proposal("exp-c", "temporal features")
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="pick-ghost",
                action=DecisionAction.IMPLEMENT,
                target_experiment_id="exp-ghost",
                rationale="Invented identity",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.orchestrate(minimal_state(phase=RunPhase.RESEARCH))

    assert result["pending_route"] == "persist_failure"
    assert "unknown proposal" in str(result["terminal_reason"])


async def test_orchestration_bounds_pending_history_and_authorizes_exposed_ids_only() -> None:
    store = _MultiProposalStore()
    for index in range(55):
        store.experiments[f"exp-{index:02d}"] = _proposal(
            f"exp-{index:02d}", "h" * 4_000
        )
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="pick-hidden",
                action=DecisionAction.IMPLEMENT,
                target_experiment_id="exp-54",
                rationale="not exposed",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.orchestrate(minimal_state(phase=RunPhase.RESEARCH))

    assert result["pending_route"] == "persist_failure"
    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert "pending_proposals" not in request.model_dump()
    assert request.experiment_history is not None
    assert request.experiment_history.total_pending_proposals == 55
    assert request.experiment_history.pending_truncated is True
    assert len(request.experiment_history.pending_proposals) == 50
    assert "exp-54" not in {
        proposal.experiment_id for proposal in request.experiment_history.pending_proposals
    }
    assert all(
        len(proposal.hypothesis) <= 2_000
        for proposal in request.experiment_history.pending_proposals
    )

async def test_orchestration_offers_empty_run_only_research() -> None:
    store = _FakeStore()
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="research-empty-run",
                action=DecisionAction.RESEARCH,
                rationale="Establish the first candidate",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.orchestrate(minimal_state(phase=RunPhase.RESEARCH))

    assert result["pending_route"] == "research"
    assert result["orchestration_decision_id"] == "research-empty-run"
    assert len(agent.calls) == 1
    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert request.allowed_actions == (DecisionAction.RESEARCH,)
    assert request.finalization_ready is False


async def test_orchestration_rejects_stop_without_finalization_authority() -> None:
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="stop-empty-run",
                action=DecisionAction.STOP,
                rationale="No frontier is available",
            )
        ]
    )
    controller = ProductionController(_make_services(agent_client=agent))

    result = await controller.orchestrate(minimal_state(phase=RunPhase.RESEARCH))

    assert result["pending_route"] == "persist_failure"
    assert "disallowed action: stop" in str(result["terminal_reason"])


async def test_orchestration_rejects_stop_before_deterministic_convergence() -> None:
    store = _FakeStore()
    store.sources["exp-1"] = SourceRegistration(
        experiment_id="exp-1",
        run_id="test-run",
        parent_commit="b" * 40,
        source_commit="a" * 40,
        patch_sha256="c" * 64,
        patch_artifact_id="patch-1",
        patch_artifact_uri="file:///tmp/patch-1.diff",
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    store.evaluations.append(_FakeEvaluator().result)
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="stop-finalizable-run",
                action=DecisionAction.STOP,
                target_experiment_id="exp-1",
                rationale="The evaluated candidate is sufficient",
            )
        ]
    )
    controller = ProductionController(
        _make_services(store=store, agent_client=agent, bundle_service=object())
    )

    result = await controller.orchestrate(
        minimal_state(
            phase=RunPhase.PERSIST,
            current_experiment_id="exp-1",
            latest_evaluation_result_id="eval-1",
        )
    )

    assert result["pending_route"] == "persist_failure"
    assert "disallowed action: stop" in str(result["terminal_reason"])
    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert request.finalization_ready is False
    assert DecisionAction.STOP not in request.allowed_actions


async def test_finalize_recovers_premature_stop_to_orchestration() -> None:
    controller = ProductionController(_make_services())

    result = await controller.finalize(
        minimal_state(
            phase=RunPhase.PERSIST,
            current_experiment_id="exp-1",
            latest_evaluation_result_id="eval-1",
            pending_route="finalize",
        )
    )

    assert result["phase"] == RunPhase.PERSIST
    assert result["pending_route"] == "orchestrate"
    assert result["state_version"] == 1


async def test_research_repairs_one_bad_response() -> None:
    """Research transition repairs one bad structured response then succeeds."""
    store = _FakeStore()
    agent = _ScriptedAgentClient([
        # First response: invalid (will fail schema validation)
        AgentFailure(
            request_id="req-1", role=AgentRole.RESEARCH,
            kind="schema", message="bad json", repair_attempts=0,
        ),
        # Second response: valid OrchestrationDecision
        OrchestrationDecision(
            decision_id="dec-1", action=DecisionAction.RESEARCH, rationale="test",
        ),
    ])
    services = _make_services(store=store, agent_client=agent)
    controller = ProductionController(services)

    state = minimal_state()
    await controller.research(state)
    # Should have persisted a transition
    assert len(store.persisted) >= 1
    # And routed to proposal_policy (since research completed)
    op = store.persisted[-1][1]
    assert op == "research"


async def test_model_unavailability_pauses_without_persisting_a_transition() -> None:
    store = _FakeStore()
    agent = _ScriptedAgentClient(
        [
            AgentFailure(
                request_id="orchestration-test-run-0",
                role=AgentRole.ORCHESTRATION,
                kind="model",
                message="429 Too Many Requests",
                repair_attempts=0,
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    with pytest.raises(ModelUnavailableError, match="429 Too Many Requests"):
        await controller.orchestrate(minimal_state(phase=RunPhase.IMPLEMENT))

    assert store.persisted == []
    assert store.failures == []


async def test_research_receives_authoritative_controller_context() -> None:
    store = _FakeStore()
    store.manifest = DatasetManifestIdentity(
        manifest_id="manifest-1", manifest_sha256="a" * 64
    )
    store.evaluator = EvaluatorIdentity(
        evaluator_id="evaluator-1", evaluator_sha256="b" * 64, validity="provisional"
    )
    dataset_context = DatasetContext(
        evidence_id="dataset-context-1",
        manifest_id="manifest-1",
        manifest_sha256="a" * 64,
        row_identity_columns=("row_id", "user_id", "item_id"),
        user_id_column="user_id",
        item_id_column="item_id",
        label_column="label",
        non_label_feature_columns=("duration_ms",),
        files=(
            DatasetFileContext(
                relative_path="train.csv",
                split="train",
                columns=("row_id", "user_id", "item_id", "duration_ms", "label"),
            ),
        ),
    )
    agent = _ScriptedAgentClient(
        [
            ResearchDecision(
                request_id="research-test-run-0",
                kind="proposal",
                experiment_spec=_estimated_experiment(("src/tiktok2026/experiment",)),
                message="bounded proposal",
            )
        ]
    )
    controller = ProductionController(
        _make_services(store=store, agent_client=agent, dataset_context=dataset_context)
    )

    result = await controller.research(
        minimal_state(
            phase=RunPhase.RESEARCH,
            current_experiment_id="old-exp",
            repair_attempts=2,
        )
    )

    assert result["pending_route"] == "research"
    assert result["repair_attempts"] == 0
    request = agent.calls[0]
    assert isinstance(request, ResearchRequest)
    assert request.controller_context is not None
    assert request.controller_context.dataset_manifest_identity == store.manifest
    assert request.controller_context.evaluator_identity == store.evaluator
    assert request.controller_context.source_commit_stage == "post_implementation"
    assert request.controller_context.experiment_registry is not None
    assert request.controller_context.experiment_registry.complete is True
    assert request.controller_context.dataset_context == dataset_context
    assert request.proposal_mode == "candidate_generation"
    assert request.proposal_batch_size == 3
    assert request.proposals_needed == 3


async def test_planning_requests_share_exact_parent_source_and_run_history() -> None:
    repository_root = Path(__file__).parents[2]
    parent_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    store = _FakeStore()
    agent = _ScriptedAgentClient(
        [
            OrchestrationDecision(
                decision_id="research-for-context",
                action=DecisionAction.RESEARCH,
                rationale="Need another bounded proposal",
            ),
            ResearchDecision(
                request_id="research-test-run-1",
                kind="proposal",
                experiment_spec=_estimated_experiment(("src/tiktok2026/experiment",)),
                message="bounded proposal",
            ),
        ]
    )
    controller = ProductionController(
        _make_services(
            store=store,
            agent_client=agent,
            repository_root=str(repository_root),
            parent_commit=parent_commit,
        )
    )
    state = minimal_state(phase=RunPhase.RESEARCH)

    orchestration = await controller.orchestrate(state)
    await controller.research(state | orchestration)  # type: ignore[arg-type]

    orchestration_request, research_request = agent.calls
    assert isinstance(orchestration_request, OrchestrationRequest)
    assert isinstance(research_request, ResearchRequest)
    assert orchestration_request.source_context is not None
    assert research_request.source_context == orchestration_request.source_context
    assert orchestration_request.source_context.source_commit == parent_commit
    assert orchestration_request.experiment_history is not None
    assert research_request.experiment_history is not None
    assert orchestration_request.experiment_history.total_records == 0


# ---------------------------------------------------------------------------
# Test 3: evaluate transition persists evaluation with provenance
# ---------------------------------------------------------------------------


class _FakeEvaluator:
    def __init__(self, result: EvaluationResult | None = None) -> None:
        self.result = result or EvaluationResult(
            evaluation_id="eval-1",
            experiment_id="exp-1",
            checkpoint_id="ckpt-1",
            metrics=(
                MetricValue(name="GAUC", value=0.5),
                MetricValue(name="nDCG@5", value=0.6),
            ),
            evaluator_artifact_id="evaluator-1",
            evaluator_sha256="0" * 64,
            prediction_sha256="1" * 64,
            validity="provisional",
        )

    def evaluate(self, request: Any) -> EvaluationResult:
        return self.result


class _CapturingExecutor:
    def __init__(self, *, fail_smoke: bool = False) -> None:
        self.request = None
        self.calls = 0
        self.fail_smoke = fail_smoke

    async def execute(self, request: Any) -> ExecutionResult:
        self.request = request
        self.calls += 1
        return ExecutionResult(
            execution_id=request.execution_id,
            experiment_id=request.experiment_id,
            source_registration_id=request.source_registration_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=1 if self.fail_smoke and request.execution_kind == "smoke" else 0,
            elapsed_seconds=0.1,
            gpu_hours=0.0,
            checkpoint_id="checkpoint-1",
            execution_kind=request.execution_kind,
            measured_peak_memory_bytes=1 << 20 if request.execution_kind == "smoke" else None,
            memory_measurement_status=(
                "measured" if request.execution_kind == "smoke" else "unavailable"
            ),
            resource_measurement_basis=(
                "docker_stats" if request.execution_kind == "smoke" else "unavailable"
            ),
            smoke_output_valid=request.execution_kind == "smoke",
            scientific_evidence=request.execution_kind != "smoke",
            dataset_manifest_id="manifest-1" if request.execution_kind == "smoke" else None,
            dataset_manifest_sha256="d" * 64 if request.execution_kind == "smoke" else None,
            dataset_view_sha256="e" * 64 if request.execution_kind == "smoke" else None,
            dataset_valid_rows=(
                DatasetViewRow(
                    row_id='["row-1","user-1","item-1"]',
                    row_identity=("row-1", "user-1", "item-1"),
                    user_id="user-1",
                    item_id="item-1",
                ),
            )
            if request.execution_kind == "smoke"
            else (),
            failure_kind=(
                FailureKind.SCHEMA_MISMATCH
                if self.fail_smoke and request.execution_kind == "smoke"
                else None
            ),
            failure_message=(
                "smoke failed" if self.fail_smoke and request.execution_kind == "smoke" else None
            ),
        )


class _FakeResourceAccountant:
    def __init__(self, *, fail_consume_once: bool = False) -> None:
        self.operations: list[str] = []
        self.fail_consume_once = fail_consume_once

    def state(self) -> ResourceState:
        return ResourceState(
            remaining_gpu_hours=10.0,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=10_000.0,
            used_tokens=0,
            remaining_tokens=10_000,
            disk_bytes_available=10_000_000_000,
            reserved_final_gpu_hours=0.0,
        )

    def reserve(self, reservation: Any) -> bool:
        self.operations.append(f"reserve:{reservation.reservation_id}")
        return True

    def consume(self, reservation_id: str, **usage: float | int) -> bool:
        del usage
        self.operations.append(f"consume:{reservation_id}")
        if self.fail_consume_once:
            self.fail_consume_once = False
            raise RuntimeError("settlement interrupted")
        return True

    def reconcile(self, reservation_id: str, **usage: float | int) -> bool:
        del usage
        self.operations.append(f"reconcile:{reservation_id}")
        return True

    def release(self, reservation_id: str) -> bool:
        self.operations.append(f"release:{reservation_id}")
        return True


def _populate_execution_authority(store: _FakeStore) -> str:
    source_commit = "a" * 40
    store.manifest = DatasetManifestIdentity(
        manifest_id="manifest-1", manifest_sha256="d" * 64
    )
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    store.sources["exp-1"] = SourceRegistration(
        experiment_id="exp-1",
        run_id="test-run",
        parent_commit="b" * 40,
        source_commit=source_commit,
        patch_sha256="c" * 64,
        patch_artifact_id="patch-1",
        patch_artifact_uri="file:///tmp/patch-1.diff",
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    store.assignments["exp-1"] = WorktreeAssignment(
        worktree_id="wt-1",
        run_id="test-run",
        experiment_id="exp-1",
        path=Path("/tmp/worktree/exp-1"),
        branch="experiment/test-run/exp-1",
        parent_commit="b" * 40,
    )
    return source_commit


async def test_execute_builds_only_deterministic_allowlisted_train_command(
    tmp_path: Path,
) -> None:
    store = _FakeStore()
    executor = _CapturingExecutor()
    source_commit = "a" * 40
    store.manifest = DatasetManifestIdentity(
        manifest_id="manifest-1", manifest_sha256="d" * 64
    )
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    store.sources["exp-1"] = SourceRegistration(
        experiment_id="exp-1",
        run_id="test-run",
        parent_commit="b" * 40,
        source_commit=source_commit,
        patch_sha256="c" * 64,
        patch_artifact_id="patch-1",
        patch_artifact_uri="file:///tmp/patch-1.diff",
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    store.assignments["exp-1"] = WorktreeAssignment(
        worktree_id="wt-1",
        run_id="test-run",
        experiment_id="exp-1",
        path=Path("/tmp/worktree/exp-1"),
        branch="experiment/test-run/exp-1",
        parent_commit="b" * 40,
    )
    accountant = _FakeResourceAccountant()
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
        default_timeout_seconds=123,
        default_memory_bytes=4_294_967_296,
        default_cpus=2.0,
        default_gpu_count=1,
    )
    controller = ProductionController(ControllerServices(services, store))

    result = await controller.execute(
        minimal_state(current_experiment_id="exp-1")
    )

    assert result["pending_route"] == "evaluate"
    assert executor.request is not None
    command = executor.request.command
    assert command[:3] == ("python", "-m", "tiktok2026.experiment.train")
    assert command[3:] == (
        "--output-dir=/output",
        command[4],
        "--fidelity=smoke",
        f"--source-commit={source_commit}",
        f"--execution-id={executor.request.execution_id}",
    )
    assert command[4].startswith("--seed=")
    assert command[4].removeprefix("--seed=").isdigit()
    assert not any(argument.startswith("--data-") for argument in command)
    assert executor.request.output_path.parent.name == ".execution"
    assert executor.request.output_path.is_dir()
    assert executor.request.output_path.stat().st_mode & 0o777 == 0o777
    assert executor.request.timeout_seconds == 123
    assert executor.request.memory_bytes == 4_294_967_296
    assert executor.request.cpus == 2.0
    assert executor.request.gpu_count == 1
    assert executor.request.dataset_manifest_sha256 == "d" * 64
    assert executor.request.execution_id in store.executions

    retried = await controller.execute(minimal_state(current_experiment_id="exp-1"))

    assert retried["pending_route"] == "evaluate"
    assert executor.calls == 1
    assert accountant.operations == [
        "reserve:reservation-execution-test-run-exp-1-0",
        "consume:reservation-execution-test-run-exp-1-0",
        "reconcile:reservation-execution-test-run-exp-1-0",
        "reserve:reservation-execution-test-run-exp-1-0",
        "consume:reservation-execution-test-run-exp-1-0",
        "reconcile:reservation-execution-test-run-exp-1-0",
    ]

    del store.executions[executor.request.execution_id]
    ambiguous = await controller.execute(minimal_state(current_experiment_id="exp-1"))

    assert ambiguous["pending_route"] == "persist_failure"
    assert "already claimed but its result is absent" in str(ambiguous["terminal_reason"])
    assert executor.calls == 1


async def test_execute_preparation_failure_releases_reservation_without_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FakeStore()
    _populate_execution_authority(store)
    executor = _CapturingExecutor()
    accountant = _FakeResourceAccountant()
    monkeypatch.setattr(
        use_cases,
        "_fresh_execution_output_path",
        lambda runtime_root, execution_id: (_ for _ in ()).throw(
            ValueError("output preparation failed")
        ),
    )
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
    )

    result = await ProductionController(ControllerServices(services, store)).execute(
        minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)
    )

    assert result["pending_route"] == "persist_failure"
    assert store.count_full_attempt_claims("test-run") == 0
    assert accountant.operations == [
        "reserve:reservation-execution-test-run-exp-1-0",
        "release:reservation-execution-test-run-exp-1-0",
    ]
    assert executor.calls == 0


async def test_execute_request_validation_failure_releases_reservation_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FakeStore()
    _populate_execution_authority(store)
    executor = _CapturingExecutor()
    accountant = _FakeResourceAccountant()

    def reject_request(**kwargs: object) -> None:
        del kwargs
        raise ValueError("request validation failed")

    monkeypatch.setattr(use_cases, "ExecutionRequest", reject_request)
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
    )

    result = await ProductionController(ControllerServices(services, store)).execute(
        minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)
    )

    assert result["pending_route"] == "persist_failure"
    assert store.count_full_attempt_claims("test-run") == 0
    assert accountant.operations == [
        "reserve:reservation-execution-test-run-exp-1-0",
        "release:reservation-execution-test-run-exp-1-0",
    ]
    assert tuple((tmp_path / "runtime" / "artifacts" / ".execution").iterdir()) == ()
    assert executor.calls == 0


async def test_execute_never_dispatches_a_fifty_first_full_attempt(tmp_path: Path) -> None:
    store = _FakeStore()
    _populate_execution_authority(store)
    for index in range(50):
        claim = FullAttemptClaimRequest(
            attempt_id=f"attempt-existing-{index}",
            execution_id=f"execution-existing-{index}",
            run_id="test-run",
            experiment_id="exp-1",
            source_registration_id=store.sources["exp-1"].registration_id,
            source_commit=store.sources["exp-1"].source_commit,
        )
        assert store.claim_full_attempt(claim) is not None
    executor = _CapturingExecutor()
    accountant = _FakeResourceAccountant()
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
    )

    result = await ProductionController(ControllerServices(services, store)).execute(
        minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)
    )

    assert result["pending_route"] == "persist_failure"
    assert executor.calls == 0
    assert store.count_full_attempt_claims("test-run") == 50
    assert accountant.operations[-1] == "release:reservation-execution-test-run-exp-1-0"


async def test_smoke_runs_before_full_execution_with_distinct_identity(tmp_path: Path) -> None:
    store = _FakeStore()
    source_commit = _populate_execution_authority(store)
    executor = _CapturingExecutor()
    accountant = _FakeResourceAccountant()
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
        docker_image="tiktok2026:test@sha256:" + "0" * 64,
        dataset_view_provenance=lambda request: DatasetViewProvenance(
            manifest_id="manifest-1",
            manifest_sha256="d" * 64,
            view_sha256="e" * 64,
            valid_rows=(
                DatasetViewRow(
                    row_id='["row-1","user-1","item-1"]',
                    row_identity=("row-1", "user-1", "item-1"),
                    user_id="user-1",
                    item_id="item-1",
                ),
            ),
        ),
    )
    controller = ProductionController(ControllerServices(services, store))
    state = minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)

    smoke = await controller.smoke(state)
    smoke_operations = tuple(accountant.operations)
    full = await controller.execute(state)

    assert smoke["pending_route"] == "execute"
    assert full["pending_route"] == "evaluate"
    assert executor.calls == 2
    assert executor.request is not None
    assert executor.request.execution_kind == "full"
    assert executor.request.source_commit == source_commit
    smoke_ids = tuple(
        execution_id for execution_id in store.executions if execution_id.startswith("smoke-")
    )
    assert len(smoke_ids) == 1
    assert smoke_ids[0] != executor.request.execution_id
    assert smoke_operations == (
        f"reserve:reservation-{smoke_ids[0]}",
        f"consume:reservation-{smoke_ids[0]}",
        f"reconcile:reservation-{smoke_ids[0]}",
    )


async def test_smoke_failure_does_not_route_to_evaluation(tmp_path: Path) -> None:
    store = _FakeStore()
    _populate_execution_authority(store)
    executor = _CapturingExecutor(fail_smoke=True)
    accountant = _FakeResourceAccountant()
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
        dataset_view_provenance=lambda request: DatasetViewProvenance(
            manifest_id="manifest-1",
            manifest_sha256="d" * 64,
            view_sha256="e" * 64,
            valid_rows=(),
        ),
    )
    controller = ProductionController(ControllerServices(services, store))

    result = await controller.smoke(
        minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)
    )

    assert result["pending_route"] == "persist_failure"
    assert executor.calls == 1
    assert result.get("latest_execution_result_id") is None


async def test_smoke_requires_resource_accountant(tmp_path: Path) -> None:
    store = _FakeStore()
    _populate_execution_authority(store)
    executor = _CapturingExecutor()
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        dataset_view_provenance=lambda request: DatasetViewProvenance(
            manifest_id="manifest-1",
            manifest_sha256="d" * 64,
            view_sha256="e" * 64,
            valid_rows=(),
        ),
    )
    controller = ProductionController(ControllerServices(services, store))

    result = await controller.smoke(
        minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)
    )

    assert result["pending_route"] == "persist_failure"
    assert "incomplete" in str(result["terminal_reason"])
    assert executor.calls == 0


async def test_smoke_settlement_replays_after_result_persistence(tmp_path: Path) -> None:
    store = _FakeStore()
    _populate_execution_authority(store)
    executor = _CapturingExecutor()
    accountant = _FakeResourceAccountant(fail_consume_once=True)
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        resource_accountant=accountant,
        dataset_view_provenance=lambda request: DatasetViewProvenance(
            manifest_id="manifest-1",
            manifest_sha256="d" * 64,
            view_sha256="e" * 64,
            valid_rows=(
                DatasetViewRow(
                    row_id='["row-1","user-1","item-1"]',
                    row_identity=("row-1", "user-1", "item-1"),
                    user_id="user-1",
                    item_id="item-1",
                ),
            ),
        ),
    )
    controller = ProductionController(ControllerServices(services, store))
    state = minimal_state(current_experiment_id="exp-1", phase=RunPhase.EXECUTE)

    with pytest.raises(RuntimeError, match="settlement interrupted"):
        await controller.smoke(state)
    replay = await controller.smoke(state)

    assert replay["pending_route"] == "execute"
    assert executor.calls == 1
    assert accountant.operations == [
        "reserve:reservation-smoke-test-run-exp-1-0",
        "consume:reservation-smoke-test-run-exp-1-0",
        "reserve:reservation-smoke-test-run-exp-1-0",
        "consume:reservation-smoke-test-run-exp-1-0",
        "reconcile:reservation-smoke-test-run-exp-1-0",
    ]


def test_execution_output_directories_are_distinct_and_writable(tmp_path: Path) -> None:
    from tiktok2026.use_cases import _fresh_execution_output_path

    first = _fresh_execution_output_path(str(tmp_path), "execution-1")
    second = _fresh_execution_output_path(str(tmp_path), "execution-2")
    assert first != second
    assert first.is_relative_to(tmp_path)
    assert second.is_relative_to(tmp_path)
    assert first.stat().st_mode & 0o777 == 0o777
    assert second.stat().st_mode & 0o777 == 0o777
    assert not tuple(first.iterdir())
    assert not tuple(second.iterdir())


async def test_evaluate_persists_evaluation_with_provenance() -> None:
    """Evaluate transition persists evaluation and routes to result_validation."""
    store = _FakeStore()
    evaluator = _FakeEvaluator()
    source_commit = "a" * 40
    store.executions["exec-1"] = ExecutionResult(
        execution_id="exec-1",
        experiment_id="exp-1",
        source_commit=source_commit,
        command=("python", "train.py"),
        exit_code=0,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        artifact_ids=(
            "stdout-1",
            "stderr-1",
            "resource-evidence-1",
            "prediction-1",
            "ckpt-1",
        ),
        checkpoint_id="ckpt-1",
    )
    store.sources["exp-1"] = SourceRegistration(
        experiment_id="exp-1",
        run_id="test-run",
        parent_commit="b" * 40,
        source_commit=source_commit,
        patch_sha256="c" * 64,
        patch_artifact_id="patch-1",
        patch_artifact_uri="file:///tmp/patch-1.diff",
        allowed_scopes=("src/tiktok2026/experiment",),
        eligible=True,
    )
    store.manifest = DatasetManifestIdentity(
        manifest_id="manifest-1", manifest_sha256="d" * 64
    )
    store.evaluator = EvaluatorIdentity(
        evaluator_id="evaluator-1", evaluator_sha256="0" * 64, validity="provisional"
    )
    store.predictions["prediction-1"] = PredictionArtifactRegistration(
        artifact_id="prediction-1",
        path=Path("/tmp/prediction-1.json"),
        sha256="1" * 64,
        checkpoint_id="ckpt-1",
        source_commit=source_commit,
        execution_id="exec-1",
        dataset_manifest_id="manifest-1",
        dataset_manifest_sha256="d" * 64,
        split="valid",
    )
    store.artifacts["prediction-1"] = ArtifactRecord(
        artifact_id="prediction-1",
        run_id="test-run",
        experiment_id="exp-1",
        kind="prediction",
        uri="file:///tmp/prediction-1.json",
        sha256="1" * 64,
        size_bytes=1,
        producer="test",
        retention=ArtifactRetention.RUN,
    )
    store.artifacts["ckpt-1"] = ArtifactRecord(
        artifact_id="ckpt-1",
        run_id="test-run",
        experiment_id="exp-1",
        kind="checkpoint",
        uri="file:///tmp/checkpoint-1.json",
        sha256="2" * 64,
        size_bytes=1,
        producer="test",
        retention=ArtifactRetention.RUN,
    )
    services = _make_services(store=store, evaluator=evaluator)
    controller = ProductionController(services)

    state = minimal_state(
        current_experiment_id="exp-1",
        latest_execution_result_id="exec-1",
    )
    result = await controller.evaluate(state)
    assert result["latest_evaluation_result_id"] is not None
    assert result["pending_route"] == "result_validation"


async def test_result_validation_excludes_authoritative_dataset_rows() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    store.executions["exec-1"] = ExecutionResult(
        execution_id="exec-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "train.py"),
        exit_code=0,
        elapsed_seconds=1.0,
        gpu_hours=0.0,
        dataset_valid_rows=(
            DatasetViewRow(
                row_id="row-1",
                row_identity=("row-1", "user-1", "item-1"),
                user_id="user-1",
                item_id="item-1",
            ),
        ),
    )
    store.evaluations.append(
        EvaluationResult(
            evaluation_id="eval-1",
            experiment_id="exp-1",
            checkpoint_id="ckpt-1",
            metrics=(
                MetricValue(name="GAUC", value=0.5),
                MetricValue(name="nDCG@5", value=0.6),
            ),
            evaluator_artifact_id="evaluator-1",
            evaluator_sha256="b" * 64,
            prediction_sha256="c" * 64,
            validity="provisional",
        )
    )
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="report-1",
                experiment_id="exp-1",
                stage=ValidationStage.RESULT,
                verdict=ValidationVerdict.APPROVED,
                leakage_risk="none",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.result_validation(
        minimal_state(
            current_experiment_id="exp-1",
            latest_execution_result_id="exec-1",
            latest_evaluation_result_id="eval-1",
            phase=RunPhase.EVALUATE,
        )
    )

    assert result["pending_route"] == "persist"
    request = agent.calls[0]
    assert isinstance(request, ValidationRequest)
    execution = request.subject["execution_result"]
    assert isinstance(execution, dict)
    assert "dataset_valid_rows" not in execution
    history = request.subject["experiment_history"]
    assert isinstance(history, dict)
    assert history["run_id"] == "test-run"


async def test_evaluate_logs_metrics_and_delta_from_prior_compatible_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    current = EvaluationResult(
        evaluation_id="eval-current",
        experiment_id="exp-1",
        checkpoint_id="ckpt-1",
        metrics=(
            MetricValue(name="GAUC", value=0.6),
            MetricValue(name="nDCG@5", value=0.7),
        ),
        evaluator_artifact_id="evaluator-1",
        evaluator_sha256="0" * 64,
        prediction_sha256="1" * 64,
        validity="provisional",
        dataset_manifest_sha256="d" * 64,
        split="valid",
        run_id="test-run",
    )
    store.evaluations.append(
        current.model_copy(
            update={
                "evaluation_id": "eval-baseline",
                "experiment_id": "exp-baseline",
                "metrics": (
                    MetricValue(name="GAUC", value=0.55),
                    MetricValue(name="nDCG@5", value=0.68),
                ),
            }
        )
    )

    import tiktok2026.use_cases as use_cases

    messages: list[str] = []
    monkeypatch.setattr(
        use_cases.logger,
        "info",
        lambda message, *arguments: messages.append(message.format(*arguments)),
    )
    use_cases._log_evaluation_metrics(ServiceTransitions(run_store=store), current)

    output = messages[-1]
    assert "GAUC=0.600000" in output
    assert "nDCG@5=0.700000" in output
    assert "baseline_evaluation_id=eval-baseline" in output
    assert "delta_GAUC=+0.050000" in output
    assert "delta_nDCG@5=+0.020000" in output


async def test_evaluation_log_prefers_starter_kit_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    current = EvaluationResult(
        evaluation_id="eval-current",
        experiment_id="exp-1",
        checkpoint_id="ckpt-1",
        metrics=(
            MetricValue(name="GAUC", value=0.6),
            MetricValue(name="nDCG@5", value=0.7),
        ),
        evaluator_artifact_id="evaluator-1",
        evaluator_sha256="0" * 64,
        prediction_sha256="1" * 64,
        validity="provisional",
        dataset_manifest_sha256="d" * 64,
        split="valid",
        run_id="test-run",
        dataset_manifest_id="manifest-1",
    )
    baseline_evaluation = current.model_copy(
        update={
            "evaluation_id": "eval-starter",
            "experiment_id": "starter-kit-fm-baseline",
            "metrics": (
                MetricValue(name="GAUC", value=0.56),
                MetricValue(name="nDCG@5", value=0.69),
            ),
        }
    )
    store.baseline_calibrations.append(
        BaselineCalibrationRecord(
            calibration_id="baseline-calibration-test",
            dataset_manifest_id="manifest-1",
            dataset_manifest_sha256="d" * 64,
            evaluator_id="evaluator-1",
            evaluator_sha256="0" * 64,
            baseline_source_sha256="2" * 64,
            config_sha256="3" * 64,
            prediction_sha256="1" * 64,
            prediction_artifact_uri="file:///tmp/predictions.json",
            evaluation=baseline_evaluation,
            diagnostic_metrics=(
                DiagnosticMetricValue(name="GAUC", value=0.66),
                DiagnosticMetricValue(name="nDCG@5", value=0.53),
                DiagnosticMetricValue(name="primary", value=0.595),
            ),
        )
    )

    import tiktok2026.use_cases as use_cases

    messages: list[str] = []
    monkeypatch.setattr(
        use_cases.logger,
        "info",
        lambda message, *arguments: messages.append(message.format(*arguments)),
    )
    use_cases._log_evaluation_metrics(ServiceTransitions(run_store=store), current)

    output = messages[-1]
    assert "baseline=starter_kit_fm" in output
    assert "baseline_calibration_id=baseline-calibration-test" in output
    assert "delta_GAUC=+0.040000" in output
    assert "delta_nDCG@5=+0.010000" in output


def test_production_evaluation_requires_run_baseline_authority() -> None:
    result = EvaluationResult(
        evaluation_id="eval-1",
        experiment_id="exp-1",
        checkpoint_id="ckpt-1",
        metrics=(
            MetricValue(name="GAUC", value=0.6),
            MetricValue(name="nDCG@5", value=0.7),
        ),
        evaluator_artifact_id="configured-evaluator",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
        run_id="run-1",
    )

    with pytest.raises(use_cases.MissingAuthorityError, match="baseline authority"):
        use_cases._log_evaluation_metrics(
            ServiceTransitions(
                run_store=cast(Any, _FakeStore()), requires_run_baseline=True
            ),
            result,
        )


def test_production_evaluation_rejects_historical_metric_pair() -> None:
    result = EvaluationResult(
        evaluation_id="eval-legacy",
        experiment_id="exp-1",
        checkpoint_id="ckpt-1",
        metrics=(
            MetricValue(name="NDCG@10", value=0.6),
            MetricValue(name="Recall@50", value=0.7),
        ),
        evaluator_artifact_id="configured-evaluator",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
        run_id="run-1",
    )

    with pytest.raises(use_cases.MissingAuthorityError, match="current judging metric pair"):
        use_cases._log_evaluation_metrics(
            ServiceTransitions(
                run_store=cast(Any, _FakeStore()), requires_run_baseline=True
            ),
            result,
        )


# ---------------------------------------------------------------------------
# Test 4: persist_failure routes repair vs persist_failure by kind/count
# ---------------------------------------------------------------------------


async def test_persist_failure_repairs_then_abandons_exhausted_experiment() -> None:
    """A repairable experiment failure cannot terminate the research run."""
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    services = _make_services(store=store)
    controller = ProductionController(services)

    # Repairable failure with 0 attempts → should route to "repair"
    state_repairable = minimal_state(
        phase=RunPhase.IMPLEMENT,
        current_experiment_id="exp-1",
        repair_attempts=0,
    )
    result = await controller.persist_failure(state_repairable)
    # The transition should record a failure and set pending_route
    assert result["pending_route"] in ("repair", "persist_failure", "orchestrate")

    # With 3 attempts, repair is exhausted and the experiment is abandoned.
    state_exhausted = minimal_state(
        phase=RunPhase.IMPLEMENT,
        current_experiment_id="exp-1",
        repair_attempts=3,
    )
    result = await controller.persist_failure(state_exhausted)

    assert result["pending_route"] == "orchestrate"
    assert result["current_experiment_id"] is None
    assert result["current_hypothesis_id"] is None
    assert result["repair_attempts"] == 0
    assert store.experiment_updates[-1] == ("exp-1", "failed")


async def test_research_after_abandonment_starts_fresh_experiment() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    proposal = _estimated_experiment(("src/tiktok2026/experiment",)).model_copy(
        update={"experiment_id": "exp-2", "parent_experiment_id": None}
    )
    agent = _ScriptedAgentClient(
        [
            ResearchDecision(
                request_id="research-1",
                kind="proposal",
                experiment_spec=proposal,
                message="fresh proposal",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))
    failed = minimal_state(
        phase=RunPhase.RESEARCH,
        current_experiment_id="exp-1",
        current_hypothesis_id="hyp-1",
        repair_attempts=3,
        terminal_reason=(
            'failure:{"evidence": [], "kind": "schema_mismatch", '
            '"message": "proposal rejected"}'
        ),
    )

    abandoned = await controller.persist_failure(failed)
    result = await controller.research(failed | abandoned)  # type: ignore[arg-type]

    request = agent.calls[0]
    assert isinstance(request, ResearchRequest)
    assert request.parent_experiment_id is None
    assert result["current_experiment_id"] == "exp-2"
    assert result["repair_attempts"] == 0


async def test_rejected_implementation_is_abandoned_then_orchestrated() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    controller = ProductionController(_make_services(store=store))
    state = minimal_state(
        phase=RunPhase.IMPLEMENT,
        current_experiment_id="exp-1",
        terminal_reason=(
            'failure:{"evidence": ["validation-1"], '
            '"kind": "unstable_validation", "message": "implementation rejected"}'
        ),
    )

    result = await controller.persist_failure(state)

    assert result["pending_route"] == "orchestrate"
    assert store.experiment_updates[-1] == ("exp-1", "failed")


async def test_pre_experiment_failure_does_not_route_to_implementor() -> None:
    store = _FakeStore()
    controller = ProductionController(_make_services(store=store))
    state = minimal_state(
        terminal_reason=(
            'failure:{"evidence": [], "kind": "schema_mismatch", '
            '"message": "malformed research response"}'
        )
    )

    with pytest.raises(TerminalLifecycleError, match="malformed research response"):
        await controller.persist_failure(state)

    assert len(store.failures) == 1
    assert not store.persisted


# ---------------------------------------------------------------------------
# Test 5: End-to-end synthetic composition runs through compiled graph
# ---------------------------------------------------------------------------


async def test_synthetic_full_graph_persists_every_transition(tmp_path: Path) -> None:
    """End-to-end synthetic run through compiled graph, persisting every transition."""
    import shutil

    from tiktok2026.bootstrap import build_synthetic_controller

    repo_root = Path(__file__).parents[2]
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "migrations", test_repo_root / "migrations")

    _controller, _store, graph = build_synthetic_controller(
        repository_root=test_repo_root,
        runtime_root=tmp_path / "runtime",
    )
    assert graph is not None

    initial: ProductionState = {
        "run_id": "test-run",
        "phase": RunPhase.BOOTSTRAP,
        "current_experiment_id": None,
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
        "state_version": 0,
    }

    result = await graph.ainvoke(initial, {"configurable": {"thread_id": "test-run"}})  # type: ignore[union-attr]
    assert result["phase"] == RunPhase.COMPLETE
    assert result["pending_route"] == "complete"
    assert any(
        '"decision":"converge"' in value for value in _store.list_json("frontier_decision")
    )
    assert any(
        '"run_id":"test-run"' in value for value in _store.list_json("frontier_policy")
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_services(
    store: Any = None,
    agent_client: Any = None,
    evaluator: Any = None,
    executor: Any = None,
    policy_gate: Any = None,
    bundle_service: Any = None,
    resource_accountant: Any = None,
    default_timeout_seconds: int = 300,
    default_memory_bytes: int = 1024**3,
    default_cpus: float = 1.0,
    default_gpu_count: int = 0,
    repository_root: str | None = None,
    parent_commit: str | None = None,
    dataset_context: DatasetContext | None = None,
) -> ControllerServices:
    """Build real service-driven transitions with fake injected services."""
    if store is None:
        store = _FakeStore()

    # Build transitions with the injected fakes
    transitions = make_service_transitions(
        agent_client=agent_client or _ScriptedAgentClient([]),
        evaluator=evaluator or _FakeEvaluator(),
        worktree_manager=None,
        executor=executor,
        policy_gate=policy_gate,
        bundle_service=bundle_service,
        resource_accountant=resource_accountant,
        run_store=store,
        evaluator_id="evaluator-1",
        repository_root=repository_root,
        parent_commit=parent_commit,
        dataset_context=dataset_context,
        repository_inspector=(
            RepositoryInspector(Path(repository_root)) if repository_root is not None else None
        ),
        default_timeout_seconds=default_timeout_seconds,
        default_memory_bytes=default_memory_bytes,
        default_cpus=default_cpus,
        default_gpu_count=default_gpu_count,
    )
    return ControllerServices(
        transitions=transitions,
        store=store,
    )
