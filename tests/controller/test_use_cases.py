from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tiktok2026.adapters import DeterministicPolicyGate
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    BaselineCalibrationRecord,
    ContractModel,
    DatasetManifestIdentity,
    DecisionAction,
    DiagnosticMetricValue,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionResult,
    ExperimentRegistryEntry,
    ExperimentRegistrySnapshot,
    ExperimentSpec,
    Fidelity,
    ImplementationAttemptRecord,
    ImplementationRequest,
    ImplementationResult,
    MetricValue,
    OrchestrationDecision,
    OrchestrationRequest,
    PredictionArtifactRegistration,
    ResearchDecision,
    ResearchRequest,
    RunPhase,
    SourceRegistration,
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
        self.evaluator: EvaluatorIdentity | None = None
        self.manifest: DatasetManifestIdentity | None = None
        self.evaluations: list[EvaluationResult] = []
        self.baseline_calibrations: list[BaselineCalibrationRecord] = []
        self.failures: list[object] = []
        self.json_records: dict[tuple[str, str], str] = {}

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        return self.experiments.get(experiment_id)

    def get_experiment_registry(self, limit: int = 50) -> ExperimentRegistrySnapshot:
        specs = tuple(self.experiments.values())[-limit:]
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
            total_experiments=len(self.experiments),
            complete=len(self.experiments) <= limit,
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
    assert result["latest_validation_report_id"] == "report-1"
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
    assert registry["total_experiments"] == 1


async def test_repairable_proposal_returns_to_research_with_feedback() -> None:
    store = _FakeStore()
    store.experiments["exp-1"] = _experiment(("src/tiktok2026/experiment",))
    agent = _ScriptedAgentClient(
        [
            ValidationReport(
                report_id="report-1",
                experiment_id="exp-1",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.REPAIRABLE,
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
            repair_attempts=2,
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
        ]
    )
    agent.scoped_repository = _FakeScopedRepository(
        "diff --git a/train.py b/train.py\n",
        ("src/tiktok2026/experiment/train.py",),
        source="unchanged entrypoint\n",
        base_source="unchanged entrypoint\n",
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

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
    assert first_request.source_context == {
        "src/tiktok2026/experiment/train.py": "unchanged entrypoint\n"
    }
    assert first_request.base_source_context == {}


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
        ]
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
    assert request.source_context == {
        "src/tiktok2026/experiment/train.py": "current repaired source\n"
    }
    assert request.base_source_context == {
        "src/tiktok2026/experiment/train.py": "committed controller entrypoint\n"
    }
    assert result["terminal_reason"] is None


# ---------------------------------------------------------------------------
# Test 2: research transition repairs one bad structured response
# ---------------------------------------------------------------------------


class _ScriptedAgentClient:
    """AgentClient that returns scripted responses, failing once then succeeding."""

    def __init__(self, responses: list[ContractModel]) -> None:
        self.responses = list(responses)
        self.calls: list[ContractModel] = []
        self.scoped_repository: _FakeScopedRepository | None = None

    async def invoke(self, request: ContractModel) -> ContractModel:
        self.calls.append(request)
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


async def test_orchestration_allows_stop_with_finalization_authority() -> None:
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

    assert result["pending_route"] == "finalize"
    request = agent.calls[0]
    assert isinstance(request, OrchestrationRequest)
    assert request.finalization_ready is True
    assert DecisionAction.STOP in request.allowed_actions


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
    agent = _ScriptedAgentClient(
        [
            ResearchDecision(
                request_id="research-test-run-0",
                kind="proposal",
                experiment_spec=_experiment(("src/tiktok2026/experiment",)),
                message="bounded proposal",
            )
        ]
    )
    controller = ProductionController(_make_services(store=store, agent_client=agent))

    result = await controller.research(
        minimal_state(
            phase=RunPhase.RESEARCH,
            current_experiment_id="old-exp",
            repair_attempts=2,
        )
    )

    assert result["pending_route"] == "proposal_policy"
    assert result["repair_attempts"] == 0
    request = agent.calls[0]
    assert isinstance(request, ResearchRequest)
    assert request.controller_context is not None
    assert request.controller_context.dataset_manifest_identity == store.manifest
    assert request.controller_context.evaluator_identity == store.evaluator
    assert request.controller_context.source_commit_stage == "post_implementation"
    assert request.controller_context.experiment_registry is not None
    assert request.controller_context.experiment_registry.complete is True


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
                MetricValue(name="NDCG@10", value=0.5),
                MetricValue(name="Recall@50", value=0.6),
            ),
            evaluator_artifact_id="evaluator-1",
            evaluator_sha256="0" * 64,
            prediction_sha256="1" * 64,
            validity="provisional",
        )

    def evaluate(self, request: Any) -> EvaluationResult:
        return self.result


class _CapturingExecutor:
    def __init__(self) -> None:
        self.request = None

    async def execute(self, request: Any) -> ExecutionResult:
        self.request = request
        return ExecutionResult(
            execution_id=request.execution_id,
            experiment_id=request.experiment_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=0,
            elapsed_seconds=0.1,
            gpu_hours=0.0,
            checkpoint_id="checkpoint-1",
        )


async def test_execute_builds_only_deterministic_allowlisted_train_command(
    tmp_path: Path,
) -> None:
    store = _FakeStore()
    executor = _CapturingExecutor()
    source_commit = "a" * 40
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
    services = make_service_transitions(
        executor=executor,
        run_store=store,
        dataset_root="/external/readonly-dataset",
        runtime_root=str(tmp_path / "runtime"),
        default_timeout_seconds=123,
        default_memory_bytes=4_294_967_296,
        default_cpus=2.0,
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
    assert executor.request.execution_id in store.executions


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
        artifact_ids=("prediction-1",),
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
    services = _make_services(store=store, evaluator=evaluator)
    controller = ProductionController(services)

    state = minimal_state(
        current_experiment_id="exp-1",
        latest_execution_result_id="exec-1",
    )
    result = await controller.evaluate(state)
    assert result["latest_evaluation_result_id"] is not None
    assert result["pending_route"] == "result_validation"


async def test_evaluate_logs_metrics_and_delta_from_prior_compatible_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    current = EvaluationResult(
        evaluation_id="eval-current",
        experiment_id="exp-1",
        checkpoint_id="ckpt-1",
        metrics=(
            MetricValue(name="NDCG@10", value=0.6),
            MetricValue(name="Recall@50", value=0.7),
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
                    MetricValue(name="NDCG@10", value=0.55),
                    MetricValue(name="Recall@50", value=0.68),
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
    assert "NDCG@10=0.600000" in output
    assert "Recall@50=0.700000" in output
    assert "baseline_evaluation_id=eval-baseline" in output
    assert "delta_NDCG@10=+0.050000" in output
    assert "delta_Recall@50=+0.020000" in output


async def test_evaluation_log_prefers_starter_kit_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeStore()
    current = EvaluationResult(
        evaluation_id="eval-current",
        experiment_id="exp-1",
        checkpoint_id="ckpt-1",
        metrics=(
            MetricValue(name="NDCG@10", value=0.6),
            MetricValue(name="Recall@50", value=0.7),
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
                MetricValue(name="NDCG@10", value=0.56),
                MetricValue(name="Recall@50", value=0.69),
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
    assert "delta_NDCG@10=+0.040000" in output
    assert "delta_Recall@50=+0.010000" in output


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

    # With 2 attempts, repair is exhausted and the experiment is abandoned.
    state_exhausted = minimal_state(
        phase=RunPhase.IMPLEMENT,
        current_experiment_id="exp-1",
        repair_attempts=2,
    )
    result = await controller.persist_failure(state_exhausted)

    assert result["pending_route"] == "orchestrate"
    assert store.experiment_updates[-1] == ("exp-1", "failed")


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
        run_store=store,
        evaluator_id="evaluator-1",
    )
    return ControllerServices(
        transitions=transitions,
        store=store,
    )
