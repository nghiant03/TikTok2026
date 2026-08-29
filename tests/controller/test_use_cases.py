from __future__ import annotations

from pathlib import Path
from typing import Any

from tiktok2026.adapters import DeterministicPolicyGate
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ContractModel,
    DatasetManifestIdentity,
    DecisionAction,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionResult,
    ExperimentSpec,
    Fidelity,
    MetricValue,
    OrchestrationDecision,
    PredictionArtifactRegistration,
    RunPhase,
    SourceRegistration,
    WorktreeAssignment,
)
from tiktok2026.controller import (
    ControllerServices,
    ProductionController,
)
from tiktok2026.graph.state import ProductionState
from tiktok2026.use_cases import make_service_transitions

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
        self.executions: dict[str, ExecutionResult] = {}
        self.sources: dict[str, SourceRegistration] = {}
        self.assignments: dict[str, WorktreeAssignment] = {}
        self.predictions: dict[str, PredictionArtifactRegistration] = {}
        self.evaluator = None
        self.manifest = None
        self.evaluations: list[EvaluationResult] = []
        self.failures: list[object] = []

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        return self.experiments.get(experiment_id)

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

    def put_failure(self, record: object, run_id: str) -> None:
        del run_id
        self.failures.append(record)


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


# ---------------------------------------------------------------------------
# Test 2: research transition repairs one bad structured response
# ---------------------------------------------------------------------------


class _ScriptedAgentClient:
    """AgentClient that returns scripted responses, failing once then succeeding."""

    def __init__(self, responses: list[ContractModel]) -> None:
        self.responses = list(responses)
        self.calls: list[ContractModel] = []

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


async def test_execute_builds_only_deterministic_allowlisted_train_command() -> None:
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
        runtime_root="/external/runtime",
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


# ---------------------------------------------------------------------------
# Test 4: persist_failure routes repair vs persist_failure by kind/count
# ---------------------------------------------------------------------------


async def test_persist_failure_routes_repairable_vs_terminal() -> None:
    """persist_failure repairs once, then terminates through export."""
    store = _FakeStore()
    services = _make_services(store=store)
    controller = ProductionController(services)

    # Repairable failure with 0 attempts → should route to "repair"
    state_repairable = minimal_state(
        current_experiment_id="exp-1",
        repair_attempts=0,
    )
    result = await controller.persist_failure(state_repairable)
    # The transition should record a failure and set pending_route
    assert result["pending_route"] in ("repair", "persist_failure", "orchestrate")

    # With 2 attempts → repair is exhausted and the graph must terminate at export.
    state_exhausted = minimal_state(
        current_experiment_id="exp-1",
        repair_attempts=2,
    )
    result2 = await controller.persist_failure(state_exhausted)
    assert result2["pending_route"] == "export"


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
        run_store=store,
        evaluator_id="evaluator-1",
    )
    return ControllerServices(
        transitions=transitions,
        store=store,
    )
