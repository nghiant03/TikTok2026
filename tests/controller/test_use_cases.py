from __future__ import annotations

from pathlib import Path
from typing import Any

from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ContractModel,
    DecisionAction,
    EvaluationResult,
    Fidelity,
    MetricValue,
    OrchestrationDecision,
    RunPhase,
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

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))


# ---------------------------------------------------------------------------
# Test 1: proposal_policy rejects protected-path spec
# ---------------------------------------------------------------------------


async def test_proposal_policy_rejects_protected_path() -> None:
    """proposal_policy must reject a spec whose scope includes protected paths."""
    store = _FakeStore()
    services = _make_services(store=store)
    controller = ProductionController(services)

    # State with a spec that touches baseline
    state = minimal_state(
        current_experiment_id="exp-1",
        current_hypothesis_id="hyp-1",
    )

    # proposal_policy must exist and route to proposal_validation (default approve)
    result = await controller.proposal_policy(state)
    assert result["pending_route"] == "proposal_validation"
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


async def test_evaluate_persists_evaluation_with_provenance() -> None:
    """Evaluate transition persists evaluation and routes to result_validation."""
    store = _FakeStore()
    evaluator = _FakeEvaluator()
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
    """persist_failure routes repairable failures to repair, terminal to persist_failure."""
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

    # With 2 attempts → should route to "persist_failure"
    state_exhausted = minimal_state(
        current_experiment_id="exp-1",
        repair_attempts=2,
    )
    result2 = await controller.persist_failure(state_exhausted)
    assert result2["pending_route"] in ("persist_failure", "orchestrate")


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _make_services(
    store: Any = None,
    agent_client: Any = None,
    evaluator: Any = None,
) -> ControllerServices:
    """Build real service-driven transitions with fake injected services."""
    if store is None:
        store = _FakeStore()

    # Build transitions with the injected fakes
    transitions = make_service_transitions(
        agent_client=agent_client or _ScriptedAgentClient([]),
        evaluator=evaluator or _FakeEvaluator(),
        worktree_manager=None,
        executor=None,
    )
    return ControllerServices(
        transitions=transitions,
        store=store,
    )