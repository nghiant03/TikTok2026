from __future__ import annotations

import pytest

from tiktok2026.contracts import Fidelity, RunPhase
from tiktok2026.controller import ProductionController
from tiktok2026.graph.build import build_production_graph
from tiktok2026.graph.nodes import ControllerOperations
from tiktok2026.graph.state import ProductionState


class RecordingController:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def bootstrap(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("bootstrap")
        return {"phase": RunPhase.BOOTSTRAP}

    async def inspect(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("inspect")
        return {"phase": RunPhase.RESEARCH}

    async def orchestrate(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("orchestrate")
        return {"pending_route": "research"}

    async def research(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("research")
        return {
            "current_experiment_id": "experiment-1",
            "current_hypothesis_id": "hypothesis-1",
            "pending_route": "proposal_policy",
        }

    async def proposal_policy(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("proposal_policy")
        return {"pending_route": "proposal_validation"}

    async def proposal_validation(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("proposal_validation")
        return {
            "latest_validation_report_id": "proposal-report",
            "pending_route": "create_worktree",
        }

    async def create_worktree(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("create_worktree")
        return {"active_worktree_id": "worktree-1", "pending_route": "implement"}

    async def implement(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("implement")
        return {"phase": RunPhase.IMPLEMENT, "pending_route": "diff_policy"}

    async def diff_policy(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("diff_policy")
        return {"pending_route": "implementation_validation"}

    async def implementation_validation(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("implementation_validation")
        return {
            "latest_validation_report_id": "implementation-report",
            "pending_route": "register_source",
        }

    async def register_source(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("register_source")
        return {"pending_route": "preflight"}

    async def preflight(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("preflight")
        return {"pending_route": "execute"}

    async def execute(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("execute")
        return {
            "phase": RunPhase.EXECUTE,
            "latest_execution_result_id": "execution-1",
            "pending_route": "evaluate",
        }

    async def evaluate(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("evaluate")
        return {
            "phase": RunPhase.EVALUATE,
            "latest_evaluation_result_id": "evaluation-1",
            "pending_route": "result_validation",
        }

    async def result_validation(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("result_validation")
        return {"latest_validation_report_id": "result-report", "pending_route": "interpret"}

    async def interpret(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("interpret")
        return {"pending_route": "persist"}

    async def persist(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("persist")
        return {"phase": RunPhase.PERSIST, "pending_route": "update_frontier"}

    async def update_frontier(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("update_frontier")
        return {"terminal_reason": "converged", "pending_route": "finalize"}

    async def repair(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("repair")
        return {"repair_attempts": state["repair_attempts"] + 1, "pending_route": "implement"}

    async def persist_failure(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("persist_failure")
        return {"pending_route": "orchestrate"}

    async def finalize(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("finalize")
        return {"phase": RunPhase.FINALIZE, "pending_route": "export"}

    async def export(self, state: ProductionState) -> dict[str, object]:
        self.calls.append("export")
        return {"phase": RunPhase.COMPLETE, "pending_route": "complete"}


@pytest.mark.asyncio
async def test_graph_runs_controller_owned_pipeline_to_completion() -> None:
    controller = RecordingController()
    graph = build_production_graph(controller)
    initial: ProductionState = {
        "run_id": "run-1",
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

    result = await graph.ainvoke(initial)

    assert result["phase"] == RunPhase.COMPLETE
    assert result["current_experiment_id"] == "experiment-1"
    assert controller.calls == [
        "bootstrap",
        "inspect",
        "orchestrate",
        "research",
        "proposal_policy",
        "proposal_validation",
        "create_worktree",
        "implement",
        "diff_policy",
        "implementation_validation",
        "register_source",
        "preflight",
        "execute",
        "evaluate",
        "result_validation",
        "interpret",
        "persist",
        "update_frontier",
        "finalize",
        "export",
    ]


def test_controller_implements_all_graph_use_cases() -> None:
    operations = {
        name
        for name, value in ControllerOperations.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
    assert operations == {
        "bootstrap",
        "inspect",
        "orchestrate",
        "research",
        "proposal_policy",
        "proposal_validation",
        "create_worktree",
        "implement",
        "diff_policy",
        "implementation_validation",
        "register_source",
        "preflight",
        "execute",
        "evaluate",
        "result_validation",
        "interpret",
        "persist",
        "update_frontier",
        "repair",
        "persist_failure",
        "finalize",
        "export",
    }
    assert operations <= {
        name
        for name, value in ProductionController.__dict__.items()
        if callable(value) and not name.startswith("_")
    }
