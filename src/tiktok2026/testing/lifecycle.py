from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from tiktok2026.contracts import EvaluationResult, ExperimentSpec, Fidelity
from tiktok2026.testing.synthetic import evaluate_fixture, fixture_rows, score_rows


class SyntheticState(TypedDict):
    run_id: str
    iteration: int
    max_iterations: int
    experiment_ids: list[str]
    scores: list[float]
    latest_spec: ExperimentSpec | None
    latest_result: EvaluationResult | None
    terminal_reason: str | None


async def research(state: SyntheticState) -> dict[str, object]:
    iteration = state["iteration"] + 1
    experiment_id = f"synthetic-{iteration}"
    parent = state["experiment_ids"][-1] if state["experiment_ids"] else None
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id=f"hypothesis-{iteration}",
        hypothesis="Increasing signal scale preserves correct within-user ordering.",
        mechanism="Apply a deterministic scale to the synthetic ranking feature.",
        motivation="Exercise proposal, execution, evaluation, and persistence boundaries cheaply.",
        parent_experiment_id=parent,
        expected_signal="NDCG@10 and Recall@50 remain valid and deterministic.",
        implementation_scope=("synthetic_trainer",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Both metrics are finite and at least the previous values.",
        failure_criteria="Execution or schema validation fails.",
        source_provenance=("synthetic-fixture-v1",),
    )
    return {"iteration": iteration, "latest_spec": spec}


async def execute(state: SyntheticState) -> dict[str, object]:
    spec = state["latest_spec"]
    if spec is None:
        raise RuntimeError("research must create an experiment spec")
    rows = fixture_rows()
    result = evaluate_fixture(spec.experiment_id, rows, score_rows(rows, state["iteration"]))
    return {"latest_result": result}


async def persist(state: SyntheticState) -> dict[str, object]:
    spec = state["latest_spec"]
    result = state["latest_result"]
    if spec is None or result is None:
        raise RuntimeError("execution must produce a result")
    return {
        "experiment_ids": [*state["experiment_ids"], spec.experiment_id],
        "scores": [*state["scores"], result.validation_score],
    }


async def finish(state: SyntheticState) -> dict[str, object]:
    if state["iteration"] < 1:
        raise RuntimeError("cannot finish before an iteration completes")
    return {"terminal_reason": "synthetic_iteration_limit"}


def route_after_persist(state: SyntheticState) -> str:
    return "finish" if state["iteration"] >= state["max_iterations"] else "research"


def build_synthetic_graph():
    graph = StateGraph(SyntheticState)
    graph.add_node("research", research)
    graph.add_node("execute", execute)
    graph.add_node("persist", persist)
    graph.add_node("finish", finish)
    graph.add_edge(START, "research")
    graph.add_edge("research", "execute")
    graph.add_edge("execute", "persist")
    graph.add_conditional_edges(
        "persist", route_after_persist, {"research": "research", "finish": "finish"}
    )
    graph.add_edge("finish", END)
    return graph.compile()


async def run_synthetic_lifecycle(iterations: int = 2) -> SyntheticState:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    initial: SyntheticState = {
        "run_id": "synthetic-run",
        "iteration": 0,
        "max_iterations": iterations,
        "experiment_ids": [],
        "scores": [],
        "latest_spec": None,
        "latest_result": None,
        "terminal_reason": None,
    }
    result = await build_synthetic_graph().ainvoke(initial)
    return SyntheticState(**result)
