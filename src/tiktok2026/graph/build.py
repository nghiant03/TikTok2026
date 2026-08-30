from collections.abc import Hashable
from typing import Any

from langgraph.graph import END, START, StateGraph

from tiktok2026.graph.nodes import ControllerOperations, controller_nodes
from tiktok2026.graph.routes import route_pending
from tiktok2026.graph.state import ProductionState

NODES = (
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
    "smoke",
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
)


def build_production_graph(
    controller: ControllerOperations,
    checkpointer: Any = None,
):
    graph = StateGraph(ProductionState)
    for name, node in controller_nodes(controller).items():
        graph.add_node(name, node)
    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", "inspect")
    graph.add_edge("inspect", "orchestrate")
    destinations: dict[Hashable, str] = {name: name for name in NODES}
    destinations["complete"] = END
    for name in NODES:
        if name not in {"bootstrap", "inspect", "export"}:
            graph.add_conditional_edges(name, route_pending, destinations)
    graph.add_conditional_edges("export", route_pending, destinations)
    return graph.compile(checkpointer=checkpointer)
