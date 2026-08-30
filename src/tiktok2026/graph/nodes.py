from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from tiktok2026.graph.state import ProductionState

Node = Callable[..., Awaitable[dict[str, object]]]


class ControllerOperations(Protocol):
    async def bootstrap(self, state: ProductionState) -> dict[str, object]: ...

    async def inspect(self, state: ProductionState) -> dict[str, object]: ...

    async def orchestrate(self, state: ProductionState) -> dict[str, object]: ...

    async def research(self, state: ProductionState) -> dict[str, object]: ...

    async def proposal_policy(self, state: ProductionState) -> dict[str, object]: ...

    async def proposal_validation(self, state: ProductionState) -> dict[str, object]: ...

    async def create_worktree(self, state: ProductionState) -> dict[str, object]: ...

    async def implement(self, state: ProductionState) -> dict[str, object]: ...

    async def diff_policy(self, state: ProductionState) -> dict[str, object]: ...

    async def implementation_validation(self, state: ProductionState) -> dict[str, object]: ...

    async def register_source(self, state: ProductionState) -> dict[str, object]: ...

    async def preflight(self, state: ProductionState) -> dict[str, object]: ...

    async def smoke(self, state: ProductionState) -> dict[str, object]: ...

    async def execute(self, state: ProductionState) -> dict[str, object]: ...

    async def evaluate(self, state: ProductionState) -> dict[str, object]: ...

    async def result_validation(self, state: ProductionState) -> dict[str, object]: ...

    async def interpret(self, state: ProductionState) -> dict[str, object]: ...

    async def persist(self, state: ProductionState) -> dict[str, object]: ...

    async def update_frontier(self, state: ProductionState) -> dict[str, object]: ...

    async def repair(self, state: ProductionState) -> dict[str, object]: ...

    async def persist_failure(self, state: ProductionState) -> dict[str, object]: ...

    async def finalize(self, state: ProductionState) -> dict[str, object]: ...

    async def export(self, state: ProductionState) -> dict[str, object]: ...


def controller_nodes(controller: ControllerOperations) -> dict[str, Node]:
    return {
        "bootstrap": controller.bootstrap,
        "inspect": controller.inspect,
        "orchestrate": controller.orchestrate,
        "research": controller.research,
        "proposal_policy": controller.proposal_policy,
        "proposal_validation": controller.proposal_validation,
        "create_worktree": controller.create_worktree,
        "implement": controller.implement,
        "diff_policy": controller.diff_policy,
        "implementation_validation": controller.implementation_validation,
        "register_source": controller.register_source,
        "preflight": controller.preflight,
        "smoke": controller.smoke,
        "execute": controller.execute,
        "evaluate": controller.evaluate,
        "result_validation": controller.result_validation,
        "interpret": controller.interpret,
        "persist": controller.persist,
        "update_frontier": controller.update_frontier,
        "repair": controller.repair,
        "persist_failure": controller.persist_failure,
        "finalize": controller.finalize,
        "export": controller.export,
    }
