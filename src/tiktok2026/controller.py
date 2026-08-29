from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from tiktok2026.contracts import RunPhase
from tiktok2026.graph.state import ProductionState

Transition = Callable[[ProductionState], Awaitable[dict[str, object]]]


class TransitionStore(Protocol):
    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None: ...


@dataclass(frozen=True)
class ControllerServices:
    transitions: dict[str, Transition]
    store: TransitionStore


class ProductionController:
    def __init__(self, services: ControllerServices) -> None:
        self.services = services

    async def _run(self, operation: str, state: ProductionState) -> dict[str, object]:
        transition = self.services.transitions.get(operation)
        updates = await transition(state) if transition is not None else {}
        next_version = state["state_version"] + 1
        persisted = {**updates, "state_version": next_version}
        self.services.store.persist_transition(state["run_id"], operation, next_version, persisted)
        return persisted

    async def bootstrap(self, state: ProductionState) -> dict[str, object]:
        return await self._run("bootstrap", state)

    async def inspect(self, state: ProductionState) -> dict[str, object]:
        return await self._run("inspect", state)

    async def orchestrate(self, state: ProductionState) -> dict[str, object]:
        return await self._run("orchestrate", state)

    async def research(self, state: ProductionState) -> dict[str, object]:
        return await self._run("research", state)

    async def proposal_policy(self, state: ProductionState) -> dict[str, object]:
        return await self._run("proposal_policy", state)

    async def proposal_validation(self, state: ProductionState) -> dict[str, object]:
        return await self._run("proposal_validation", state)

    async def create_worktree(self, state: ProductionState) -> dict[str, object]:
        return await self._run("create_worktree", state)

    async def implement(self, state: ProductionState) -> dict[str, object]:
        return await self._run("implement", state)

    async def diff_policy(self, state: ProductionState) -> dict[str, object]:
        return await self._run("diff_policy", state)

    async def implementation_validation(self, state: ProductionState) -> dict[str, object]:
        return await self._run("implementation_validation", state)

    async def register_source(self, state: ProductionState) -> dict[str, object]:
        return await self._run("register_source", state)

    async def preflight(self, state: ProductionState) -> dict[str, object]:
        return await self._run("preflight", state)

    async def execute(self, state: ProductionState) -> dict[str, object]:
        return await self._run("execute", state)

    async def evaluate(self, state: ProductionState) -> dict[str, object]:
        return await self._run("evaluate", state)

    async def result_validation(self, state: ProductionState) -> dict[str, object]:
        return await self._run("result_validation", state)

    async def interpret(self, state: ProductionState) -> dict[str, object]:
        return await self._run("interpret", state)

    async def persist(self, state: ProductionState) -> dict[str, object]:
        return await self._run("persist", state)

    async def update_frontier(self, state: ProductionState) -> dict[str, object]:
        return await self._run("update_frontier", state)

    async def repair(self, state: ProductionState) -> dict[str, object]:
        return await self._run("repair", state)

    async def persist_failure(self, state: ProductionState) -> dict[str, object]:
        return await self._run("persist_failure", state)

    async def finalize(self, state: ProductionState) -> dict[str, object]:
        updates = await self._run("finalize", state)
        return {"phase": RunPhase.FINALIZE, **updates}

    async def export(self, state: ProductionState) -> dict[str, object]:
        updates = await self._run("export", state)
        return {"phase": RunPhase.COMPLETE, "pending_route": "complete", **updates}
