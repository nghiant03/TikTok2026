from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    BenchmarkManifest,
    verify_protected_files,
)
from tiktok2026.contracts import RuntimePaths
from tiktok2026.controller import (
    ControllerServices,
    ProductionController,
    Transition,
)
from tiktok2026.graph.build import build_production_graph
from tiktok2026.graph.state import ProductionState
from tiktok2026.persistence.checkpointer import SqliteCheckpointer
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository


@dataclass(frozen=True)
class RuntimeServices:
    repository_root: Path
    paths: RuntimePaths
    repository: ApplicationRepository


def initialize_runtime(repository_root: Path, runtime_root: Path) -> RuntimeServices:
    repository_root = repository_root.resolve()
    paths = RuntimePaths.create(repository_root, runtime_root)
    MigrationRunner(paths.application_db, repository_root / "migrations" / "application").apply()
    MigrationRunner(paths.graph_db, repository_root / "migrations" / "graph").apply()
    return RuntimeServices(
        repository_root=repository_root,
        paths=paths,
        repository=ApplicationRepository(paths.application_db),
    )


def verify_manifests(repository_root: Path) -> BenchmarkManifest:
    manifest_path = (
        repository_root / "src" / "tiktok2026" / "benchmark" / "kuaireand_pure" / "manifest.json"
    )
    manifest = BenchmarkManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    verify_protected_files(repository_root, manifest.protected_reference_files)
    return manifest


# ---------------------------------------------------------------------------
# Synthetic composition — scripted agents, real persistence/graph/exports
# ---------------------------------------------------------------------------


class _SyntheticTransitionStore:
    """In-memory store that records every transition."""

    def __init__(self) -> None:
        self.persisted: list[tuple[str, str, int, dict[str, object]]] = []

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))


async def _scripted_research(state: ProductionState) -> dict[str, object]:
    """Deterministic research that produces a fixed experiment spec."""
    _ = state
    return {
        "current_experiment_id": "synth-exp-1",
        "current_hypothesis_id": "synth-hyp-1",
        "pending_route": "proposal_policy",
    }


async def _scripted_orchestrate(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "research"}


async def _scripted_bootstrap(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "inspect"}


async def _scripted_inspect(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "orchestrate"}


async def _scripted_implement(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "diff_policy"}


async def _scripted_validate_approved(state: ProductionState) -> dict[str, object]:
    _ = state
    return {
        "latest_validation_report_id": "synth-report",
        "pending_route": "create_worktree",
    }


async def _scripted_validate_impl_approved(state: ProductionState) -> dict[str, object]:
    _ = state
    return {
        "latest_validation_report_id": "synth-impl-report",
        "pending_route": "register_source",
    }


async def _scripted_validate_result_approved(state: ProductionState) -> dict[str, object]:
    _ = state
    return {
        "latest_validation_report_id": "synth-result-report",
        "pending_route": "interpret",
    }


async def _scripted_policy_approved(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "proposal_validation"}


async def _scripted_worktree(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"active_worktree_id": "synth-wt-1", "pending_route": "implement"}


async def _scripted_source(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "preflight"}


async def _scripted_preflight(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "execute"}


async def _scripted_execute(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"latest_execution_result_id": "synth-exec-1", "pending_route": "evaluate"}


async def _scripted_evaluate(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"latest_evaluation_result_id": "synth-eval-1", "pending_route": "result_validation"}


async def _scripted_interpret(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "persist"}


async def _scripted_persist(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"phase": "persist", "pending_route": "update_frontier"}


async def _scripted_frontier(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"terminal_reason": "converged", "pending_route": "finalize"}


async def _scripted_repair(state: ProductionState) -> dict[str, object]:
    return {"repair_attempts": 1, "pending_route": "implement"}


async def _scripted_persist_failure(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"pending_route": "orchestrate"}


async def _scripted_finalize(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"phase": "finalize", "pending_route": "export"}


async def _scripted_export(state: ProductionState) -> dict[str, object]:
    _ = state
    return {"phase": "complete", "pending_route": "complete"}


def _make_synthetic_transitions() -> Mapping[str, Transition]:
    """Return a full set of scripted transitions for the synthetic lifecycle."""
    static = {
        "bootstrap": _scripted_bootstrap,
        "inspect": _scripted_inspect,
        "orchestrate": _scripted_orchestrate,
        "research": _scripted_research,
        "proposal_policy": _scripted_policy_approved,
        "proposal_validation": _scripted_validate_approved,
        "create_worktree": _scripted_worktree,
        "implement": _scripted_implement,
        "diff_policy": _scripted_policy_approved,
        "implementation_validation": _scripted_validate_impl_approved,
        "register_source": _scripted_source,
        "preflight": _scripted_preflight,
        "execute": _scripted_execute,
        "evaluate": _scripted_evaluate,
        "result_validation": _scripted_validate_result_approved,
        "interpret": _scripted_interpret,
        "persist": _scripted_persist,
        "update_frontier": _scripted_frontier,
        "repair": _scripted_repair,
        "persist_failure": _scripted_persist_failure,
        "finalize": _scripted_finalize,
        "export": _scripted_export,
    }
    return static


def build_synthetic_controller(
    repository_root: Path,
    runtime_root: Path,
) -> tuple[ProductionController, ApplicationRepository, object]:
    """Build a synthetic composition that uses real persistence but scripted agents.

    No network, Docker, or GPU resources are required.
    Returns (controller, repository, compiled_graph).
    """
    runtime = initialize_runtime(repository_root, runtime_root)
    repo = runtime.repository

    # Build the controller with scripted transitions
    store = _SyntheticTransitionStore()
    raw_transitions = _make_synthetic_transitions()
    # The transitions dict must be typed as Transition for the controller
    services = ControllerServices(
        transitions=raw_transitions,
        store=store,
    )
    controller = ProductionController(services)

    # Build the graph with a checkpointer backed by the real graph DB
    checkpointer = SqliteCheckpointer(runtime.paths.graph_db)
    graph = build_production_graph(controller, checkpointer=checkpointer)

    return controller, repo, graph