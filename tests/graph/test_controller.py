from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from tiktok2026.contracts import Fidelity, RunPhase
from tiktok2026.controller import (
    ControllerServices,
    MissingTransitionError,
    ProductionController,
)
from tiktok2026.graph.state import ProductionState
from tiktok2026.persistence.checkpointer import SqliteCheckpointer


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


# ---------------------------------------------------------------------------
# Test 1: Controller missing transition fails closed
# ---------------------------------------------------------------------------


async def test_missing_transition_raises_typed_error() -> None:
    """A missing transition never returns {}; it raises MissingTransitionError."""
    store = _FakeTransitionStore()
    services = ControllerServices(transitions={}, store=store)
    controller = ProductionController(services)

    with pytest.raises(MissingTransitionError) as exc:
        await controller.orchestrate(minimal_state())

    assert "orchestrate" in str(exc.value)
    # No state change should have been persisted
    assert store.persisted == []


# ---------------------------------------------------------------------------
# Test 2: Two-step scripted lifecycle persists each transition
# ---------------------------------------------------------------------------


async def _research(state: ProductionState) -> dict[str, object]:
    return {
        "current_experiment_id": "exp-1",
        "current_hypothesis_id": "hyp-1",
        "pending_route": "proposal_policy",
    }


async def _persist(state: ProductionState) -> dict[str, object]:
    return {
        "phase": RunPhase.PERSIST,
        "pending_route": "update_frontier",
    }


async def test_two_step_lifecycle_persists_each_transition() -> None:
    """A scripted orchestrate->research->persist path persists each step."""
    store = _FakeTransitionStore()
    transitions: dict[str, Any] = {
        "orchestrate": _research,  # orchestrate sets up experiment
        "research": _research,  # research creates spec
        "persist": _persist,  # persist finalizes
    }
    services = ControllerServices(transitions=transitions, store=store)
    controller = ProductionController(services)

    state = minimal_state()

    # Step 1: orchestrate
    result1 = await controller.orchestrate(state)
    assert result1["pending_route"] == "proposal_policy"
    assert len(store.persisted) == 1
    run_id, op, ver, updates = store.persisted[0]
    assert run_id == "test-run"
    assert op == "orchestrate"
    assert ver == 1
    assert updates["pending_route"] == "proposal_policy"

    # Step 2: research (with updated state from step 1)
    state2: ProductionState = {
        **state,
        **result1,  # type: ignore[typeddict-item]
        "state_version": 1,
    }
    result2 = await controller.research(state2)
    assert result2["pending_route"] == "proposal_policy"
    assert len(store.persisted) == 2
    _, op2, ver2, _ = store.persisted[1]
    assert op2 == "research"
    assert ver2 == 2

    # Step 3: persist
    state3: ProductionState = {
        **state2,
        **result2,  # type: ignore[typeddict-item]
        "state_version": 2,
    }
    result3 = await controller.persist(state3)
    assert result3["pending_route"] == "update_frontier"
    assert result3["phase"] == RunPhase.PERSIST
    assert len(store.persisted) == 3
    _, op3, ver3, _ = store.persisted[2]
    assert op3 == "persist"
    assert ver3 == 3

    # Verify all three persisted
    assert [p[1] for p in store.persisted] == ["orchestrate", "research", "persist"]


# ---------------------------------------------------------------------------
# Test 3: SqliteCheckpointer round-trips a checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_db(tmp_path: Path) -> Path:
    db = tmp_path / "graph.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE graph_checkpoints (
            run_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, checkpoint_id)
        );
        """
    )
    conn.close()
    return db


async def test_sqlite_checkpointer_round_trip(graph_db: Path) -> None:
    """SqliteCheckpointer stores and retrieves a checkpoint by run/thread id."""
    checkpointer = SqliteCheckpointer(graph_db)

    # Write a checkpoint
    config: RunnableConfig = {"configurable": {"thread_id": "thread-1", "run_id": "run-1"}}
    checkpoint: dict[str, Any] = {
        "v": 1,
        "id": "checkpoint-1",
        "ts": "2026-01-01T00:00:00",
        "channel_values": {
            "run_id": "run-1",
            "phase": "bootstrap",
            "current_experiment_id": "exp-1",
            "current_hypothesis_id": None,
            "active_worktree_id": None,
            "latest_validation_report_id": None,
            "latest_execution_result_id": None,
            "latest_evaluation_result_id": None,
            "orchestration_decision_id": None,
            "repair_attempts": 0,
            "fidelity": "smoke",
            "pending_route": "research",
            "terminal_reason": None,
            "state_version": 1,
        },
        "channel_versions": {},
        "versions_seen": {},
    }
    metadata: dict[str, Any] = {
        "source": "loop",
        "step": 1,
        "parents": {},
        "run_id": "run-1",
    }

    result_config = await checkpointer.aput(config, checkpoint, metadata, {})  # type: ignore[arg-type]
    assert result_config is not None

    # Round-trip: read it back
    retrieved = await checkpointer.aget_tuple(config)
    assert retrieved is not None
    assert retrieved.checkpoint["id"] == "checkpoint-1"
    assert retrieved.checkpoint["channel_values"]["run_id"] == "run-1"
    assert retrieved.checkpoint["channel_values"]["pending_route"] == "research"
    assert retrieved.checkpoint["channel_values"]["state_version"] == 1
    assert retrieved.metadata is not None
    assert retrieved.metadata["step"] == 1
    assert retrieved.config["configurable"]["run_id"] == "thread-1"

    # A different thread_id should not find it
    other_config: RunnableConfig = {
        "configurable": {"thread_id": "thread-other", "run_id": "run-1"}
    }
    assert await checkpointer.aget_tuple(other_config) is None


# ---------------------------------------------------------------------------
# Test 4: Bootstrap synthetic composition builds without network/Docker/GPU
# ---------------------------------------------------------------------------


def test_synthetic_bootstrap_builds_without_external_services(tmp_path: Path) -> None:
    """Synthetic bootstrap produces a valid controller, repository, and graph."""
    import shutil

    from tiktok2026.bootstrap import build_synthetic_controller

    # Copy migrations from the real repo into the tmp repo root
    repo_root = Path(__file__).parents[2]
    test_repo_root = tmp_path / "repo"
    test_repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "migrations", test_repo_root / "migrations")

    controller, repo, graph = build_synthetic_controller(
        repository_root=test_repo_root,
        runtime_root=tmp_path / "runtime",
    )

    # Controller is a ProductionController
    assert isinstance(controller, ProductionController)
    # Repository is accessible
    assert repo is not None
    # Graph compiles
    assert graph is not None

    # Check repository is initialized
    with sqlite3.connect(tmp_path / "runtime" / "application.sqlite3") as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "experiments" in tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTransitionStore:
    def __init__(self) -> None:
        self.persisted: list[tuple[str, str, int, dict[str, object]]] = []

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))
