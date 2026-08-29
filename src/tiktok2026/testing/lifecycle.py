from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from tiktok2026.adapters import RepositoryExportService
from tiktok2026.bootstrap import build_synthetic_controller
from tiktok2026.contracts import (
    EvaluationResult,
    Fidelity,
    FinalizationRecord,
    RunPhase,
    RuntimePaths,
)
from tiktok2026.persistence.checkpointer import SqliteCheckpointer
from tiktok2026.persistence.repositories import ApplicationRepository


@dataclass(frozen=True)
class SyntheticExports:
    jsonl: Path
    markdown: Path
    jsonl_bytes: bytes
    markdown_bytes: bytes


@dataclass(frozen=True)
class SyntheticLifecycleResult:
    run_id: str
    experiment_ids: tuple[str, ...]
    scores: tuple[float, ...]
    terminal_reason: str
    finalization: FinalizationRecord
    exports: SyntheticExports
    paths: RuntimePaths


def _initial_state(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
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


def _checkpoint_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("synthetic graph returned a non-object state")
    typed_value = cast(dict[object, object], value)
    return {str(key): item for key, item in typed_value.items()}


def _evaluation_records(
    repository: ApplicationRepository, run_id: str
) -> tuple[EvaluationResult, ...]:
    evaluations: list[EvaluationResult] = []
    for raw in repository.list_json("evaluation"):
        payload = json.loads(raw)
        result = EvaluationResult.model_validate(payload.get("result", payload))
        if result.run_id == run_id:
            evaluations.append(result)
    return tuple(evaluations)


async def run_synthetic_lifecycle(
    iterations: int = 2, runtime_root: Path | None = None
) -> SyntheticLifecycleResult:
    """Run the deterministic fixture through the production graph composition."""
    if iterations < 2:
        raise ValueError("synthetic lifecycle requires at least two iterations")
    repository_root = Path(__file__).resolve().parents[3]
    selected_runtime = (
        runtime_root or repository_root.parent / f"{repository_root.name}.synthetic-runtime"
    )
    run_id = f"synthetic-run-{uuid.uuid4().hex}"
    _controller, _store, graph = build_synthetic_controller(
        repository_root,
        selected_runtime,
        iterations=iterations,
    )
    config: RunnableConfig = {"configurable": {"thread_id": run_id}, "recursion_limit": 10}
    interrupted = False
    try:
        await graph.ainvoke(_initial_state(run_id), config)
    except GraphRecursionError:
        interrupted = True
    if not interrupted:
        raise RuntimeError("synthetic lifecycle did not reach its interruption boundary")
    checkpointer = SqliteCheckpointer(selected_runtime / "graph.sqlite3")
    checkpoint = await checkpointer.aget_tuple(config)
    if checkpoint is None or checkpoint.metadata is None:
        raise RuntimeError("synthetic interruption did not persist compatible checkpoint metadata")
    checkpoint_config = checkpoint.config.get("configurable", {})
    if (
        checkpoint_config.get("thread_id") != run_id
        or checkpoint_config.get("checkpoint_id") is None
    ):
        raise RuntimeError("synthetic interruption did not persist compatible checkpoint config")
    # Resume with no in-memory state through the same canonical checkpointer and thread.
    resume_config = {**config, "recursion_limit": max(100, iterations * 30)}
    state = _checkpoint_state(await graph.ainvoke(None, resume_config))
    if state.get("phase") != RunPhase.COMPLETE and state.get("phase") != RunPhase.COMPLETE.value:
        raise RuntimeError(f"synthetic graph did not complete: {state.get('pending_route')}")

    repository = ApplicationRepository(selected_runtime / "application.sqlite3")
    evaluations = _evaluation_records(repository, run_id)
    if len(evaluations) != iterations:
        raise RuntimeError(
            f"synthetic graph persisted {len(evaluations)} evaluations; expected {iterations}"
        )
    finalization = repository.get_finalization(f"finalization-{run_id}")
    if finalization is None:
        raise RuntimeError("synthetic graph did not persist finalization")
    export_service = RepositoryExportService(repository, selected_runtime)
    jsonl = selected_runtime / "exports" / run_id / "iterations.jsonl"
    markdown = selected_runtime / "exports" / run_id / "iterations.md"
    if not jsonl.is_file() or not markdown.is_file():
        raise RuntimeError("synthetic graph did not produce deterministic exports")
    graph_jsonl = jsonl.read_bytes()
    graph_markdown = markdown.read_bytes()
    if not graph_jsonl or not graph_markdown:
        raise RuntimeError("synthetic graph produced empty exports")
    deterministic_dir = selected_runtime / "tmp" / "determinism" / run_id
    rerun = await export_service.export_run(run_id, deterministic_dir)
    comparison_jsonl = rerun["jsonl"].read_bytes()
    comparison_markdown = rerun["markdown"].read_bytes()
    graph_records = tuple(json.loads(line) for line in graph_jsonl.splitlines())
    comparison_records = tuple(json.loads(line) for line in comparison_jsonl.splitlines())
    export_events = tuple(
        record
        for record in comparison_records
        if record.get("event_type") == "controller_transition"
        and record.get("payload", {}).get("operation") == "export"
    )
    comparable_records = tuple(
        record for record in comparison_records if record not in export_events
    )
    comparable_markdown = comparison_markdown
    if len(export_events) == 1:
        marker = f"## {export_events[0]['event_id']}\n\n".encode()
        start = comparable_markdown.find(marker)
        if start >= 0:
            next_heading = comparable_markdown.find(b"\n## ", start + len(marker))
            end = len(comparable_markdown) if next_heading < 0 else next_heading + 1
            comparable_markdown = comparable_markdown[:start] + comparable_markdown[end:]
    if (
        len(export_events) != 1
        or graph_records != comparable_records
        or comparable_markdown != graph_markdown
    ):
        raise RuntimeError("synthetic graph export differs from deterministic repository export")
    if jsonl.read_bytes() != graph_jsonl or markdown.read_bytes() != graph_markdown:
        raise RuntimeError("deterministic comparison overwrote graph exports")
    return SyntheticLifecycleResult(
        run_id=run_id,
        experiment_ids=tuple(result.experiment_id for result in evaluations),
        scores=tuple(result.validation_score for result in evaluations),
        terminal_reason=str(state.get("terminal_reason") or "completed"),
        finalization=finalization,
        exports=SyntheticExports(
            jsonl=jsonl,
            markdown=markdown,
            jsonl_bytes=graph_jsonl,
            markdown_bytes=graph_markdown,
        ),
        paths=RuntimePaths.create(repository_root, selected_runtime),
    )
