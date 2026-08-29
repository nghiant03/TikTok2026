from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata

ChannelVersions = dict[str, str | int | float]


class SqliteCheckpointer(BaseCheckpointSaver[int]):
    """LangGraph BaseCheckpointSaver backed by the graph_checkpoints table.

    Stores only compact recovery references — not full artifacts or logs.
    The table schema matches ``migrations/graph/001_initial.sql``.
    """

    def __init__(self, database: Path) -> None:
        super().__init__()
        self._database = database

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._database))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---- sync stubs (not used by async graph, but required by ABC) ----

    def get_tuple(self, config: RunnableConfig) -> Any:
        raise NotImplementedError("use aget_tuple")

    def list(self, config: RunnableConfig, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("use alist")

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> Any:  # type: ignore[override]
        raise NotImplementedError("use aput")

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:  # type: ignore[override]
        raise NotImplementedError("use aput_writes")

    # ---- async implementations ----

    async def aget_tuple(self, config: RunnableConfig) -> Any | None:
        thread_id = _resolve_thread_id(config)
        if thread_id is None:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT checkpoint_id, state_json, created_at FROM graph_checkpoints "
                "WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
            if row is None:
                return None
            checkpoint = json.loads(row["state_json"])
            parent_config: RunnableConfig | None = None
            return _CheckpointTuple(
                checkpoint=checkpoint,
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": row["checkpoint_id"],
                    }
                },
                metadata=None,
                parent_config=parent_config,
            )
        finally:
            conn.close()

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        tup = await self.aget_tuple(config)
        return tup.checkpoint if tup is not None else None

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Any]:
        if config is None:
            return
        thread_id = _resolve_thread_id(config)
        if thread_id is None:
            return
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT checkpoint_id, state_json, created_at FROM graph_checkpoints "
                "WHERE run_id = ? ORDER BY created_at DESC LIMIT ?",
                (thread_id, limit or 100),
            ).fetchall()
            for row in rows:
                checkpoint = json.loads(row["state_json"])
                yield _CheckpointTuple(
                    checkpoint=checkpoint,
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    metadata=None,
                    parent_config=None,
                )
        finally:
            conn.close()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = _resolve_thread_id(config)
        if thread_id is None:
            thread_id = config.get("configurable", {}).get("run_id", "unknown")
        checkpoint_id = checkpoint.get("id", "checkpoint-" + _now_str())
        now = _now_str()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO graph_checkpoints "
                "(run_id, checkpoint_id, state_json, created_at) VALUES (?, ?, ?, ?)",
                (thread_id, checkpoint_id, json.dumps(checkpoint), now),
            )
            conn.commit()
        finally:
            conn.close()
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # Intermediate writes are not stored in the compact checkpoint table.
        # The controller stores full state in the application DB.
        pass

    async def adelete_thread(self, thread_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM graph_checkpoints WHERE run_id = ?", (thread_id,))
            conn.commit()
        finally:
            conn.close()


class _CheckpointTuple:
    """Minimal compatible checkpoint tuple."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        config: RunnableConfig,
        metadata: Any,
        parent_config: RunnableConfig | None,
        pending_writes: list[Any] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.config = config
        self.metadata = metadata
        self.parent_config = parent_config
        self.pending_writes = pending_writes or []


def _resolve_thread_id(config: RunnableConfig) -> str | None:
    conf = config.get("configurable", {})
    return conf.get("thread_id") or conf.get("run_id")


def _now_str() -> str:
    return datetime.now(UTC).isoformat()