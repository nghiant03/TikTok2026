from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tiktok2026.contracts.models import (
    ArtifactRecord,
    ContractModel,
    EvaluationRequest,
    EvaluationResult,
    ExecutionRequest,
    ExecutionResult,
    ExperimentSpec,
    SourceRegistration,
    WorktreeAssignment,
)


class AgentClient(Protocol):
    async def invoke(self, request: ContractModel) -> ContractModel: ...


class ArtifactRegistry(Protocol):
    def register(self, record: ArtifactRecord) -> None: ...


class WorktreeManager(Protocol):
    def create(
        self, run_id: str, spec: ExperimentSpec, parent_commit: str
    ) -> WorktreeAssignment: ...

    def register_source(
        self, assignment: WorktreeAssignment, allowed_scopes: tuple[str, ...]
    ) -> SourceRegistration: ...

    def remove(self, assignment: WorktreeAssignment) -> None: ...


class Executor(Protocol):
    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class RepositoryReader(Protocol):
    def read(self, relative_path: str, max_characters: int = 20_000) -> str: ...

    def search(self, query: str, max_results: int = 20) -> tuple[str, ...]: ...


class ScopedRepository(RepositoryReader, Protocol):
    def write(self, relative_path: str, content: str) -> None: ...

    def diff(self) -> str: ...

    def run_check(self, command: tuple[str, ...], timeout_seconds: int) -> str: ...


class DataSummaryReader(Protocol):
    def summarize(self, manifest_id: str) -> tuple[ContractModel, ...]: ...


class MemoryReader(Protocol):
    def retrieve(self, query: str, limit: int) -> tuple[ContractModel, ...]: ...


class LiteratureReader(Protocol):
    async def retrieve(self, query: str, limit: int) -> tuple[ContractModel, ...]: ...


class TraceSink(Protocol):
    def record(self, run_id: str, payload: ContractModel) -> Path: ...
