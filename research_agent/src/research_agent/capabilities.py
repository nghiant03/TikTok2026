from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from research_agent.contracts import EvidenceItem, ResearchMemoryQueryResult, ResearchRequest


class RepositoryEvidenceReader(Protocol):
    async def read_repository_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]: ...


class DataEvidenceReader(Protocol):
    async def read_data_evidence(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]: ...


class ResearchMemoryReader(Protocol):
    async def query_research_memory(
        self, request: ResearchRequest
    ) -> ResearchMemoryQueryResult: ...


class LiteratureEvidenceReader(Protocol):
    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]: ...


@dataclass(frozen=True)
class ResearchCapabilities:
    repository: RepositoryEvidenceReader
    data: DataEvidenceReader
    memory: ResearchMemoryReader
    literature: LiteratureEvidenceReader
