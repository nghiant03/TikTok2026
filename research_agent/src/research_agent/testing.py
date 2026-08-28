from __future__ import annotations

from dataclasses import dataclass

from research_agent.capabilities import ResearchCapabilities
from research_agent.contracts import (
    EvidenceItem,
    ExperimentHistoryItem,
    ResearchLesson,
    ResearchMemoryQueryResult,
    ResearchRequest,
)


@dataclass(frozen=True)
class FakeRepositoryEvidenceReader:
    evidence: tuple[EvidenceItem, ...] = ()

    async def read_repository_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        del request
        return self.evidence


@dataclass(frozen=True)
class FakeDataEvidenceReader:
    evidence: tuple[EvidenceItem, ...] = ()

    async def read_data_evidence(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        del request
        return self.evidence


@dataclass(frozen=True)
class FakeExperimentHistoryReader:
    history: tuple[ExperimentHistoryItem, ...] = ()
    lineage: tuple[ExperimentHistoryItem, ...] = ()
    lessons: tuple[ResearchLesson, ...] = ()

    async def query_research_memory(
        self, request: ResearchRequest
    ) -> ResearchMemoryQueryResult:
        return ResearchMemoryQueryResult(
            query=request.objective,
            related_experiments=self.history,
            experiment_lineage=self.lineage,
            retrieved_lessons=self.lessons,
        )


@dataclass(frozen=True)
class FakeLiteratureEvidenceReader:
    evidence: tuple[EvidenceItem, ...] = ()

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        del request
        return self.evidence


def fake_capabilities(
    *,
    repository_evidence: tuple[EvidenceItem, ...] = (),
    data_evidence: tuple[EvidenceItem, ...] = (),
    history: tuple[ExperimentHistoryItem, ...] = (),
    lineage: tuple[ExperimentHistoryItem, ...] = (),
    lessons: tuple[ResearchLesson, ...] = (),
    literature_evidence: tuple[EvidenceItem, ...] = (),
) -> ResearchCapabilities:
    return ResearchCapabilities(
        repository=FakeRepositoryEvidenceReader(repository_evidence),
        data=FakeDataEvidenceReader(data_evidence),
        memory=FakeExperimentHistoryReader(history, lineage, lessons),
        literature=FakeLiteratureEvidenceReader(literature_evidence),
    )
