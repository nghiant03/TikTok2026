from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from tiktok2026.contracts import EvidenceItem, ResearchRequest


class EvidenceReader(Protocol):
    async def read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]: ...


@dataclass(frozen=True)
class ResearchCapabilities:
    repository: EvidenceReader
    data: EvidenceReader
    memory: EvidenceReader
    literature: EvidenceReader


async def build_context(
    request: ResearchRequest, capabilities: ResearchCapabilities, maximum: int = 32
) -> tuple[EvidenceItem, ...]:
    groups = await asyncio.gather(
        capabilities.repository.read(request),
        capabilities.data.read(request),
        capabilities.memory.read(request),
        capabilities.literature.read(request),
    )
    evidence = tuple(item for group in groups for item in group)
    identifiers = [item.evidence_id for item in evidence]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evidence IDs must be unique")
    if any(not item.authorized or item.contains_test_labels for item in evidence):
        raise ValueError("unauthorized or test-label evidence")
    return evidence[:maximum]
