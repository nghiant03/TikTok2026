import json
import stat
from pathlib import Path

import pytest

from tiktok2026.contracts import AgentFailure, AgentRole, EvidenceItem, LessonRecord
from tiktok2026.literature.retrieval import LiteratureSource, LocalLiteratureReader
from tiktok2026.memory.retrieval import PersistenceMemoryReader
from tiktok2026.observability.traces import RestrictedTraceSink
from tiktok2026.persistence.repositories import ApplicationRepository


def test_memory_retrieval_is_bounded_and_evidence_backed(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "application.sqlite3")
    repository.initialize()
    matching = LessonRecord(
        lesson_id="lesson-1",
        statement="Proxy fidelity preserved ranking quality.",
        evidence_strength="strong",
        experiment_ids=("experiment-1",),
        tags=("proxy",),
    )
    ignored = matching.model_copy(
        update={"lesson_id": "lesson-2", "statement": "Unrelated observation."}
    )
    repository.put_json("lesson", matching.lesson_id, matching.model_dump_json())
    repository.put_json("lesson", ignored.lesson_id, ignored.model_dump_json())

    evidence = PersistenceMemoryReader(repository).retrieve("proxy", limit=1)

    assert evidence == (
        EvidenceItem(
            evidence_id="memory-lesson-1",
            kind="memory_lesson",
            summary=matching.statement,
            source_ref="lesson:lesson-1;experiments:experiment-1",
        ),
    )


@pytest.mark.asyncio
async def test_literature_reader_requires_configured_local_sources(tmp_path: Path) -> None:
    document = tmp_path / "paper.txt"
    document.write_text("Counterfactual evaluation reduces exposure bias.", encoding="utf-8")
    reader = LocalLiteratureReader(
        (LiteratureSource(source_id="paper-1", path=document, license="CC-BY-4.0"),)
    )

    evidence = await reader.retrieve("exposure", limit=1)

    assert len(evidence) == 1
    assert evidence[0].source_ref == f"{document.resolve().as_uri()}#license=CC-BY-4.0"


@pytest.mark.asyncio
async def test_literature_reader_rejects_unconfigured_paths(tmp_path: Path) -> None:
    reader = LocalLiteratureReader(())

    assert await reader.retrieve("anything", limit=2) == ()


def test_restricted_trace_sink_redacts_secrets_and_restricts_permissions(tmp_path: Path) -> None:
    sink = RestrictedTraceSink(tmp_path)
    payload = AgentFailure(
        request_id="request-1",
        role=AgentRole.RESEARCH,
        kind="model",
        message="Authorization: Bearer secret-token api_key=hidden",
        repair_attempts=0,
    )

    path = sink.record("run-1", payload)
    serialized = path.read_text(encoding="utf-8")

    assert "secret-token" not in serialized
    assert "hidden" not in serialized
    assert "[REDACTED]" in serialized
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(serialized)["request_id"] == "request-1"
