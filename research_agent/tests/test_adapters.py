from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.adapters import (
    ArxivLiteratureEvidenceReader,
    CrossrefLiteratureEvidenceReader,
    EvidenceAccessError,
    FileSystemRepositoryEvidenceReader,
    JsonlExperimentHistoryReader,
    KuaiRandPureDataEvidenceReader,
    LocalPdfLiteratureEvidenceReader,
    SemanticScholarLiteratureEvidenceReader,
    WebPageLiteratureEvidenceReader,
)
from research_agent.contracts import (
    EvidenceStrength,
    ExperimentHistoryItem,
    ExperimentOutcome,
    ResearchLesson,
)

LOG_HEADER = (
    "user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,is_comment,"
    "is_forward,is_hate,long_view,play_time_ms,duration_ms,profile_stay_time,"
    "comment_stay_time,is_profile_enter,is_rand,tab"
)


def _make_minimal_dataset(root: Path) -> Path:
    dataset = root / "KuaiRand-Pure"
    data = dataset / "data"
    data.mkdir(parents=True)
    train_rows = [
        "1,10,20220409,900,1,1,0,0,0,0,0,1,8000,7000,0,0,0,0,1",
        "1,11,20220410,901,2,0,0,0,0,0,0,0,1000,9000,0,0,0,0,1",
        "2,10,20220411,902,3,1,0,0,0,0,0,1,8000,7000,0,0,0,0,1",
    ]
    (data / "log_standard_4_08_to_4_21_pure.csv").write_text(
        LOG_HEADER + "\n" + "\n".join(train_rows) + "\n", encoding="utf-8"
    )
    # The reader must inspect only this header, never the hidden-row payload.
    (data / "log_standard_4_22_to_5_08_pure.csv").write_text(
        LOG_HEADER + "\nTHIS_ROW_MUST_NOT_BE_PARSED\n", encoding="utf-8"
    )
    (data / "log_random_4_22_to_5_08_pure.csv").write_text(
        LOG_HEADER + "\n", encoding="utf-8"
    )
    (data / "user_features_pure.csv").write_text("user_id,feature\n", encoding="utf-8")
    (data / "video_features_basic_pure.csv").write_text(
        "video_id,author_id\n", encoding="utf-8"
    )
    (data / "video_features_statistic_pure.csv").write_text(
        "video_id,show_cnt\n", encoding="utf-8"
    )
    return dataset


@pytest.mark.asyncio
async def test_repository_reader_returns_bounded_authorized_evidence(
    tmp_path, proposal_request
) -> None:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "ARCHITECTURE.md").write_text(
        "Research Agent reads bounded repository evidence.", encoding="utf-8"
    )
    (root / ".env").write_text("API_KEY=must-not-appear", encoding="utf-8")
    reader = FileSystemRepositoryEvidenceReader(root, max_items=2)

    evidence = await reader.read_repository_evidence(proposal_request)

    assert len(evidence) == 1
    assert "bounded repository evidence" in evidence[0].summary
    assert "must-not-appear" not in evidence[0].summary


@pytest.mark.asyncio
async def test_data_reader_uses_training_labels_only(tmp_path, proposal_request) -> None:
    dataset = _make_minimal_dataset(tmp_path / "allowed")
    denied = tmp_path / "denied"
    denied.mkdir()
    reader = KuaiRandPureDataEvidenceReader(dataset, denied_roots=(denied,))

    evidence = await reader.read_data_evidence(proposal_request)

    combined = " ".join(item.summary for item in evidence)
    assert "rows=3" in combined
    assert "long_view_positive_rate=0.666667" in combined
    assert "long_view is the primary target" in combined
    assert "auxiliary training targets" in combined
    assert "No validation/public-holdout rows were read" in combined
    assert "organizer hidden test is not locally available" in combined
    assert "THIS_ROW_MUST_NOT_BE_PARSED" not in combined
    assert all(item.contains_test_labels is False for item in evidence)


@pytest.mark.asyncio
async def test_data_reader_rejects_denied_dataset_root(tmp_path, proposal_request) -> None:
    denied = tmp_path / "denied"
    dataset = _make_minimal_dataset(denied)
    reader = KuaiRandPureDataEvidenceReader(dataset, denied_roots=(denied,))

    with pytest.raises(EvidenceAccessError, match="denied root"):
        await reader.read_data_evidence(proposal_request)


@pytest.mark.asyncio
async def test_jsonl_history_reader_loads_typed_records(tmp_path, proposal_request) -> None:
    path = tmp_path / "experiments.jsonl"
    expected = ExperimentHistoryItem(
        experiment_id="experiment-1",
        normalized_signature="signature-1",
        summary="Pairwise loss did not improve validation.",
        outcome=ExperimentOutcome.NO_CLEAR_CHANGE,
        evidence_refs=("evaluation-1",),
    )
    path.write_text(expected.model_dump_json() + "\n", encoding="utf-8")

    items = await JsonlExperimentHistoryReader(path).read_experiment_history(proposal_request)

    assert items == (expected,)


@pytest.mark.asyncio
async def test_memory_query_returns_related_lineage_and_lessons(tmp_path, proposal_request) -> None:
    path = tmp_path / "experiments.jsonl"
    ancestor = ExperimentHistoryItem(
        experiment_id="experiment-ancestor",
        normalized_signature="signature-ancestor",
        summary="Initial ranking baseline.",
        outcome=ExperimentOutcome.NO_CLEAR_CHANGE,
        hypothesis_id="hypothesis-ancestor",
        tags=("ranking",),
    )
    parent = ExperimentHistoryItem(
        experiment_id="experiment-parent",
        normalized_signature="signature-parent",
        summary="Ranking features improved validation.",
        outcome=ExperimentOutcome.IMPROVED,
        hypothesis_id="hypothesis-parent",
        parent_experiment_id=ancestor.experiment_id,
        tags=("ranking", "features"),
    )
    path.write_text(
        ancestor.model_dump_json() + "\n" + parent.model_dump_json() + "\n",
        encoding="utf-8",
    )
    lesson = ResearchLesson(
        lesson_id="lesson-ranking-1",
        claim="Ranking features should be tested before model expansion.",
        evidence_strength=EvidenceStrength.STRONG,
        scope="experiment/features",
        tags=("ranking",),
        supporting_experiment_ids=(parent.experiment_id,),
        evidence_refs=("evaluation-parent",),
    )
    (tmp_path / "lessons.jsonl").write_text(lesson.model_dump_json() + "\n", encoding="utf-8")
    request = proposal_request.model_copy(
        update={
            "objective": "Propose a ranking feature experiment.",
            "parent_experiment_id": parent.experiment_id,
        }
    )

    result = await JsonlExperimentHistoryReader(path).query_research_memory(request)

    assert result.experiment_lineage == (ancestor, parent)
    assert parent in result.related_experiments
    assert result.retrieved_lessons == (lesson,)


@pytest.mark.asyncio
async def test_local_pdf_reader_uses_bounded_extractor(tmp_path, proposal_request) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"fake-pdf")
    calls = []

    def extractor(path, max_pages):
        calls.append((path, max_pages))
        return "A bounded abstract about autonomous machine learning research."

    reader = LocalPdfLiteratureEvidenceReader(tmp_path, extractor=extractor)
    evidence = await reader.read_literature_evidence(proposal_request)

    assert len(evidence) == 1
    assert "bounded abstract" in evidence[0].summary
    assert calls == [(paper, 3)]


@pytest.mark.asyncio
async def test_crossref_reader_returns_traceable_public_literature(proposal_request) -> None:
    def transport(request, timeout):
        assert "query.bibliographic" in request.full_url
        assert timeout == 15.0
        return {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/example",
                        "title": ["Pairwise Ranking for Recommenders"],
                        "URL": "https://example.test/paper",
                        "abstract": "<jats:p>A ranking study.</jats:p>",
                    }
                ]
            }
        }

    reader = CrossrefLiteratureEvidenceReader(transport=transport)
    evidence = await reader.read_literature_evidence(proposal_request)

    assert len(evidence) == 1
    assert evidence[0].source_ref == "https://doi.org/10.1000/example"
    assert "A ranking study" in evidence[0].summary


@pytest.mark.asyncio
async def test_semantic_scholar_reader_returns_traceable_literature(proposal_request) -> None:
    def transport(request, timeout):
        assert "api.semanticscholar.org" in request.full_url
        assert timeout == 15.0
        return {
            "data": [
                {
                    "paperId": "paper-1",
                    "title": "Sequential Recommendation",
                    "abstract": "A controlled ranking study.",
                    "url": "https://www.semanticscholar.org/paper/paper-1",
                }
            ]
        }

    evidence = await SemanticScholarLiteratureEvidenceReader(
        transport=transport
    ).read_literature_evidence(proposal_request)

    assert len(evidence) == 1
    assert evidence[0].source_ref.endswith("paper-1")
    assert "controlled ranking study" in evidence[0].summary


@pytest.mark.asyncio
async def test_arxiv_reader_returns_traceable_literature(proposal_request) -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>https://arxiv.org/abs/2601.00001</id><title>Ranking Agents</title>
      <summary>An autonomous recommendation study.</summary></entry>
    </feed>"""

    def transport(request, timeout):
        assert "export.arxiv.org" in request.full_url
        assert timeout == 15.0
        return feed

    evidence = await ArxivLiteratureEvidenceReader(
        transport=transport
    ).read_literature_evidence(proposal_request)

    assert len(evidence) == 1
    assert evidence[0].source_ref == "https://arxiv.org/abs/2601.00001"
    assert "autonomous recommendation study" in evidence[0].summary


@pytest.mark.asyncio
async def test_web_page_reader_returns_bounded_traceable_text(proposal_request) -> None:
    def transport(request, timeout):
        assert request.full_url == "https://example.test/research"
        assert timeout == 15.0
        return "<html><style>ignore</style><body>Public ranking documentation.</body></html>"

    evidence = await WebPageLiteratureEvidenceReader(
        ("https://example.test/research",), transport=transport
    ).read_literature_evidence(proposal_request)

    assert len(evidence) == 1
    assert evidence[0].source_ref == "https://example.test/research"
    assert "Public ranking documentation" in evidence[0].summary
    assert "ignore" not in evidence[0].summary
