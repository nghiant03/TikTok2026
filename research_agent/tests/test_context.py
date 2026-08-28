from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.context import build_research_context
from research_agent.contracts import (
    OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
    BaselineStatus,
    EvaluationProtocolStatus,
    EvidenceItem,
    EvidenceKind,
    ResearchRequest,
    ResearchTaskType,
)
from research_agent.testing import fake_capabilities


@pytest.mark.asyncio
async def test_context_collects_traceable_bounded_evidence(proposal_request) -> None:
    capabilities = fake_capabilities(
        repository_evidence=(
            EvidenceItem(
                evidence_id="repo-1",
                kind=EvidenceKind.REPOSITORY,
                summary="The editable feature pipeline has no recent-click feature.",
                source_ref="repository-map-v1",
            ),
        ),
        data_evidence=(
            EvidenceItem(
                evidence_id="data-1",
                kind=EvidenceKind.DATA,
                summary="Training users have timestamped click histories.",
                source_ref="safe-data-summary-v1",
            ),
        ),
    )

    context = await build_research_context(
        proposal_request,
        capabilities,
        max_evidence_items=4,
    )

    assert context.evidence_ids == frozenset(
        {
            "benchmark:kuairand-pure:v1",
            OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
            "repo-1",
            "data-1",
        }
    )
    assert len(context.evidence) == 4
    assert "authority=user-confirmed resolution" in context.evidence[0].summary
    assert "baseline_status=missing" in context.evidence[0].summary
    assert "evaluation_protocol_status=confirmed" in context.evidence[0].summary
    assert "label=long_view" in context.evidence[1].summary
    assert "metrics=GAUC,nDCG@5" in context.evidence[1].summary
    assert "public_holdout=20220429..20220508" in context.evidence[1].summary
    assert "organizer hidden test is not locally available" in context.evidence[1].summary
    assert "features=user_id,video_id,author_id,tab,dur_bucket" in context.evidence[1].summary
    assert "data.py::load,baseline.py::run_fm" in context.evidence[1].summary


@pytest.mark.asyncio
async def test_context_rejects_test_label_evidence(proposal_request) -> None:
    capabilities = fake_capabilities(
        data_evidence=(
            EvidenceItem(
                evidence_id="forbidden-test-labels",
                kind=EvidenceKind.DATA,
                summary="Hidden test labels.",
                source_ref="hidden-test",
                contains_test_labels=True,
            ),
        )
    )

    with pytest.raises(ValidationError, match="forbidden evidence"):
        await build_research_context(proposal_request, capabilities)


@pytest.mark.asyncio
async def test_context_truncates_to_declared_bound(proposal_request) -> None:
    data_evidence = tuple(
        EvidenceItem(
            evidence_id=f"data-{index}",
            kind=EvidenceKind.DATA,
            summary=f"Safe summary {index}",
            source_ref="safe-summary",
        )
        for index in range(10)
    )

    context = await build_research_context(
        proposal_request,
        fake_capabilities(data_evidence=data_evidence),
        max_evidence_items=4,
    )

    assert len(context.evidence) == 4
    assert context.evidence[0].kind is EvidenceKind.BENCHMARK


@pytest.mark.asyncio
async def test_context_prioritizes_failed_execution_evidence(
    failed_execution_request,
) -> None:
    context = await build_research_context(
        failed_execution_request,
        fake_capabilities(),
        max_evidence_items=2,
    )

    assert context.evidence_ids == frozenset(
        {"benchmark:kuairand-pure:v1", "execution:execution-failed-1"}
    )


@pytest.mark.asyncio
async def test_context_requires_declared_baseline_evidence(
    proposal_request,
) -> None:
    payload = proposal_request.model_dump(mode="python")
    payload.update(
        baseline_status=BaselineStatus.PROVISIONAL,
        baseline_evidence_refs=("missing-baseline-evidence",),
    )
    request = ResearchRequest.model_validate(payload)

    with pytest.raises(ValidationError, match="baseline evidence is absent"):
        await build_research_context(request, fake_capabilities())


@pytest.mark.asyncio
async def test_context_requires_declared_evaluation_protocol_evidence(
    proposal_request,
) -> None:
    payload = proposal_request.model_dump(mode="python")
    payload.update(
        evaluation_protocol_status=EvaluationProtocolStatus.CONFIRMED,
        evaluation_protocol_evidence_refs=("missing-protocol-evidence",),
    )
    request = ResearchRequest.model_validate(payload)

    with pytest.raises(
        ValidationError,
        match="evaluation protocol evidence is absent",
    ):
        await build_research_context(request, fake_capabilities())


@pytest.mark.asyncio
async def test_context_marks_default_evaluation_protocol_unconfirmed(
    benchmark,
    resource_state,
) -> None:
    request = ResearchRequest(
        request_id="request-unconfirmed-protocol",
        task_type=ResearchTaskType.PROPOSE_EXPERIMENT,
        objective="Determine what evidence is required before proposing an experiment.",
        benchmark=benchmark,
        resource_state=resource_state,
        allowed_implementation_scope=("experiment/models",),
    )

    context = await build_research_context(request, fake_capabilities())

    assert "evaluation_protocol_status=unconfirmed" in context.evidence[0].summary
