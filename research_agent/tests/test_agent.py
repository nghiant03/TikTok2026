from __future__ import annotations

import pytest

from research_agent.agent import run_research_with_repair
from research_agent.contracts import (
    BaselineStatus,
    EvaluationProtocolStatus,
    EvidenceRequest,
    EvidenceRequestCategory,
    ExperimentHistoryItem,
    ExperimentOutcome,
    ResearchAgentFailure,
    ResearchDecisionKind,
    ResearchFailureKind,
    ResearchRequest,
    ResearchResponse,
)
from research_agent.model import ScriptedResearchModelClient
from research_agent.policy import experiment_signature
from tests.factories import make_context, make_proposal_response


@pytest.mark.asyncio
async def test_agent_returns_valid_proposal(proposal_request) -> None:
    context = make_context(proposal_request)
    expected = make_proposal_response(proposal_request)
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchResponse)
    assert result == expected
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_agent_repairs_one_schema_error(proposal_request) -> None:
    context = make_context(proposal_request)
    expected = make_proposal_response(proposal_request)
    model = ScriptedResearchModelClient.from_outputs(
        "not-json",
        expected.model_dump_json(),
    )

    result = await run_research_with_repair(context, model)

    assert result == expected
    assert len(model.calls) == 2
    assert model.calls[1]["validation_error"]


@pytest.mark.asyncio
async def test_agent_fails_after_one_unsuccessful_repair(proposal_request) -> None:
    context = make_context(proposal_request)
    model = ScriptedResearchModelClient.from_outputs("not-json", "still-not-json")

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.SCHEMA
    assert result.repair_attempts == 1
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_agent_rejects_unknown_evidence_after_repair(proposal_request) -> None:
    context = make_context(proposal_request)
    invalid = make_proposal_response(
        proposal_request,
        evidence_refs=("invented-evidence",),
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "unknown evidence" in result.message


@pytest.mark.asyncio
async def test_agent_rejects_scope_escape(proposal_request) -> None:
    context = make_context(proposal_request)
    invalid = make_proposal_response(
        proposal_request,
        implementation_scope=("experiment/features/../../baseline/evaluate.py",),
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "unsafe implementation path" in result.message


@pytest.mark.asyncio
async def test_agent_can_request_missing_evidence(proposal_request) -> None:
    context = make_context(proposal_request)
    expected = ResearchResponse(
        response_id="response-evidence-1",
        request_id=proposal_request.request_id,
        kind=ResearchDecisionKind.EVIDENCE_REQUEST,
        evidence_request=EvidenceRequest(
            request_id=proposal_request.request_id,
            reason="The candidate-generation behavior is not represented in the context.",
            categories=(EvidenceRequestCategory.REPOSITORY,),
            requested_items=("candidate-generation repository summary",),
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))

    result = await run_research_with_repair(context, model)

    assert result == expected


@pytest.mark.asyncio
async def test_agent_rejects_exact_historical_duplicate(proposal_request) -> None:
    proposal = make_proposal_response(proposal_request)
    signature = experiment_signature(proposal.experiment_proposal.spec)
    context = make_context(proposal_request).model_copy(
        update={
            "experiment_history": (
                ExperimentHistoryItem(
                    experiment_id="historical-experiment",
                    normalized_signature=signature,
                    summary="The same recent-click feature was already evaluated.",
                    outcome=ExperimentOutcome.NO_CLEAR_CHANGE,
                    evidence_refs=("repo-1", "data-1"),
                ),
            )
        }
    )
    model = ScriptedResearchModelClient.from_outputs(
        proposal.model_dump(mode="json"),
        proposal.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "duplicates" in result.message


@pytest.mark.asyncio
async def test_agent_rejects_optimization_before_baseline(proposal_request) -> None:
    from research_agent.contracts import ProposalPurpose

    context = make_context(proposal_request)
    invalid = make_proposal_response(
        proposal_request,
        purpose=ProposalPurpose.OPTIMIZATION,
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "baseline" in result.message


@pytest.mark.asyncio
async def test_agent_accepts_optimization_after_provisional_baseline(
    proposal_request,
) -> None:
    from research_agent.contracts import ProposalPurpose

    payload = proposal_request.model_dump(mode="python")
    payload.update(
        baseline_status=BaselineStatus.PROVISIONAL,
        baseline_evidence_refs=("repo-1",),
    )
    request = ResearchRequest.model_validate(payload)
    context = make_context(request)
    expected = make_proposal_response(request, purpose=ProposalPurpose.OPTIMIZATION)
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))

    result = await run_research_with_repair(context, model)

    assert result == expected


@pytest.mark.asyncio
async def test_agent_rejects_empty_source_provenance(proposal_request) -> None:
    context = make_context(proposal_request)
    invalid = make_proposal_response(proposal_request, source_provenance=())
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "source_provenance" in result.message


@pytest.mark.asyncio
async def test_agent_rejects_unsubstantiated_numeric_thresholds(proposal_request) -> None:
    context = make_context(proposal_request)
    invalid = make_proposal_response(
        proposal_request,
        expected_signal="Mean validation score improves by +0.02.",
        success_criteria="Mean validation score is at least 0.50.",
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "numeric score" in result.message


@pytest.mark.asyncio
async def test_agent_accepts_official_baseline_number_from_protocol_evidence(
    proposal_request,
) -> None:
    context = make_context(proposal_request)
    expected = make_proposal_response(
        proposal_request,
        success_criteria="The reproduced validation primary score reaches 0.6016.",
    )
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))

    result = await run_research_with_repair(context, model)

    assert result == expected


@pytest.mark.asyncio
async def test_agent_rejects_proposal_without_protocol_evidence(
    proposal_request,
) -> None:
    context = make_context(proposal_request)
    invalid = make_proposal_response(
        proposal_request,
        evidence_refs=("repo-1", "data-1"),
        source_provenance=("repository-summary-v1", "data-summary-v1"),
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "evaluation protocol evidence" in result.message


@pytest.mark.asyncio
async def test_agent_repairs_missing_baseline_reproduction_control(
    proposal_request,
) -> None:
    context = make_context(proposal_request)
    expected = make_proposal_response(proposal_request)
    invalid = expected.model_dump(mode="json")
    invalid["experiment_proposal"]["baseline_reproduction_control"] = None
    model = ScriptedResearchModelClient.from_outputs(
        invalid,
        expected.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert result == expected
    assert len(model.calls) == 2
    assert "baseline_reproduction_control" in str(
        model.calls[1]["validation_error"]
    )


@pytest.mark.asyncio
async def test_agent_repairs_missing_official_fm_config(proposal_request) -> None:
    context = make_context(proposal_request)
    expected = make_proposal_response(proposal_request)
    invalid = expected.model_dump(mode="json")
    del invalid["experiment_proposal"]["baseline_reproduction_control"][
        "official_fm_config"
    ]
    model = ScriptedResearchModelClient.from_outputs(
        invalid,
        expected.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert result == expected
    assert len(model.calls) == 2
    assert "official_fm_config" in str(model.calls[1]["validation_error"])


@pytest.mark.asyncio
async def test_agent_rejects_gpu_or_empty_leakage_for_baseline(
    proposal_request,
) -> None:
    context = make_context(proposal_request)
    gpu_proposal = make_proposal_response(
        proposal_request,
        predicted_gpu_hours=0.5,
    )
    empty_leakage = make_proposal_response(
        proposal_request,
        leakage_risks=(),
    )
    model = ScriptedResearchModelClient.from_outputs(
        gpu_proposal.model_dump(mode="json"),
        empty_leakage.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "leakage_risks" in result.message


@pytest.mark.asyncio
async def test_agent_repairs_unconfirmed_protocol_proposal_into_evidence_request(
    proposal_request,
) -> None:
    payload = proposal_request.model_dump(mode="python")
    payload.update(
        evaluation_protocol_status=EvaluationProtocolStatus.UNCONFIRMED,
        evaluation_protocol_evidence_refs=(),
    )
    request = ResearchRequest.model_validate(payload)
    context = make_context(request)
    invalid = make_proposal_response(request)
    expected = ResearchResponse(
        response_id="response-protocol-evidence-1",
        request_id=request.request_id,
        kind=ResearchDecisionKind.EVIDENCE_REQUEST,
        evidence_request=EvidenceRequest(
            request_id=request.request_id,
            reason="The formal split and evaluator have not been confirmed.",
            categories=(
                EvidenceRequestCategory.DATA_SPLIT,
                EvidenceRequestCategory.EVALUATION_PROTOCOL,
            ),
            requested_items=(
                "Organizer-confirmed train/validation split definition.",
                "Organizer-compatible GAUC and nDCG@5 evaluator.",
            ),
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        expected.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert result == expected
    assert len(model.calls) == 2
    assert "evaluation protocol is unconfirmed" in str(
        model.calls[1]["validation_error"]
    )


@pytest.mark.asyncio
async def test_agent_rejects_incomplete_protocol_evidence_request(
    proposal_request,
) -> None:
    payload = proposal_request.model_dump(mode="python")
    payload.update(
        evaluation_protocol_status=EvaluationProtocolStatus.UNCONFIRMED,
        evaluation_protocol_evidence_refs=(),
    )
    request = ResearchRequest.model_validate(payload)
    context = make_context(request)
    incomplete = ResearchResponse(
        response_id="response-incomplete-protocol-evidence-1",
        request_id=request.request_id,
        kind=ResearchDecisionKind.EVIDENCE_REQUEST,
        evidence_request=EvidenceRequest(
            request_id=request.request_id,
            reason="Only the split is requested.",
            categories=(EvidenceRequestCategory.DATA_SPLIT,),
            requested_items=("Organizer-confirmed split definition.",),
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(
        incomplete.model_dump(mode="json"),
        incomplete.model_dump(mode="json"),
    )

    result = await run_research_with_repair(context, model)

    assert isinstance(result, ResearchAgentFailure)
    assert result.kind is ResearchFailureKind.POLICY
    assert "evaluation_protocol" in result.message
