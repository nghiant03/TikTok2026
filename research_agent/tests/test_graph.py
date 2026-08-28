from __future__ import annotations

import pytest

from research_agent.contracts import (
    EvidenceRequest,
    EvidenceRequestCategory,
    ExperimentOutcome,
    InterpretationNextStep,
    ResearchAgentFailure,
    ResearchDecisionKind,
    ResearchInterpretation,
    ResearchResponse,
)
from research_agent.graph import build_research_graph
from research_agent.model import ScriptedResearchModelClient
from research_agent.shared_contracts import FailureKind
from research_agent.testing import fake_capabilities
from tests.factories import make_capabilities, make_proposal_response


@pytest.mark.asyncio
async def test_graph_returns_structured_response(proposal_request) -> None:
    expected = make_proposal_response(proposal_request)
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))
    graph = build_research_graph(model, make_capabilities())

    result = await graph.ainvoke({"request": proposal_request})

    assert isinstance(result["response"], ResearchResponse)
    assert result["response"] == expected
    assert "failure" not in result


@pytest.mark.asyncio
async def test_graph_repairs_once_then_returns_failure(proposal_request) -> None:
    model = ScriptedResearchModelClient.from_outputs("bad-json", "bad-json-again")
    graph = build_research_graph(model, make_capabilities())

    result = await graph.ainvoke({"request": proposal_request})

    assert isinstance(result["failure"], ResearchAgentFailure)
    assert result["failure"].repair_attempts == 1
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_graph_can_run_two_research_cycles(proposal_request) -> None:
    first = make_proposal_response(
        proposal_request,
        response_id="response-1",
        experiment_id="experiment-1",
    )
    second = make_proposal_response(
        proposal_request,
        response_id="response-2",
        experiment_id="experiment-2",
    )
    model = ScriptedResearchModelClient.from_outputs(
        first.model_dump(mode="json"),
        second.model_dump(mode="json"),
    )
    graph = build_research_graph(model, make_capabilities())

    result_1 = await graph.ainvoke({"request": proposal_request})
    result_2 = await graph.ainvoke({"request": proposal_request})

    assert result_1["response"].experiment_proposal.spec.experiment_id == "experiment-1"
    assert result_2["response"].experiment_proposal.spec.experiment_id == "experiment-2"


@pytest.mark.asyncio
async def test_graph_interprets_evaluation_result(interpretation_request) -> None:
    expected = ResearchResponse(
        response_id="response-interpret-1",
        request_id=interpretation_request.request_id,
        kind=ResearchDecisionKind.RESULT_INTERPRETATION,
        result_interpretation=ResearchInterpretation(
            experiment_id="experiment-1",
            outcome=ExperimentOutcome.IMPROVED,
            objective_findings=("GAUC=0.667400; nDCG@5=0.535700.",),
            interpretation="The provisional evaluation is consistent with improvement.",
            evidence_refs=("evaluation:evaluation-1",),
            next_step=InterpretationNextStep.REPLICATE,
            next_step_rationale="Replication is needed before increasing fidelity.",
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))
    graph = build_research_graph(model, fake_capabilities())

    result = await graph.ainvoke({"request": interpretation_request})

    assert result["response"] == expected


@pytest.mark.asyncio
async def test_graph_validates_request_before_context() -> None:
    model = ScriptedResearchModelClient.from_outputs()
    graph = build_research_graph(model, fake_capabilities())

    result = await graph.ainvoke(
        {"request": {"request_id": "invalid-request", "task_type": "unknown"}}
    )

    assert result["failure"].kind.value == "schema"
    assert len(model.calls) == 0


@pytest.mark.asyncio
async def test_graph_returns_evidence_request(proposal_request) -> None:
    expected = ResearchResponse(
        response_id="response-evidence-graph-1",
        request_id=proposal_request.request_id,
        kind=ResearchDecisionKind.EVIDENCE_REQUEST,
        evidence_request=EvidenceRequest(
            request_id=proposal_request.request_id,
            reason="Candidate-generation behavior is missing.",
            categories=(EvidenceRequestCategory.REPOSITORY,),
            requested_items=("candidate-generation repository summary",),
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))
    graph = build_research_graph(model, fake_capabilities())

    result = await graph.ainvoke({"request": proposal_request})

    assert result["response"] == expected


@pytest.mark.asyncio
async def test_graph_interprets_failed_execution(failed_execution_request) -> None:
    expected = ResearchResponse(
        response_id="response-failed-execution-1",
        request_id=failed_execution_request.request_id,
        kind=ResearchDecisionKind.RESULT_INTERPRETATION,
        result_interpretation=ResearchInterpretation(
            experiment_id="experiment-1",
            outcome=ExperimentOutcome.INVALID_EXECUTION,
            objective_findings=("Execution exited with code 1.",),
            execution_failure_kind=FailureKind.SYNTAX_IMPORT,
            interpretation="The run failed before producing scientific evidence.",
            evidence_refs=("execution:execution-failed-1",),
            next_step=InterpretationNextStep.REQUEST_EVIDENCE,
            next_step_rationale="Implementation diagnostics are required before retrying.",
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(expected.model_dump(mode="json"))
    graph = build_research_graph(model, fake_capabilities())

    result = await graph.ainvoke({"request": failed_execution_request})

    assert result["response"] == expected


@pytest.mark.asyncio
async def test_graph_rejects_mismatched_execution_failure_kind(
    failed_execution_request,
) -> None:
    invalid = ResearchResponse(
        response_id="response-wrong-failure-1",
        request_id=failed_execution_request.request_id,
        kind=ResearchDecisionKind.RESULT_INTERPRETATION,
        result_interpretation=ResearchInterpretation(
            experiment_id="experiment-1",
            outcome=ExperimentOutcome.INVALID_EXECUTION,
            objective_findings=("Execution exited with code 1.",),
            execution_failure_kind=FailureKind.CUDA_OOM,
            interpretation="This incorrectly classifies the recorded failure.",
            evidence_refs=("execution:execution-failed-1",),
            next_step=InterpretationNextStep.REQUEST_EVIDENCE,
            next_step_rationale="Diagnostics are required.",
        ),
    )
    model = ScriptedResearchModelClient.from_outputs(
        invalid.model_dump(mode="json"),
        invalid.model_dump(mode="json"),
    )
    graph = build_research_graph(model, fake_capabilities())

    result = await graph.ainvoke({"request": failed_execution_request})

    assert result["failure"].kind.value == "policy"
    assert "does not match execution" in result["failure"].message
