from __future__ import annotations

import json
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from research_agent.agent import parse_and_validate_response
from research_agent.capabilities import ResearchCapabilities
from research_agent.context import build_research_context
from research_agent.contracts import (
    ResearchAgentFailure,
    ResearchContext,
    ResearchFailureKind,
    ResearchRequest,
    ResearchResponse,
)
from research_agent.model import ModelOutput, ResearchModelClient
from research_agent.policy import ResearchPolicyError
from research_agent.prompt import build_research_prompt


class ResearchGraphState(TypedDict, total=False):
    request: ResearchRequest | dict[str, object]
    context: ResearchContext
    prompt: str
    raw_output: ModelOutput
    response: ResearchResponse
    failure: ResearchAgentFailure
    validation_error: str | None
    validation_failure_kind: ResearchFailureKind | None
    repair_attempts: int


def build_research_graph(
    model_client: ResearchModelClient,
    capabilities: ResearchCapabilities,
):
    def validate_request_node(state: ResearchGraphState) -> dict[str, object]:
        raw_request = state.get("request")
        try:
            request = ResearchRequest.model_validate(raw_request)
        except ValidationError as exc:
            request_id = (
                str(raw_request.get("request_id", "invalid-request"))
                if isinstance(raw_request, dict)
                else "invalid-request"
            )
            return {
                "failure": ResearchAgentFailure(
                    request_id=request_id or "invalid-request",
                    kind=ResearchFailureKind.SCHEMA,
                    message=str(exc),
                    repair_attempts=0,
                )
            }
        return {"request": request}

    async def build_context_node(state: ResearchGraphState) -> dict[str, object]:
        request = state["request"]
        if not isinstance(request, ResearchRequest):
            raise TypeError("validate_request must run before build_context")
        try:
            context = await build_research_context(request, capabilities)
        except Exception as exc:
            return {
                "failure": ResearchAgentFailure(
                    request_id=request.request_id,
                    kind=ResearchFailureKind.POLICY,
                    message=str(exc),
                    repair_attempts=0,
                )
            }
        return {
            "context": context,
            "prompt": build_research_prompt(context),
            "repair_attempts": 0,
            "validation_error": None,
            "validation_failure_kind": None,
        }

    async def call_model_node(state: ResearchGraphState) -> dict[str, object]:
        if "failure" in state:
            return {}
        try:
            raw_output = await model_client.generate(state["prompt"])
        except Exception as exc:
            return {
                "failure": ResearchAgentFailure(
                    request_id=_validated_request(state).request_id,
                    kind=ResearchFailureKind.MODEL,
                    message=str(exc),
                    repair_attempts=0,
                )
            }
        return {"raw_output": raw_output}

    def validate_response_node(state: ResearchGraphState) -> dict[str, object]:
        if "failure" in state:
            return {}
        try:
            response = parse_and_validate_response(state["raw_output"], state["context"])
        except (ValidationError, ResearchPolicyError, json.JSONDecodeError) as exc:
            kind = (
                ResearchFailureKind.POLICY
                if isinstance(exc, ResearchPolicyError)
                else ResearchFailureKind.SCHEMA
            )
            return {
                "validation_error": str(exc),
                "validation_failure_kind": kind,
            }
        return {
            "response": response,
            "validation_error": None,
            "validation_failure_kind": None,
        }

    async def repair_node(state: ResearchGraphState) -> dict[str, object]:
        try:
            raw_output = await model_client.generate(
                state["prompt"],
                previous_output=state["raw_output"],
                validation_error=state["validation_error"],
            )
        except Exception as exc:
            return {
                "failure": ResearchAgentFailure(
                    request_id=_validated_request(state).request_id,
                    kind=ResearchFailureKind.MODEL,
                    message=str(exc),
                    repair_attempts=1,
                )
            }
        return {"raw_output": raw_output, "repair_attempts": 1}

    def terminal_failure_node(state: ResearchGraphState) -> dict[str, object]:
        return {
            "failure": ResearchAgentFailure(
                request_id=_validated_request(state).request_id,
                kind=state.get("validation_failure_kind") or ResearchFailureKind.SCHEMA,
                message=state.get("validation_error") or "unknown response validation failure",
                repair_attempts=state.get("repair_attempts", 0),
            )
        }

    def route_after_request(state: ResearchGraphState) -> str:
        return "end" if "failure" in state else "build_context"

    def route_after_context(state: ResearchGraphState) -> str:
        return "end" if "failure" in state else "call_model"

    def route_after_model(state: ResearchGraphState) -> str:
        return "end" if "failure" in state else "validate_response"

    def route_after_validation(state: ResearchGraphState) -> str:
        if "response" in state:
            return "end"
        if "failure" in state:
            return "end"
        if state.get("repair_attempts", 0) < 1:
            return "repair"
        return "terminal_failure"

    graph = StateGraph(ResearchGraphState)
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("call_model", call_model_node)
    graph.add_node("validate_response", validate_response_node)
    graph.add_node("repair", repair_node)
    graph.add_node("terminal_failure", terminal_failure_node)
    graph.add_edge(START, "validate_request")
    graph.add_conditional_edges(
        "validate_request", route_after_request, {"build_context": "build_context", "end": END}
    )
    graph.add_conditional_edges(
        "build_context", route_after_context, {"call_model": "call_model", "end": END}
    )
    graph.add_conditional_edges(
        "call_model", route_after_model, {"validate_response": "validate_response", "end": END}
    )
    graph.add_conditional_edges(
        "validate_response",
        route_after_validation,
        {"repair": "repair", "terminal_failure": "terminal_failure", "end": END},
    )
    graph.add_edge("repair", "validate_response")
    graph.add_edge("terminal_failure", END)
    return graph.compile()


def _validated_request(state: ResearchGraphState) -> ResearchRequest:
    request = state["request"]
    if not isinstance(request, ResearchRequest):
        raise TypeError("research request has not been validated")
    return request


async def run_research_graph(
    request: ResearchRequest,
    model_client: ResearchModelClient,
    capabilities: ResearchCapabilities,
) -> ResearchGraphState:
    graph = build_research_graph(model_client, capabilities)
    result = await graph.ainvoke({"request": request})
    return ResearchGraphState(**result)
