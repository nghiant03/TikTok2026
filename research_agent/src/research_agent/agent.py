from __future__ import annotations

import json

from pydantic import ValidationError

from research_agent.contracts import (
    ResearchAgentFailure,
    ResearchContext,
    ResearchFailureKind,
    ResearchResponse,
)
from research_agent.model import ModelOutput, ResearchModelClient
from research_agent.policy import ResearchPolicyError, validate_research_response
from research_agent.prompt import build_research_prompt


def parse_and_validate_response(
    raw_output: ModelOutput,
    context: ResearchContext,
) -> ResearchResponse:
    if isinstance(raw_output, str):
        response = ResearchResponse.model_validate_json(raw_output)
    else:
        response = ResearchResponse.model_validate(raw_output)
    return validate_research_response(response, context)


async def run_research_with_repair(
    context: ResearchContext,
    model_client: ResearchModelClient,
) -> ResearchResponse | ResearchAgentFailure:
    """Standalone execution path shared by tests and non-graph callers."""

    prompt = build_research_prompt(context)
    previous_output: ModelOutput | None = None
    validation_error: str | None = None

    for attempt in range(2):
        try:
            raw_output = await model_client.generate(
                prompt,
                previous_output=previous_output,
                validation_error=validation_error,
            )
        except Exception as exc:  # model boundary; converted to typed failure
            return ResearchAgentFailure(
                request_id=context.request.request_id,
                kind=ResearchFailureKind.MODEL,
                message=str(exc),
                repair_attempts=attempt,
            )

        try:
            return parse_and_validate_response(raw_output, context)
        except (ValidationError, ResearchPolicyError, json.JSONDecodeError) as exc:
            previous_output = raw_output
            validation_error = str(exc)
            if attempt == 1:
                failure_kind = (
                    ResearchFailureKind.POLICY
                    if isinstance(exc, ResearchPolicyError)
                    else ResearchFailureKind.SCHEMA
                )
                return ResearchAgentFailure(
                    request_id=context.request.request_id,
                    kind=failure_kind,
                    message=validation_error,
                    repair_attempts=1,
                )

    raise AssertionError("unreachable")

