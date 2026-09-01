from __future__ import annotations

import json

from pydantic import ValidationError

from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.research.context import ResearchCapabilities, build_context
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ResearchDecision,
    ResearchRequest,
)


class ResearchAgent:
    def __init__(self, client: OpenAICompatibleClient, capabilities: ResearchCapabilities) -> None:
        self.client = client
        self.capabilities = capabilities

    async def invoke(self, request: ResearchRequest) -> ResearchDecision | AgentFailure:
        try:
            evidence = await build_context(request, self.capabilities)
        except Exception as error:
            return AgentFailure(
                request_id=request.request_id,
                role=AgentRole.RESEARCH,
                kind="policy",
                message=str(error),
                repair_attempts=0,
            )
        evidence_ids = {item.evidence_id for item in evidence}
        if request.source_context is not None:
            evidence_ids.add(request.source_context.evidence_id)
        if request.experiment_history is not None:
            evidence_ids.add(request.experiment_history.evidence_id)
        if request.controller_context is not None:
            if request.controller_context.experiment_registry is not None:
                evidence_ids.add(request.controller_context.experiment_registry.evidence_id)
            if request.controller_context.dataset_context is not None:
                evidence_ids.add(request.controller_context.dataset_context.evidence_id)
        user = json.dumps(
            {
                "request": request.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in evidence],
            },
            sort_keys=True,
        )
        validation_error = ""
        for attempt in range(2):
            try:
                raw = await self.client.complete(
                    "Return one ResearchDecision JSON object. Never request test labels.",
                    user
                    if attempt == 0
                    else f"{user}\nRepair this validation error: {validation_error}",
                )
                decision = ResearchDecision.model_validate(raw)
                if decision.request_id != request.request_id:
                    raise ValueError("research request ID mismatch")
                if not set(decision.evidence_refs).issubset(evidence_ids):
                    raise ValueError("research decision cites unknown evidence")
                if decision.experiment_spec is not None and not set(
                    decision.experiment_spec.implementation_scope
                ).issubset(set(request.allowed_paths)):
                    raise ValueError("experiment scope is not authorized")
                return decision
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                validation_error = str(error)
            except Exception as error:
                return AgentFailure(
                    request_id=request.request_id,
                    role=AgentRole.RESEARCH,
                    kind="model",
                    message=str(error),
                    repair_attempts=attempt,
                )
        return AgentFailure(
            request_id=request.request_id,
            role=AgentRole.RESEARCH,
            kind="schema",
            message=validation_error,
            repair_attempts=1,
        )
