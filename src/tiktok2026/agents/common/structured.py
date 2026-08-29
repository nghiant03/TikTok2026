from __future__ import annotations

import json
from typing import TypeVar

from pydantic import ValidationError

from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.contracts import AgentFailure, AgentRole, ContractModel

ModelT = TypeVar("ModelT", bound=ContractModel)


async def invoke_structured(
    client: OpenAICompatibleClient,
    role: AgentRole,
    request_id: str,
    model_type: type[ModelT],
    prompt: str,
    payload: dict[str, object],
) -> ModelT | AgentFailure:
    error_text = ""
    for attempt in range(2):
        try:
            raw = await client.complete(
                prompt,
                json.dumps(payload, sort_keys=True)
                if attempt == 0
                else json.dumps(
                    {"payload": payload, "validation_error": error_text}, sort_keys=True
                ),
            )
            return model_type.model_validate(raw)
        except ValidationError as error:
            error_text = str(error)
        except Exception as error:
            return AgentFailure(
                request_id=request_id,
                role=role,
                kind="model",
                message=str(error),
                repair_attempts=attempt,
            )
    return AgentFailure(
        request_id=request_id,
        role=role,
        kind="schema",
        message=error_text,
        repair_attempts=1,
    )
