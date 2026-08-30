from __future__ import annotations

import json
from typing import TypeVar

from loguru import logger
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
    errors: list[str] = []
    contract = {
        "request": payload,
        "response_json_schema": model_type.model_json_schema(),
    }
    for attempt in range(2):
        try:
            raw = await client.complete(
                f"{prompt} Return only a JSON object matching response_json_schema exactly.",
                json.dumps(contract, sort_keys=True)
                if attempt == 0
                else json.dumps(
                    {
                        **contract,
                        "instruction": (
                            "Correct the prior response and return only the JSON object."
                        ),
                        "validation_error": error_text,
                    },
                    sort_keys=True,
                ),
                request_id=request_id,
                role=role.value,
                attempt=attempt + 1,
            )
            result = model_type.model_validate(raw)
            logger.debug(
                "Structured response accepted request_id={} role={} attempt={} schema={}",
                request_id,
                role.value,
                attempt + 1,
                model_type.__name__,
            )
            return result
        except (ValidationError, ValueError) as error:
            error_text = str(error)[:2000]
            errors.append(f"attempt {attempt + 1}: {error_text}")
            logger.warning(
                "Structured response rejected request_id={} role={} attempt={} "
                "schema={} error_type={} error_chars={}",
                request_id,
                role.value,
                attempt + 1,
                model_type.__name__,
                type(error).__name__,
                len(str(error)),
            )
        except Exception as error:
            logger.warning(
                "Model invocation failed request_id={} role={} attempt={} error_type={}",
                request_id,
                role.value,
                attempt + 1,
                type(error).__name__,
            )
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
        message="; ".join(errors),
        repair_attempts=1,
    )
