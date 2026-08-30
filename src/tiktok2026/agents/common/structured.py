from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import TypeVar, cast

from loguru import logger
from pydantic import ValidationError

from tiktok2026.agents.common.client import EmptyChoicesError, OpenAICompatibleClient
from tiktok2026.contracts import AgentFailure, AgentRole, ContractModel

ModelT = TypeVar("ModelT", bound=ContractModel)

# Tool handler: (tool_name, arguments_dict) → result_string
ToolHandler = Callable[[str, dict[str, object]], str]
# Terminal guard: return a bounded diagnostic to reject a validated submission,
# or ``None`` to accept it.
TerminalGuard = Callable[[ModelT], str | None]


def _validate_model(model_type: type[ModelT], raw: object) -> ModelT:
    # The repository diff, rather than a model-reported edit list, is the
    # authority for implementation changes.  In particular, an implementation
    # submission may contain metadata only when the agent has already changed
    # the bound worktree through tools.
    return model_type.model_validate(raw)


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
            result = _validate_model(model_type, raw)
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


async def invoke_agentic(
    client: OpenAICompatibleClient,
    role: AgentRole,
    request_id: str,
    model_type: type[ModelT],
    prompt: str,
    payload: dict[str, object],
    tools: Sequence[dict[str, object]],
    tool_handler: ToolHandler,
    *,
    max_turns: int = 20,
    terminal_tool: str | None = None,
    terminal_guard: TerminalGuard[ModelT] | None = None,
) -> ModelT | AgentFailure:
    """Multi-turn tool-use loop.  The model drives via tool calls.

    Termination: if ``terminal_tool`` is set, the loop ends when the model calls
    that tool — its arguments are validated against ``model_type``.  A
    validation error, or a rejection from ``terminal_guard``, is returned to
    the model as the tool result so it can correct and resubmit.  Otherwise
    the first turn without tool calls is parsed as the structured result.
    """
    system = (
        f"{prompt}\n\n"
        "You are in a multi-turn tool-use conversation. Use the provided tools to "
        "inspect and act on the assigned worktree."
    )
    if terminal_tool is not None:
        system += (
            f" When your work is complete, call the {terminal_tool} tool with the "
            "final result fields matching response_json_schema."
        )
    else:
        system += (
            " When your work is complete, respond with the final JSON object "
            "matching response_json_schema and no tool calls."
        )
    user_payload: dict[str, object] = {"request": payload}
    # In terminal-tool mode the submit_result tool already carries the complete
    # response schema.  Keeping a second copy in the user message wastes context
    # and can let the two schema representations drift.
    if terminal_tool is None:
        user_payload["response_json_schema"] = model_type.model_json_schema()
    user = json.dumps(user_payload, sort_keys=True)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    request_attempt = 0
    for turn in range(1, max_turns + 1):
        assistant: dict[str, object] = {}
        for response_attempt in range(2):
            request_attempt += 1
            try:
                assistant = await client.complete_with_tools(
                    messages,
                    list(tools),
                    request_id=request_id,
                    role=role.value,
                    attempt=request_attempt,
                )
                break
            except EmptyChoicesError as error:
                if response_attempt == 0:
                    logger.warning(
                        "Empty model response; retrying request_id={} role={} turn={} "
                        "request_attempt={} error={}",
                        request_id,
                        role.value,
                        turn,
                        request_attempt,
                        str(error),
                    )
                    continue
                logger.warning(
                    "Agentic turn failed request_id={} role={} turn={} error_type={}",
                    request_id,
                    role.value,
                    turn,
                    type(error).__name__,
                )
                return AgentFailure(
                    request_id=request_id,
                    role=role,
                    kind="model",
                    message=str(error),
                    repair_attempts=min(turn - 1, 1),
                )
            except Exception as error:
                logger.warning(
                    "Agentic turn failed request_id={} role={} turn={} error_type={}",
                    request_id,
                    role.value,
                    turn,
                    type(error).__name__,
                )
                return AgentFailure(
                    request_id=request_id,
                    role=role,
                    kind="model",
                    message=str(error),
                    repair_attempts=min(turn - 1, 1),
                )
        messages.append(assistant)
        tool_calls = assistant.get("tool_calls")
        if not tool_calls:
            # Final turn: parse content as the structured result.
            content = assistant.get("content")
            if not isinstance(content, str) or not content:
                return AgentFailure(
                    request_id=request_id,
                    role=role,
                    kind="schema",
                    message="final turn produced no content",
                    repair_attempts=min(turn - 1, 1),
                )
            try:
                raw = _extract_agentic_json(content)
                if raw is None:
                    raise ValueError("no JSON object in final response")
                result = _validate_model(model_type, raw)
                logger.debug(
                    "Agentic result accepted request_id={} role={} turns={} schema={}",
                    request_id,
                    role.value,
                    turn,
                    model_type.__name__,
                )
                return result
            except (ValidationError, ValueError) as error:
                logger.warning(
                    "Agentic result rejected request_id={} role={} turn={} "
                    "schema={} error_type={} error_chars={}",
                    request_id,
                    role.value,
                    turn,
                    model_type.__name__,
                    type(error).__name__,
                    len(str(error)),
                )
                return AgentFailure(
                    request_id=request_id,
                    role=role,
                    kind="schema",
                    message=f"turn {turn}: {error}",
                    repair_attempts=min(turn - 1, 1),
                )
        # Execute each tool call and append results.
        if not isinstance(tool_calls, list):
            return AgentFailure(
                request_id=request_id,
                role=role,
                kind="schema",
                message="tool_calls is not a list",
                repair_attempts=min(turn - 1, 1),
            )
        for tc_value in cast(list[object], tool_calls):
            if not isinstance(tc_value, dict):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": "invalid",
                        "content": "error: tool_call is not an object",
                    }
                )
                continue
            tc = cast(dict[str, object], tc_value)
            tc_id = str(tc.get("id", f"call-{turn}"))
            function_value = tc.get("function")
            if not isinstance(function_value, dict):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": "error: function is not an object",
                    }
                )
                continue
            func = cast(dict[str, object], function_value)
            tool_name = str(func.get("name", ""))
            arguments_raw = func.get("arguments", "{}")
            arguments: dict[str, object]
            if isinstance(arguments_raw, str):
                try:
                    parsed_args = json.loads(arguments_raw)
                    arguments = (
                        cast(dict[str, object], parsed_args)
                        if isinstance(parsed_args, dict)
                        else {}
                    )
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(arguments_raw, dict):
                arguments = cast(dict[str, object], arguments_raw)
            else:
                arguments = {}
            logger.debug(
                "Tool call request_id={} role={} turn={} tool={} tc_id={}",
                request_id,
                role.value,
                turn,
                tool_name,
                tc_id,
            )
            if terminal_tool is not None and tool_name == terminal_tool:
                try:
                    result = _validate_model(model_type, arguments)
                    logger.debug(
                        "Agentic result accepted request_id={} role={} turns={} "
                        "schema={} via terminal tool",
                        request_id,
                        role.value,
                        turn,
                        model_type.__name__,
                    )
                except (ValidationError, ValueError) as error:
                    # Feed the validation error back so the model can correct it.
                    logger.warning(
                        "Terminal result rejected request_id={} role={} turn={} "
                        "schema={} error_type={} error_chars={}",
                        request_id,
                        role.value,
                        turn,
                        model_type.__name__,
                        type(error).__name__,
                        len(str(error)),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": (
                                "error: result rejected, correct and resubmit: "
                                f"{str(error)[:2000]}"
                            ),
                        }
                    )
                    continue
                if terminal_guard is not None:
                    try:
                        rejection = terminal_guard(result)
                    except Exception as error:
                        rejection = f"{type(error).__name__}: {error}"
                        logger.warning(
                            "Terminal guard failed request_id={} role={} turn={} "
                            "error_type={} error_chars={}",
                            request_id,
                            role.value,
                            turn,
                            type(error).__name__,
                            len(str(error)),
                        )
                    if rejection is not None:
                        diagnostic = str(rejection)[:2000]
                        logger.warning(
                            "Terminal result rejected by guard request_id={} role={} "
                            "turn={} diagnostic_chars={}",
                            request_id,
                            role.value,
                            turn,
                            len(diagnostic),
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": (
                                    "error: result rejected, correct and resubmit: "
                                    f"{diagnostic}"
                                ),
                            }
                        )
                        continue
                return result
            try:
                result_text = tool_handler(tool_name, arguments)
            except Exception as error:
                result_text = f"error: {type(error).__name__}: {error}"
                logger.warning(
                    "Tool call failed request_id={} role={} turn={} tool={} "
                    "error_type={} error_chars={}",
                    request_id,
                    role.value,
                    turn,
                    tool_name,
                    type(error).__name__,
                    len(str(error)),
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_text,
                }
            )
    return AgentFailure(
        request_id=request_id,
        role=role,
        kind="model",
        message=f"agentic loop exceeded {max_turns} turns",
        repair_attempts=1,
    )


_FENCE_RE2 = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _extract_agentic_json(content: str) -> dict[str, object] | None:
    text = content.strip()
    fence = _FENCE_RE2.match(text)
    if fence is not None:
        text = fence.group(1).strip()
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(candidate, dict):
        return None
    return cast(dict[str, object], candidate)
