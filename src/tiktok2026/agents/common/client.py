from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

import httpx
from loguru import logger

from tiktok2026.config import ModelSettings


class ChatTransport(Protocol):
    async def post_json(
        self, url: str, payload: dict[str, object], headers: dict[str, str], timeout: float
    ) -> dict[str, object]: ...


class HttpxChatTransport:
    async def post_json(
        self, url: str, payload: dict[str, object], headers: dict[str, str], timeout: float
    ) -> dict[str, object]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict):
            raise ValueError("chat response must be an object")
        value["_tiktok2026_proxy_metadata"] = {
            name: response.headers[name]
            for name in (
                "x-litellm-call-id",
                "x-litellm-response-cost-original",
                "x-litellm-response-duration-ms",
                "x-litellm-attempted-retries",
                "x-litellm-attempted-fallbacks",
                "llm_provider-x-codex-primary-used-percent",
                "llm_provider-x-codex-secondary-used-percent",
                "llm_provider-x-codex-primary-reset-after-seconds",
                "llm_provider-x-codex-secondary-reset-after-seconds",
            )
            if name in response.headers
        }
        return cast(dict[str, object], value)


class OpenAICompatibleClient:
    def __init__(self, settings: ModelSettings, transport: ChatTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport or HttpxChatTransport()

    async def complete(
        self,
        system: str,
        user: str,
        *,
        request_id: str = "unscoped",
        role: str = "unknown",
        attempt: int = 1,
    ) -> dict[str, object]:
        api_key = os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing model credential: {self.settings.api_key_env}")
        payload: dict[str, object] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": {"type": "json_object"},
        }
        logger.info(
            "Model request started request_id={} role={} attempt={} model={} "
            "timeout={} max_tokens={} system_chars={} user_chars={}",
            request_id,
            role,
            attempt,
            self.settings.model,
            self.settings.timeout_seconds,
            self.settings.max_tokens,
            len(system),
            len(user),
        )
        started = time.monotonic()
        response = await self.transport.post_json(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            self.settings.timeout_seconds,
        )
        elapsed_seconds = time.monotonic() - started
        choices_value = response.get("choices")
        if not isinstance(choices_value, list) or not choices_value:
            raise ValueError("chat response has no choices")
        choices = cast(list[object], choices_value)
        usage = response.get("usage")
        usage_details: Mapping[str, object] = (
            cast(dict[str, object], usage) if isinstance(usage, dict) else {}
        )
        completion_details = usage_details.get("completion_tokens_details")
        token_details: Mapping[str, object] = (
            cast(dict[str, object], completion_details)
            if isinstance(completion_details, dict)
            else {}
        )
        choice_summaries: list[str] = []
        selected_index: int | None = None
        parsed: dict[str, object] | None = None
        for index, choice_value in reversed(tuple(enumerate(choices))):
            if not isinstance(choice_value, dict):
                choice_summaries.append(f"{index}:invalid-choice")
                continue
            choice = cast(dict[str, object], choice_value)
            message_value = choice.get("message")
            if not isinstance(message_value, dict):
                choice_summaries.append(f"{index}:missing-message")
                continue
            content = cast(dict[str, object], message_value).get("content")
            content_length = len(content) if isinstance(content, str) else -1
            choice_summaries.append(
                f"{index}:{choice.get('finish_reason')}:{content_length}"
            )
            if not isinstance(content, str) or not content:
                continue
            try:
                candidate = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                selected_index = index
                parsed = cast(dict[str, object], candidate)
                break
        metadata_value = response.get("_tiktok2026_proxy_metadata")
        metadata: Mapping[str, object] = (
            cast(dict[str, object], metadata_value)
            if isinstance(metadata_value, dict)
            else {}
        )
        logger.info(
            "Model response request_id={} role={} attempt={} response_id={} "
            "model={} elapsed_seconds={:.3f} choices={} selected_choice={} "
            "prompt_tokens={} completion_tokens={} reasoning_tokens={} cost={} "
            "proxy_retries={} proxy_fallbacks={} quota_primary_percent={} "
            "quota_secondary_percent={} quota_primary_reset_seconds={} "
            "quota_secondary_reset_seconds={}",
            request_id,
            role,
            attempt,
            response.get("id"),
            response.get("model", self.settings.model),
            elapsed_seconds,
            len(choices),
            selected_index,
            usage_details.get("prompt_tokens"),
            usage_details.get("completion_tokens"),
            token_details.get("reasoning_tokens"),
            metadata.get("x-litellm-response-cost-original"),
            metadata.get("x-litellm-attempted-retries"),
            metadata.get("x-litellm-attempted-fallbacks"),
            metadata.get("llm_provider-x-codex-primary-used-percent"),
            metadata.get("llm_provider-x-codex-secondary-used-percent"),
            metadata.get("llm_provider-x-codex-primary-reset-after-seconds"),
            metadata.get("llm_provider-x-codex-secondary-reset-after-seconds"),
        )
        logger.debug(
            "Model choice summary request_id={} role={} attempt={} choices={}",
            request_id,
            role,
            attempt,
            tuple(reversed(choice_summaries)),
        )
        if parsed is None:
            raise ValueError(
                "chat response contains no JSON object choice "
                f"(choices={tuple(reversed(choice_summaries))}, "
                f"completion_tokens={usage_details.get('completion_tokens')}, "
                f"reasoning_tokens={token_details.get('reasoning_tokens')})"
            )
        return parsed

    async def complete_with_tools(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | Sequence[dict[str, object]],
        *,
        request_id: str = "unscoped",
        role: str = "unknown",
        attempt: int = 1,
    ) -> dict[str, object]:
        """One turn of a tool-use conversation.

        Returns the raw assistant message dict.  The caller inspects
        ``tool_calls`` to decide whether to continue the loop.
        """
        api_key = os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing model credential: {self.settings.api_key_env}")
        payload: dict[str, object] = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        system_chars = sum(
            len(str(m.get("content", ""))) for m in messages if m.get("role") == "system"
        )
        user_chars = sum(
            len(str(m.get("content", ""))) for m in messages if m.get("role") != "system"
        )
        logger.info(
            "Model request started request_id={} role={} attempt={} model={} "
            "timeout={} max_tokens={} system_chars={} user_chars={}",
            request_id,
            role,
            attempt,
            self.settings.model,
            self.settings.timeout_seconds,
            self.settings.max_tokens,
            system_chars,
            user_chars,
        )
        started = time.monotonic()
        response = await self.transport.post_json(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            self.settings.timeout_seconds,
        )
        elapsed_seconds = time.monotonic() - started
        choices_value = response.get("choices")
        if not isinstance(choices_value, list) or not choices_value:
            raise ValueError("chat response has no choices")
        choices = cast(list[object], choices_value)
        usage = response.get("usage")
        usage_details: Mapping[str, object] = (
            cast(dict[str, object], usage) if isinstance(usage, dict) else {}
        )
        completion_details = usage_details.get("completion_tokens_details")
        token_details: Mapping[str, object] = (
            cast(dict[str, object], completion_details)
            if isinstance(completion_details, dict)
            else {}
        )
        choice_summaries: list[str] = []
        assistant_message: dict[str, object] | None = None
        for index, choice_value in reversed(tuple(enumerate(choices))):
            if not isinstance(choice_value, dict):
                choice_summaries.append(f"{index}:invalid-choice")
                continue
            choice = cast(dict[str, object], choice_value)
            message_value = choice.get("message")
            if not isinstance(message_value, dict):
                choice_summaries.append(f"{index}:missing-message")
                continue
            message = cast(dict[str, object], message_value)
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            content_length = len(content) if isinstance(content, str) else -1
            tc_count = len(cast(list[object], tool_calls)) if isinstance(tool_calls, list) else 0
            choice_summaries.append(
                f"{index}:{choice.get('finish_reason')}:{content_length}:tc{tc_count}"
            )
            if tool_calls or (isinstance(content, str) and content):
                assistant_message = message
                break
        metadata_value = response.get("_tiktok2026_proxy_metadata")
        metadata: Mapping[str, object] = (
            cast(dict[str, object], metadata_value)
            if isinstance(metadata_value, dict)
            else {}
        )
        logger.info(
            "Model response request_id={} role={} attempt={} response_id={} "
            "model={} elapsed_seconds={:.3f} choices={} "
            "prompt_tokens={} completion_tokens={} reasoning_tokens={} cost={} "
            "proxy_retries={} proxy_fallbacks={} quota_primary_percent={} "
            "quota_secondary_percent={} quota_primary_reset_seconds={} "
            "quota_secondary_reset_seconds={}",
            request_id,
            role,
            attempt,
            response.get("id"),
            response.get("model", self.settings.model),
            elapsed_seconds,
            len(choices),
            usage_details.get("prompt_tokens"),
            usage_details.get("completion_tokens"),
            token_details.get("reasoning_tokens"),
            metadata.get("x-litellm-response-cost-original"),
            metadata.get("x-litellm-attempted-retries"),
            metadata.get("x-litellm-attempted-fallbacks"),
            metadata.get("llm_provider-x-codex-primary-used-percent"),
            metadata.get("llm_provider-x-codex-secondary-used-percent"),
            metadata.get("llm_provider-x-codex-primary-reset-after-seconds"),
            metadata.get("llm_provider-x-codex-secondary-reset-after-seconds"),
        )
        logger.debug(
            "Model choice summary request_id={} role={} attempt={} choices={}",
            request_id,
            role,
            attempt,
            tuple(reversed(choice_summaries)),
        )
        if assistant_message is None:
            raise ValueError(
                "chat response contains no usable assistant message "
                f"(choices={tuple(reversed(choice_summaries))}, "
                f"completion_tokens={usage_details.get('completion_tokens')}, "
                f"reasoning_tokens={token_details.get('reasoning_tokens')})"
            )
        return assistant_message
