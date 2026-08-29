from __future__ import annotations

import json
import os
from typing import Protocol, cast

import httpx

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
        return cast(dict[str, object], value)


class OpenAICompatibleClient:
    def __init__(self, settings: ModelSettings, transport: ChatTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport or HttpxChatTransport()

    async def complete(self, system: str, user: str) -> dict[str, object]:
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
        response = await self.transport.post_json(
            f"{self.settings.base_url.rstrip('/')}/chat/completions",
            payload,
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            self.settings.timeout_seconds,
        )
        choices_value = response.get("choices")
        if not isinstance(choices_value, list) or not choices_value:
            raise ValueError("chat response has no choices")
        choices = cast(list[object], choices_value)
        choice_value = choices[0]
        if not isinstance(choice_value, dict):
            raise ValueError("chat response has no choice object")
        choice = cast(dict[str, object], choice_value)
        message_value = choice.get("message")
        if not isinstance(message_value, dict):
            raise ValueError("chat response has no message")
        message = cast(dict[str, object], message_value)
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("chat response content must be text")
        parsed: object = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("structured response must be an object")
        return cast(dict[str, object], parsed)
