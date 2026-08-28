from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ModelOutput = str | dict[str, object]


class ResearchModelClient(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        previous_output: ModelOutput | None = None,
        validation_error: str | None = None,
    ) -> ModelOutput: ...


class ResearchModelConfigurationError(RuntimeError):
    """The model client cannot start because required local configuration is absent."""


class ResearchModelTransportError(RuntimeError):
    """The provider could not be reached or rejected the request."""


class ResearchModelResponseError(RuntimeError):
    """The provider returned a response that does not match the API contract."""


@dataclass(frozen=True)
class DeepSeekModelConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    api_key_env: str = "RESEARCH_MODEL_API_KEY"
    timeout_seconds: float | None = None
    max_tokens: int | None = None
    thinking: Literal["enabled", "disabled"] = "enabled"
    reasoning_effort: Literal["low", "high", "max"] = "high"

    def __post_init__(self) -> None:
        if not self.base_url.startswith("https://"):
            raise ValueError("DeepSeek base_url must use HTTPS")
        if not self.model.strip():
            raise ValueError("DeepSeek model must not be empty")
        if not self.api_key_env.strip():
            raise ValueError("DeepSeek API-key environment variable must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive or None")


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


JsonTransport = Callable[[Request, float | None], dict[str, Any]]


def _default_json_transport(request: Request, timeout: float | None) -> dict[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ResearchModelResponseError("provider response must be a JSON object")
    return decoded


@dataclass
class DeepSeekResearchModelClient:
    """OpenAI-compatible DeepSeek Chat Completions client.

    The secret is resolved from the environment for each call and is never stored in
    prompts, call records, exceptions, or the dataclass representation.
    """

    config: DeepSeekModelConfig = field(default_factory=DeepSeekModelConfig)
    transport: JsonTransport = field(default=_default_json_transport, repr=False)
    usage: list[ModelUsage] = field(default_factory=list, init=False)

    async def generate(
        self,
        prompt: str,
        *,
        previous_output: ModelOutput | None = None,
        validation_error: str | None = None,
    ) -> ModelOutput:
        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            previous_output,
            validation_error,
        )

    def _generate_sync(
        self,
        prompt: str,
        previous_output: ModelOutput | None,
        validation_error: str | None,
    ) -> str:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ResearchModelConfigurationError(
                f"required environment variable is not set: {self.config.api_key_env}"
            )

        user_content = prompt
        if previous_output is not None or validation_error is not None:
            rendered_output = (
                previous_output
                if isinstance(previous_output, str)
                else json.dumps(previous_output, ensure_ascii=False, sort_keys=True)
            )
            user_content += (
                "\nRepair the previous response exactly once. Return only the corrected JSON."
                f"\nPrevious response:\n{rendered_output or '<empty>'}"
                f"\nValidation error:\n{validation_error or '<unspecified>'}"
            )

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Follow the supplied Research Agent contract. Return exactly one "
                        "valid JSON object and no markdown."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "thinking": {"type": self.config.thinking},
            "reasoning_effort": self.config.reasoning_effort,
            "response_format": {"type": "json_object"},
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens

        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        request = Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            payload = self.transport(request, self.config.timeout_seconds)
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise ResearchModelTransportError(
                f"DeepSeek HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ResearchModelTransportError(f"DeepSeek request failed: {exc}") from exc

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ResearchModelResponseError(
                "DeepSeek response is missing choices[0].message.content"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ResearchModelResponseError("DeepSeek returned empty model content")

        raw_usage = payload.get("usage")
        if isinstance(raw_usage, dict):
            self.usage.append(
                ModelUsage(
                    prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
                    completion_tokens=int(raw_usage.get("completion_tokens", 0)),
                    total_tokens=int(raw_usage.get("total_tokens", 0)),
                )
            )
        return content


@dataclass
class ScriptedResearchModelClient:
    """Deterministic fake model for contract and graph tests."""

    outputs: deque[ModelOutput]
    calls: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_outputs(cls, *outputs: ModelOutput) -> ScriptedResearchModelClient:
        return cls(outputs=deque(outputs))

    async def generate(
        self,
        prompt: str,
        *,
        previous_output: ModelOutput | None = None,
        validation_error: str | None = None,
    ) -> ModelOutput:
        self.calls.append(
            {
                "prompt": prompt,
                "previous_output": previous_output,
                "validation_error": validation_error,
            }
        )
        if not self.outputs:
            raise RuntimeError("scripted model has no remaining output")
        return self.outputs.popleft()
