from __future__ import annotations

import json

import pytest

from research_agent.model import (
    DeepSeekModelConfig,
    DeepSeekResearchModelClient,
    ResearchModelConfigurationError,
    ResearchModelResponseError,
)


@pytest.mark.asyncio
async def test_deepseek_client_sends_json_without_storing_secret(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_MODEL_API_KEY", "test-secret-value")
    captured = {}

    def transport(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return {
            "choices": [{"message": {"content": '{"schema_version":"1"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }

    client = DeepSeekResearchModelClient(transport=transport)
    output = await client.generate("Return JSON for the research contract.")

    assert output == '{"schema_version":"1"}'
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer test-secret-value"
    assert captured["timeout"] is None
    assert captured["body"]["model"] == "deepseek-v4-pro"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in captured["body"]
    assert "test-secret-value" not in repr(client)
    assert client.usage[0].total_tokens == 14


@pytest.mark.asyncio
async def test_deepseek_client_includes_single_repair_feedback(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_MODEL_API_KEY", "fake")
    captured = {}

    def transport(request, timeout):
        del timeout
        captured.update(json.loads(request.data.decode("utf-8")))
        return {"choices": [{"message": {"content": "{}"}}]}

    client = DeepSeekResearchModelClient(transport=transport)
    await client.generate(
        "original prompt",
        previous_output="bad-json",
        validation_error="invalid schema",
    )

    repair_prompt = captured["messages"][1]["content"]
    assert "bad-json" in repair_prompt
    assert "invalid schema" in repair_prompt
    assert "exactly once" in repair_prompt


@pytest.mark.asyncio
async def test_deepseek_client_fails_without_configured_key(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_MODEL_API_KEY", raising=False)
    called = False

    def transport(request, timeout):
        nonlocal called
        called = True
        raise AssertionError((request, timeout))

    client = DeepSeekResearchModelClient(transport=transport)
    with pytest.raises(ResearchModelConfigurationError, match="RESEARCH_MODEL_API_KEY"):
        await client.generate("prompt")
    assert called is False


@pytest.mark.asyncio
async def test_deepseek_client_rejects_malformed_provider_response(monkeypatch) -> None:
    monkeypatch.setenv("CUSTOM_KEY", "fake")
    client = DeepSeekResearchModelClient(
        config=DeepSeekModelConfig(api_key_env="CUSTOM_KEY"),
        transport=lambda request, timeout: {"choices": []},
    )

    with pytest.raises(ResearchModelResponseError, match="choices"):
        await client.generate("prompt")
