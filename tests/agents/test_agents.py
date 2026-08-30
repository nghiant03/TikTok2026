import json
from collections.abc import Sequence

from pytest import MonkeyPatch, raises

from tiktok2026.agents.common.client import ChatTransport, OpenAICompatibleClient
from tiktok2026.agents.research.agent import ResearchAgent
from tiktok2026.agents.research.context import ResearchCapabilities
from tiktok2026.config import ModelSettings
from tiktok2026.contracts import (
    AgentFailure,
    EvidenceItem,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
)


class RecordingTransport(ChatTransport):
    def __init__(self, responses: Sequence[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, object], dict[str, str]]] = []

    async def post_json(
        self, url: str, payload: dict[str, object], headers: dict[str, str], timeout: float
    ) -> dict[str, object]:
        self.requests.append((url, payload, headers))
        return self.responses.pop(0)


class Reader:
    def __init__(self, evidence: EvidenceItem) -> None:
        self.evidence = evidence

    async def read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        return (self.evidence,)


class EmptyReader:
    async def read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        return ()


def request() -> ResearchRequest:
    return ResearchRequest(
        request_id="request-1",
        objective="Propose a safe experiment",
        resource_state=ResourceState(
            remaining_gpu_hours=1.0,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=100.0,
            used_tokens=0,
            remaining_tokens=1000,
            disk_bytes_available=1000,
            reserved_final_gpu_hours=0.25,
        ),
    )


async def test_openai_compatible_client_uses_configured_endpoint(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [{"choices": [{"message": {"content": json.dumps({"message": "ok"})}}]}]
    )
    client = OpenAICompatibleClient(
        ModelSettings(base_url="https://example.test/v1", model="gpt-4.1"), transport
    )
    output = await client.complete("system", "user")
    assert output == {"message": "ok"}
    assert transport.requests[0][0] == "https://example.test/v1/chat/completions"
    assert transport.requests[0][1]["model"] == "gpt-4.1"
    assert transport.requests[0][2]["Authorization"] == "Bearer secret"


async def test_openai_compatible_client_reports_empty_reasoning_response(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [
            {
                "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                "usage": {
                    "completion_tokens": 8192,
                    "completion_tokens_details": {"reasoning_tokens": 8192},
                },
            }
        ]
    )

    with raises(ValueError, match="reasoning_tokens=8192"):
        await OpenAICompatibleClient(ModelSettings(), transport).complete("system", "user")


async def test_openai_compatible_client_selects_last_json_choice(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "I will inspect."}},
                    {"finish_reason": "stop", "message": {"content": ""}},
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"message": "implemented"})},
                    },
                ]
            }
        ]
    )

    result = await OpenAICompatibleClient(ModelSettings(), transport).complete(
        "system", "user"
    )

    assert result == {"message": "implemented"}


async def test_research_repairs_invalid_response_once(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    valid = ResearchDecision(
        request_id="request-1",
        kind="evidence_request",
        message="Need baseline evidence",
        evidence_refs=("repo-1",),
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": "{}"}}]},
            {"choices": [{"message": {"content": json.dumps(valid)}}]},
        ]
    )
    evidence = EvidenceItem(
        evidence_id="repo-1",
        kind="repository",
        summary="Experiment package is editable",
        source_ref="repository://src/tiktok2026/experiment",
    )
    capabilities = ResearchCapabilities(
        repository=Reader(evidence),
        data=EmptyReader(),
        memory=EmptyReader(),
        literature=EmptyReader(),
    )
    agent = ResearchAgent(OpenAICompatibleClient(ModelSettings(), transport), capabilities)
    result = await agent.invoke(request())
    assert isinstance(result, ResearchDecision)
    assert len(transport.requests) == 2


async def test_research_rejects_test_label_evidence(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    evidence = EvidenceItem(
        evidence_id="test-1",
        kind="data",
        summary="hidden labels",
        source_ref="dataset://test",
        contains_test_labels=True,
    )
    capabilities = ResearchCapabilities(
        repository=Reader(evidence),
        data=EmptyReader(),
        memory=EmptyReader(),
        literature=EmptyReader(),
    )
    agent = ResearchAgent(
        OpenAICompatibleClient(ModelSettings(), RecordingTransport([])), capabilities
    )
    result = await agent.invoke(request())
    assert isinstance(result, AgentFailure)
    assert result.kind == "policy"
