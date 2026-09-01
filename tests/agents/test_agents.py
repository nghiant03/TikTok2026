import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import httpx
from pytest import MonkeyPatch, raises

from tiktok2026.adapters import RoleSpecificAgentClient
from tiktok2026.agents.common.client import ChatTransport, OpenAICompatibleClient
from tiktok2026.agents.common.structured import invoke_agentic
from tiktok2026.agents.research.agent import ResearchAgent
from tiktok2026.agents.research.context import ResearchCapabilities
from tiktok2026.agents.research.online import (
    HttpxPageFetcher,
    LiteLLMSearchProvider,
    OpenAIWebSearchProvider,
)
from tiktok2026.config import LiteLLMSearchSettings, ModelSettings
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ContractModel,
    EvidenceItem,
    OnlineSearchRequest,
    OnlineSearchResult,
    OnlineSource,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    ValidationVerdict,
)


class RecordingTransport(ChatTransport):
    def __init__(self, responses: Sequence[dict[str, object] | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, object], dict[str, str]]] = []

    async def post_json(
        self, url: str, payload: dict[str, object], headers: dict[str, str], timeout: float
    ) -> dict[str, object]:
        self.requests.append((url, payload, headers))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _validation_operation(stage: ValidationStage) -> ValidationOperationIdentity:
    return ValidationOperationIdentity(
        operation_id="validation-operation-test",
        run_id="run-1",
        experiment_id="exp-1",
        stage=stage,
        repair_attempt=0,
        subject_sha256=hashlib.sha256(b"{}").hexdigest(),
        implementation_diff_sha256="a" * 64
        if stage == ValidationStage.IMPLEMENTATION
        else None,
    )


class Reader:
    def __init__(self, evidence: EvidenceItem) -> None:
        self.evidence = evidence

    async def read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        return (self.evidence,)


class EmptyReader:
    async def read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        return ()


class OnlineProvider:
    def __init__(self) -> None:
        self.requests: list[OnlineSearchRequest] = []

    async def search(self, request: OnlineSearchRequest) -> OnlineSearchResult:
        self.requests.append(request)
        return OnlineSearchResult(
            result_id="search-result-1",
            request_id=request.request_id,
            provider="fixture",
            sources=(
                OnlineSource(
                    source_id="online-source-1",
                    url="https://example.org/paper",
                    title="A current paper",
                    excerpt="A bounded scientific finding",
                    content_sha256=hashlib.sha256(
                        b"A bounded scientific finding"
                    ).hexdigest(),
                ),
            ),
        )


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
    assert "reasoning_effort" not in transport.requests[0][1]


async def test_openai_compatible_client_sends_reasoning_effort_when_set(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [{"choices": [{"message": {"content": json.dumps({"message": "ok"})}}]}]
    )
    client = OpenAICompatibleClient(
        ModelSettings(model="deepseek-v4-pro", reasoning_effort="xhigh"), transport
    )
    await client.complete("system", "user")
    assert transport.requests[0][1]["reasoning_effort"] == "xhigh"


async def test_openai_compatible_client_strips_markdown_fences(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    fenced = '```json\n{"message": "ok"}\n```'
    transport = RecordingTransport(
        [{"choices": [{"message": {"content": fenced}}]}]
    )
    client = OpenAICompatibleClient(ModelSettings(), transport)
    output = await client.complete("system", "user")
    assert output == {"message": "ok"}


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


async def test_openai_compatible_client_retries_read_timeout_once(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [
            httpx.ReadTimeout("provider timed out"),
            _final_response({"message": "recovered"}),
        ]
    )

    result = await OpenAICompatibleClient(ModelSettings(), transport).complete_with_tools(
        [], []
    )

    assert result["content"] == json.dumps({"message": "recovered"})
    assert len(transport.requests) == 2


async def test_openai_compatible_client_stops_after_second_read_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [httpx.ReadTimeout("first timeout"), httpx.ReadTimeout("second timeout")]
    )

    with raises(httpx.ReadTimeout, match="second timeout"):
        await OpenAICompatibleClient(ModelSettings(), transport).complete_with_tools([], [])

    assert len(transport.requests) == 2


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


async def test_production_research_searches_online_and_enforces_citations(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = OnlineProvider()
    decision = ResearchDecision(
        request_id="request-1",
        kind="evidence_request",
        message="Use the current finding",
        evidence_refs=("online-source-1",),
    )
    transport = RecordingTransport(
        [
            _tool_calls_response(
                "search_online", {"query": "recent recommender calibration evidence"}
            ),
            _tool_calls_response("submit_result", decision.model_dump(mode="json")),
        ]
    )
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.RESEARCH,
        "Research with evidence.",
        online_research=provider,
        max_online_searches=2,
        max_online_results=4,
        online_allowed_domains=("example.org",),
    )

    result = await agent.invoke(request())

    assert isinstance(result, ResearchDecision)
    assert result.evidence_refs == ("online-source-1",)
    assert provider.requests[0].allowed_domains == ("example.org",)
    assert provider.requests[0].max_results == 4
    tools = transport.requests[0][1]["tools"]
    assert isinstance(tools, list)
    assert [tool["function"]["name"] for tool in tools] == [  # type: ignore[index]
        "search_online",
        "submit_result",
    ]


async def test_production_research_repairs_unknown_online_citation(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    provider = OnlineProvider()
    invalid = ResearchDecision(
        request_id="request-1",
        kind="evidence_request",
        message="Unsupported citation",
        evidence_refs=("unknown-source",),
    )
    valid = invalid.model_copy(update={"evidence_refs": ("online-source-1",)})
    transport = RecordingTransport(
        [
            _tool_calls_response("search_online", {"query": "recent recommender evidence"}),
            _tool_calls_response("submit_result", invalid.model_dump(mode="json")),
            _tool_calls_response("submit_result", valid.model_dump(mode="json")),
        ]
    )
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.RESEARCH,
        "Research with evidence.",
        online_research=provider,
    )

    result = await agent.invoke(request())

    assert isinstance(result, ResearchDecision)
    assert result.evidence_refs == ("online-source-1",)
    messages = transport.requests[2][1]["messages"]
    assert isinstance(messages, list)
    assert any("unknown evidence IDs" in str(message) for message in messages)


async def test_openai_web_search_records_only_authorized_https_sources(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [
            {
                "id": "response-1",
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "sources": [
                                {"url": "https://arxiv.org/abs/1234", "title": "Paper"},
                                {"url": "http://localhost/private", "title": "Blocked"},
                            ]
                        },
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "The paper reports a current result.",
                                "annotations": [],
                            }
                        ],
                    },
                ],
            }
        ]
    )
    provider = OpenAIWebSearchProvider(ModelSettings(), tmp_path, transport)

    result = await provider.search(
        OnlineSearchRequest(
            request_id="online-1",
            query="recent recommender evaluation methods",
            allowed_domains=("arxiv.org",),
        )
    )

    assert tuple(source.url for source in result.sources) == (
        "https://arxiv.org/abs/1234",
    )
    assert (tmp_path / f"{result.result_id}.json").is_file()
    assert transport.requests[0][0].endswith("/responses")
    web_tool = transport.requests[0][1]["tools"]
    assert isinstance(web_tool, list)
    assert web_tool[0]["filters"] == {"allowed_domains": ["arxiv.org"]}  # type: ignore[index]


class RecordingPageFetcher:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.requests: list[tuple[str, tuple[str, ...]]] = []

    async def fetch(self, url: str, allowed_domains: tuple[str, ...]) -> str | None:
        self.requests.append((url, allowed_domains))
        return self.text


async def test_litellm_search_reads_authorized_pages_and_persists_evidence(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "proxy-secret")
    transport = RecordingTransport(
        [
            {
                "object": "search",
                "results": [
                    {
                        "title": "Current paper",
                        "url": "https://arxiv.org/abs/1234",
                        "snippet": "Search snippet",
                    },
                    {
                        "title": "Blocked result",
                        "url": "https://example.org/private",
                        "snippet": "Must not be returned",
                    },
                ],
            }
        ]
    )
    page_fetcher = RecordingPageFetcher("Full authorized page text")
    provider = LiteLLMSearchProvider(
        LiteLLMSearchSettings(),
        tmp_path,
        transport,
        page_fetcher,
    )

    result = await provider.search(
        OnlineSearchRequest(
            request_id="online-1",
            query="recent recommender evaluation methods",
            allowed_domains=("arxiv.org",),
        )
    )

    assert result.provider == "litellm_search"
    assert tuple(source.url for source in result.sources) == (
        "https://arxiv.org/abs/1234",
    )
    assert result.sources[0].excerpt == "Full authorized page text"
    assert page_fetcher.requests == [
        ("https://arxiv.org/abs/1234", ("arxiv.org",)),
    ]
    assert transport.requests[0][0].endswith("/search/research-search")
    assert transport.requests[0][1] == {
        "query": "recent recommender evaluation methods",
        "max_results": 5,
        "search_domain_filter": ["arxiv.org"],
    }
    assert (tmp_path / f"{result.result_id}.json").is_file()


async def test_litellm_search_rejects_malformed_results(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "proxy-secret")
    provider = LiteLLMSearchProvider(
        LiteLLMSearchSettings(),
        tmp_path,
        RecordingTransport([{"object": "search", "results": [{"url": 3}]}]),
        RecordingPageFetcher(None),
    )

    with raises(ValueError, match="requires title, url, and snippet strings"):
        await provider.search(
            OnlineSearchRequest(request_id="online-1", query="recommender evaluation")
        )


async def test_page_fetcher_extracts_text_and_blocks_disallowed_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/paper":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<html><style>hidden</style><body>Useful <b>paper</b> text</body></html>",
            )
        return httpx.Response(302, headers={"location": "https://example.org/private"})

    fetcher = HttpxPageFetcher(httpx.MockTransport(handler))

    assert await fetcher.fetch("https://arxiv.org/paper", ("arxiv.org",)) == (
        "Useful paper text"
    )
    assert await fetcher.fetch("https://arxiv.org/redirect", ("arxiv.org",)) is None


# ---------------------------------------------------------------------------
# invoke_agentic — multi-turn tool-use loop
# ---------------------------------------------------------------------------


class _FakeResult(ContractModel):
    message: str


def _tool_calls_response(tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


def _final_response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(payload)},
            }
        ]
    }


async def test_agentic_loop_executes_tool_and_returns_result(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    tool_calls_log: list[tuple[str, dict[str, object]]] = []

    def handler(name: str, args: dict[str, object]) -> str:
        tool_calls_log.append((name, args))
        return "check output"

    transport = RecordingTransport(
        [
            _tool_calls_response("run_check", {"command": ["python", "-c", "import x"]}),
            _final_response({"message": "done"}),
        ]
    )
    client = OpenAICompatibleClient(ModelSettings(), transport)
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "run_check",
                "description": "Run a check",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = await invoke_agentic(
        client,
        AgentRole.IMPLEMENTOR,
        "test-1",
        _FakeResult,
        "system prompt",
        {"request_id": "test-1"},
        tools,
        handler,
    )
    assert isinstance(result, _FakeResult)
    assert result.message == "done"
    assert len(tool_calls_log) == 1
    assert tool_calls_log[0][0] == "run_check"


async def test_agentic_loop_turn_limit(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    def handler(name: str, args: dict[str, object]) -> str:
        return "ok"

    # Every turn returns tool calls — never terminates.
    transport = RecordingTransport(
        [_tool_calls_response("noop", {}) for _ in range(25)]
    )
    client = OpenAICompatibleClient(ModelSettings(), transport)
    result = await invoke_agentic(
        client,
        AgentRole.IMPLEMENTOR,
        "test-2",
        _FakeResult,
        "system prompt",
        {},
        [],
        handler,
        max_turns=3,
    )
    assert isinstance(result, AgentFailure)
    assert "turns" in result.message
    assert result.repair_attempts == 1


async def test_agentic_loop_can_submit_after_twenty_tool_turns(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    transport = RecordingTransport(
        [
            *[_tool_calls_response("run_check", {}) for _ in range(20)],
            _tool_calls_response("submit_result", {"message": "done"}),
        ]
    )
    result = await invoke_agentic(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "test-long-implementation",
        _FakeResult,
        "system prompt",
        {},
        [],
        lambda _name, _arguments: "ok",
        max_turns=32,
        terminal_tool="submit_result",
    )

    assert isinstance(result, _FakeResult)
    assert result.message == "done"
    assert len(transport.requests) == 21


async def test_agentic_loop_model_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport([{"choices": []}, {"choices": []}])

    def handler(name: str, args: dict[str, object]) -> str:
        return "ok"

    client = OpenAICompatibleClient(ModelSettings(), transport)
    result = await invoke_agentic(
        client,
        AgentRole.IMPLEMENTOR,
        "test-3",
        _FakeResult,
        "system prompt",
        {},
        [],
        handler,
    )
    assert isinstance(result, AgentFailure)
    assert result.kind == "model"
    assert "no choices" in result.message
    assert len(transport.requests) == 2


async def test_agentic_loop_retries_empty_choices_without_consuming_turn(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [{"choices": []}, _final_response({"message": "recovered"})]
    )

    def handler(name: str, args: dict[str, object]) -> str:
        return "ok"

    client = OpenAICompatibleClient(ModelSettings(), transport)
    result = await invoke_agentic(
        client,
        AgentRole.VALIDATOR,
        "test-empty-choices",
        _FakeResult,
        "system prompt",
        {},
        [],
        handler,
        max_turns=1,
    )
    assert isinstance(result, _FakeResult)
    assert result.message == "recovered"
    assert len(transport.requests) == 2


async def test_agentic_loop_final_schema_error(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport([_final_response({"wrong_field": "oops"})])

    def handler(name: str, args: dict[str, object]) -> str:
        return "ok"

    client = OpenAICompatibleClient(ModelSettings(), transport)
    result = await invoke_agentic(
        client,
        AgentRole.IMPLEMENTOR,
        "test-4",
        _FakeResult,
        "system prompt",
        {},
        [],
        handler,
    )
    assert isinstance(result, AgentFailure)
    assert result.kind == "schema"


async def test_agentic_loop_tool_error_returned_as_content(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    def handler(name: str, args: dict[str, object]) -> str:
        raise PermissionError("blocked path")

    transport = RecordingTransport(
        [
            _tool_calls_response("write_file", {"path": "baseline/data.py"}),
            _final_response({"message": "gave up"}),
        ]
    )
    client = OpenAICompatibleClient(ModelSettings(), transport)
    tools: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    result = await invoke_agentic(
        client,
        AgentRole.IMPLEMENTOR,
        "test-5",
        _FakeResult,
        "system prompt",
        {},
        tools,
        handler,
    )
    assert isinstance(result, _FakeResult)
    # The tool error was sent back as tool content, not raised.
    messages = transport.requests[1][1].get("messages", [])
    tool_messages = [
        m for m in messages  # type: ignore[union-attr]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "PermissionError" in str(tool_messages[0]["content"])


async def test_agentic_terminal_tool_validates_and_returns_result(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [
            _tool_calls_response("submit_result", {"wrong_field": "retry"}),
            _tool_calls_response("submit_result", {"message": "done"}),
        ]
    )

    def handler(name: str, args: dict[str, object]) -> str:
        raise AssertionError("terminal tool must not reach the regular handler")

    result = await invoke_agentic(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "test-terminal",
        _FakeResult,
        "system prompt",
        {},
        [],
        handler,
        terminal_tool="submit_result",
    )
    assert isinstance(result, _FakeResult)
    assert result.message == "done"
    messages = transport.requests[1][1]["messages"]
    assert "result rejected" in str(messages)


async def test_agentic_terminal_tool_does_not_duplicate_response_schema(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport([_tool_calls_response("submit_result", {"message": "done"})])

    result = await invoke_agentic(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "test-terminal-schema",
        _FakeResult,
        "system prompt",
        {"request_id": "test-terminal-schema"},
        [],
        lambda _name, _arguments: "ok",
        terminal_tool="submit_result",
    )

    assert isinstance(result, _FakeResult)
    user_message = transport.requests[0][1]["messages"][1]  # type: ignore[index]
    assert "response_json_schema" not in str(user_message)


async def test_agentic_terminal_guard_rejects_then_accepts(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [
            _tool_calls_response("submit_result", {"message": "first"}),
            _tool_calls_response("submit_result", {"message": "repaired"}),
        ]
    )
    guard_calls = 0

    def guard(_result: _FakeResult) -> str | None:
        nonlocal guard_calls
        guard_calls += 1
        return "controller check failed" if guard_calls == 1 else None

    result = await invoke_agentic(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "test-guard",
        _FakeResult,
        "system prompt",
        {},
        [],
        lambda _name, _arguments: "ok",
        terminal_tool="submit_result",
        terminal_guard=guard,
    )

    assert isinstance(result, _FakeResult)
    assert result.message == "repaired"
    assert guard_calls == 2
    assert "controller check failed" in str(transport.requests[1][1]["messages"])


async def test_agentic_terminal_guard_exhaustion_returns_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    transport = RecordingTransport(
        [_tool_calls_response("submit_result", {"message": "still broken"}) for _ in range(2)]
    )

    result = await invoke_agentic(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "test-guard-exhaustion",
        _FakeResult,
        "system prompt",
        {},
        [],
        lambda _name, _arguments: "ok",
        max_turns=2,
        terminal_tool="submit_result",
        terminal_guard=lambda _result: "permanent controller failure",
    )

    assert isinstance(result, AgentFailure)
    assert "exceeded 2 turns" in result.message
    assert len(transport.requests) == 2


async def test_validator_agentic_tools_are_read_only_and_can_run_checks(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "test"), cwd=repository, check=True
    )
    target = repository / "src/tiktok2026/experiment"
    target.mkdir(parents=True)
    train = target / "train.py"
    train.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(
        ("git", "-c", "commit.gpgsign=false", "commit", "-qm", "base"),
        cwd=repository,
        check=True,
    )
    train.write_text("VALUE = 2\n", encoding="utf-8")

    report = ValidationReport(
        report_id="report-1",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        verdict=ValidationVerdict.APPROVED,
        leakage_risk="none",
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            _tool_calls_response(
                "read_file", {"path": "src/tiktok2026/experiment/train.py"}
            ),
            _tool_calls_response(
                "run_check", {"check": "compile_entrypoint"}
            ),
            _tool_calls_response("submit_result", report),
        ]
    )
    validator = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.VALIDATOR,
        "validate",
    )
    validator.bind_worktree(repository, ("src/tiktok2026/experiment",))
    result = await validator.invoke(
        ValidationRequest(
            request_id="validation-1",
            experiment_id="exp-1",
            stage=ValidationStage.IMPLEMENTATION,
            validation_operation=_validation_operation(ValidationStage.IMPLEMENTATION),
            subject={},
        )
    )
    assert isinstance(result, ValidationReport)
    assert result.verdict == ValidationVerdict.APPROVED
    assert validator.scoped_repository is not None
    assert "VALUE = 2" in validator.scoped_repository.diff()
    with raises(PermissionError):
        validator.scoped_repository.write("src/tiktok2026/experiment/blocked.py", "blocked\n")
    first_tools = transport.requests[0][1]["tools"]
    tool_names = {
        tool["function"]["name"]  # type: ignore[index]
        for tool in first_tools  # type: ignore[union-attr]
    }
    assert tool_names == {"read_file", "run_check", "diff", "submit_result"}
    assert "compile_entrypoint" in str(transport.requests[2][1]["messages"])
    assert "controller_check_results" in str(transport.requests[0][1]["messages"])


async def test_validator_cannot_approve_failed_controller_checks(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    repository = tmp_path / "repo"
    target = repository / "src/tiktok2026/experiment"
    target.mkdir(parents=True)
    (target / "train.py").write_text("def broken(:\n", encoding="utf-8")
    report = ValidationReport(
        report_id="report-1",
        experiment_id="exp-1",
        stage=ValidationStage.IMPLEMENTATION,
        verdict=ValidationVerdict.APPROVED,
        leakage_risk="none",
    ).model_dump(mode="json")
    transport = RecordingTransport([_tool_calls_response("submit_result", report)])
    validator = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.VALIDATOR,
        "validate",
    )
    validator.bind_worktree(repository, ("src/tiktok2026/experiment",))

    result = await validator.invoke(
        ValidationRequest(
            request_id="validation-1",
            experiment_id="exp-1",
            stage=ValidationStage.IMPLEMENTATION,
            validation_operation=_validation_operation(ValidationStage.IMPLEMENTATION),
            subject={},
        )
    )

    assert isinstance(result, ValidationReport)
    assert result.verdict == ValidationVerdict.REPAIRABLE
    assert any("controller-owned check failed" in blocker for blocker in result.blockers)


async def test_bound_validator_uses_single_shot_path_outside_implementation(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    repository = tmp_path / "repo"
    repository.mkdir()
    report = ValidationReport(
        report_id="proposal-report",
        experiment_id="exp-1",
        stage=ValidationStage.PROPOSAL,
        verdict=ValidationVerdict.APPROVED,
        leakage_risk="none",
    ).model_dump(mode="json")
    transport = RecordingTransport([_final_response(report)])
    validator = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.VALIDATOR,
        "validate",
    )
    validator.bind_worktree(repository, ("src/tiktok2026/experiment",))
    result = await validator.invoke(
        ValidationRequest(
            request_id="proposal-validation",
            experiment_id="exp-1",
            stage=ValidationStage.PROPOSAL,
            validation_operation=_validation_operation(ValidationStage.PROPOSAL),
            subject={},
        )
    )
    assert isinstance(result, ValidationReport)
    assert "tools" not in transport.requests[0][1]
