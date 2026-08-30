import json
from pathlib import Path

from pytest import MonkeyPatch

from tests.agents.test_agents import RecordingTransport
from tiktok2026.adapters import RoleSpecificAgentClient
from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.implementor.agent import ImplementorAgent
from tiktok2026.agents.orchestration.agent import OrchestrationAgent
from tiktok2026.agents.validator.agent import ValidatorAgent
from tiktok2026.config import ModelSettings
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    DecisionAction,
    ExperimentSpec,
    Fidelity,
    ImplementationEdit,
    ImplementationRequest,
    ImplementationResult,
    OrchestrationDecision,
    OrchestrationRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
)


class ScopedRepository:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    def write(self, path: str, content: str) -> None:
        self.writes.append((path, content))

    def diff(self) -> str:
        return "\n".join(f"{path}:{content}" for path, content in self.writes)

    def changed_files(self) -> tuple[str, ...]:
        return tuple(path for path, _ in self.writes)


def client(monkeypatch: MonkeyPatch, payload: dict[str, object]) -> OpenAICompatibleClient:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    return OpenAICompatibleClient(
        ModelSettings(),
        RecordingTransport([{"choices": [{"message": {"content": json.dumps(payload)}}]}]),
    )


async def test_orchestration_returns_typed_decision(monkeypatch: MonkeyPatch) -> None:
    payload = OrchestrationDecision(
        decision_id="decision-1",
        action=DecisionAction.RESEARCH,
        rationale="More evidence is required",
    ).model_dump(mode="json")
    result = await OrchestrationAgent(client(monkeypatch, payload)).invoke(
        "request-1", {"allowed_actions": ["research", "stop"]}
    )
    assert isinstance(result, OrchestrationDecision)


async def test_orchestration_repairs_malformed_json_once(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    payload = OrchestrationDecision(
        decision_id="decision-1",
        action=DecisionAction.RESEARCH,
        rationale="More evidence is required",
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": "not json"}}]},
            {"choices": [{"message": {"content": json.dumps(payload)}}]},
        ]
    )

    result = await OrchestrationAgent(
        OpenAICompatibleClient(ModelSettings(), transport)
    ).invoke("request-1", {"allowed_actions": ["research", "stop"]})

    assert isinstance(result, OrchestrationDecision)
    assert len(transport.requests) == 2
    first_user_message = transport.requests[0][1]["messages"][1]["content"]  # type: ignore[index]
    repair_user_message = transport.requests[1][1]["messages"][1]["content"]  # type: ignore[index]
    assert '"response_json_schema"' in first_user_message
    assert '"validation_error"' in repair_user_message


async def test_production_orchestration_rejects_disallowed_action(
    monkeypatch: MonkeyPatch,
) -> None:
    payload = OrchestrationDecision(
        decision_id="decision-1",
        action=DecisionAction.STOP,
        rationale="Stop before establishing a candidate",
    ).model_dump(mode="json")
    agent = RoleSpecificAgentClient(
        client(monkeypatch, payload),
        AgentRole.ORCHESTRATION,
        "Select one allowed action.",
    )
    request = OrchestrationRequest(
        request_id="request-1",
        run_id="run-1",
        phase="research",
        allowed_actions=(DecisionAction.RESEARCH,),
        resource_state=ResourceState(
            remaining_gpu_hours=1,
            accumulated_gpu_hours=0,
            remaining_wall_seconds=100,
            used_tokens=0,
            remaining_tokens=1000,
            disk_bytes_available=1000,
            reserved_final_gpu_hours=0.25,
        ),
    )

    result = await agent.invoke(request)

    assert isinstance(result, AgentFailure)
    assert result.kind == "policy"
    assert "disallowed action: stop" in result.message


async def test_production_orchestration_repairs_research_target(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    invalid = {
        "decision_id": "decision-1",
        "action": "research",
        "target_experiment_id": "historical-exp",
        "rationale": "Research from historical evidence",
    }
    valid = {
        **invalid,
        "target_experiment_id": None,
    }
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": json.dumps(invalid)}}]},
            {"choices": [{"message": {"content": json.dumps(valid)}}]},
        ]
    )
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.ORCHESTRATION,
        "Select one allowed action.",
    )
    request = OrchestrationRequest(
        request_id="request-1",
        run_id="run-1",
        phase="research",
        allowed_actions=(DecisionAction.RESEARCH,),
        resource_state=ResourceState(
            remaining_gpu_hours=1,
            accumulated_gpu_hours=0,
            remaining_wall_seconds=100,
            used_tokens=0,
            remaining_tokens=1000,
            disk_bytes_available=1000,
            reserved_final_gpu_hours=0.25,
        ),
    )

    result = await agent.invoke(request)

    assert isinstance(result, OrchestrationDecision)
    assert result.target_experiment_id is None
    assert len(transport.requests) == 2


async def test_proposal_request_repairs_evidence_request(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    invalid = ResearchDecision(
        request_id="request-1",
        kind="evidence_request",
        message="Need repository evidence",
    ).model_dump(mode="json")
    spec = ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="test hypothesis",
        mechanism="test mechanism",
        motivation="test motivation",
        expected_signal="test signal",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="test success",
        failure_criteria="test failure",
    )
    valid = ResearchDecision(
        request_id="request-1",
        kind="proposal",
        experiment_spec=spec,
        message="Propose a bounded smoke test",
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": json.dumps(invalid)}}]},
            {"choices": [{"message": {"content": json.dumps(valid)}}]},
        ]
    )
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.RESEARCH,
        "Return one evidence-backed ResearchDecision as JSON.",
    )
    request = ResearchRequest(
        request_id="request-1",
        objective="propose next experiment",
        resource_state=ResourceState(
            remaining_gpu_hours=1,
            accumulated_gpu_hours=0,
            remaining_wall_seconds=100,
            used_tokens=0,
            remaining_tokens=1000,
            disk_bytes_available=1000,
            reserved_final_gpu_hours=0.25,
        ),
    )

    result = await agent.invoke(request)

    assert isinstance(result, ResearchDecision)
    assert result.experiment_spec == spec
    assert len(transport.requests) == 2


async def test_implementor_cannot_write_protected_path(monkeypatch: MonkeyPatch) -> None:
    payload = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="patch-1",
        changed_files=("baseline/data.py",),
    ).model_dump(mode="json")
    repository = ScopedRepository()
    result = await ImplementorAgent(client(monkeypatch, payload), repository).invoke(
        "request-1", "exp-1", ("src/tiktok2026/experiment",)
    )
    assert isinstance(result, AgentFailure)
    assert result.role == AgentRole.IMPLEMENTOR
    assert not repository.writes


async def test_production_implementor_repairs_empty_edits(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    invalid = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="inline-patch",
        changed_files=(),
    ).model_dump(mode="json")
    valid = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="inline-patch",
        changed_files=("src/tiktok2026/experiment/model.py",),
        edits=(
            ImplementationEdit(
                relative_path="src/tiktok2026/experiment/model.py", content="VALUE = 1\n"
            ),
        ),
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": json.dumps(invalid)}}]},
            {"choices": [{"message": {"content": json.dumps(valid)}}]},
        ]
    )
    repository = ScopedRepository()
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "Apply bounded edits.",
        scoped_repository=repository,
    )
    spec = ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="hypothesis",
        mechanism="mechanism",
        motivation="motivation",
        expected_signal="signal",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="success",
        failure_criteria="failure",
    )

    result = await agent.invoke(
        ImplementationRequest(
            request_id="request-1",
            experiment_id="exp-1",
            experiment_spec=spec,
            allowed_scopes=spec.implementation_scope,
        )
    )

    assert isinstance(result, ImplementationResult)
    assert result.changed_files == ("src/tiktok2026/experiment/model.py",)
    assert len(transport.requests) == 2


async def test_validator_returns_read_only_report(monkeypatch: MonkeyPatch) -> None:
    payload = ValidationReport(
        report_id="report-1",
        experiment_id="exp-1",
        stage=ValidationStage.PROPOSAL,
        verdict=ValidationVerdict.APPROVED,
        leakage_risk="none",
    ).model_dump(mode="json")
    result = await ValidatorAgent(client(monkeypatch, payload)).invoke(
        "request-1", {"experiment_id": "exp-1"}
    )
    assert isinstance(result, ValidationReport)


def test_agent_packages_are_exactly_four() -> None:
    root = Path("src/tiktok2026/agents")
    roles = {path.name for path in root.iterdir() if path.is_dir() and path.name != "common"}
    assert roles == {"orchestration", "research", "implementor", "validator"}
