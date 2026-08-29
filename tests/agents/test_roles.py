import json
from pathlib import Path

from pytest import MonkeyPatch

from tests.agents.test_agents import RecordingTransport
from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.implementor.agent import ImplementorAgent
from tiktok2026.agents.orchestration.agent import OrchestrationAgent
from tiktok2026.agents.validator.agent import ValidatorAgent
from tiktok2026.config import ModelSettings
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    DecisionAction,
    ImplementationResult,
    OrchestrationDecision,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
)


class ScopedRepository:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    def write(self, path: str, content: str) -> None:
        self.writes.append((path, content))


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
