import json
import subprocess
from pathlib import Path

from pytest import MonkeyPatch

import tiktok2026.adapters as adapters
from tests.agents.test_agents import RecordingTransport
from tiktok2026.adapters import (
    IMPLEMENTOR_CHECK_NAMES,
    RoleSpecificAgentClient,
    ScopedWorktreeRepository,
)
from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.orchestration.agent import OrchestrationAgent
from tiktok2026.agents.validator.agent import ValidatorAgent
from tiktok2026.config import ModelSettings
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    DecisionAction,
    ExperimentHistoryContext,
    ExperimentSpec,
    Fidelity,
    ImplementationEdit,
    ImplementationRequest,
    ImplementationResourceEstimate,
    ImplementationResult,
    ImplementationSubmission,
    OrchestrationDecision,
    OrchestrationRequest,
    ProposalSummary,
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


async def test_production_orchestration_accepts_pending_proposal_target(
    monkeypatch: MonkeyPatch,
) -> None:
    payload = OrchestrationDecision(
        decision_id="decision-pending",
        action=DecisionAction.IMPLEMENT,
        target_experiment_id="pending-exp",
        rationale="Use the controller-authorized pending proposal",
    ).model_dump(mode="json")
    agent = RoleSpecificAgentClient(
        client(monkeypatch, payload),
        AgentRole.ORCHESTRATION,
        "Select one allowed action.",
    )
    request = OrchestrationRequest(
        request_id="request-pending",
        run_id="run-1",
        phase="research",
        allowed_actions=(DecisionAction.IMPLEMENT,),
        resource_state=ResourceState(
            remaining_gpu_hours=1,
            accumulated_gpu_hours=0,
            remaining_wall_seconds=100,
            used_tokens=0,
            remaining_tokens=1000,
            disk_bytes_available=1000,
            reserved_final_gpu_hours=0.25,
        ),
        experiment_history=ExperimentHistoryContext(
            evidence_id="history-1",
            run_id="run-1",
            pending_proposals=(
                ProposalSummary(experiment_id="pending-exp", hypothesis="h", mechanism="m"),
            ),
            total_pending_proposals=1,
        ),
    )

    result = await agent.invoke(request)

    assert isinstance(result, OrchestrationDecision)
    assert result.target_experiment_id == "pending-exp"


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
        implementation_resource_estimate=ImplementationResourceEstimate(
            predicted_wall_seconds=10.0,
            predicted_peak_memory_bytes=100,
            predicted_artifact_bytes=100,
            dataset_passes=1,
        ),
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


async def test_production_implementor_uses_repository_diff_as_authority(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    valid = ImplementationResult(
        experiment_id="exp-1",
        patch_artifact_id="inline-patch",
        changed_files=(),
        edits=(),
    ).model_dump(mode="json")
    transport = RecordingTransport([{"choices": [{"message": {"content": json.dumps(valid)}}]}])
    repository = ScopedRepository()
    repository.write("src/tiktok2026/experiment/model.py", "VALUE = 1\n")
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
    assert len(transport.requests) == 1


def _git_repo_with_entrypoint(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(("git", "config", "user.name", "test"), cwd=repository, check=True)
    package = repository / "src/tiktok2026/experiment"
    package.mkdir(parents=True)
    (repository / "src/tiktok2026/__init__.py").write_text("\n", encoding="utf-8")
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    (package / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(
        ("git", "-c", "commit.gpgsign=false", "commit", "-qm", "base"),
        cwd=repository,
        check=True,
    )
    return repository


def _implementation_request() -> ImplementationRequest:
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
    return ImplementationRequest(
        request_id="request-1",
        experiment_id="exp-1",
        experiment_spec=spec,
        allowed_scopes=spec.implementation_scope,
    )


async def test_agentic_implementor_runs_all_controller_checks(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    repository = _git_repo_with_entrypoint(tmp_path)
    calls: list[tuple[tuple[str, ...], int]] = []

    async def fake_invoke_agentic(*args: object, **kwargs: object) -> ImplementationSubmission:
        handler = args[7]
        assert callable(handler)
        handler(
            "write_file",
            {"path": "src/tiktok2026/experiment/train.py", "content": "VALUE = 2\n"},
        )  # type: ignore[operator]
        return ImplementationSubmission(
            experiment_id="exp-1",
            patch_artifact_id="patch-1",
            changed_files=("src/tiktok2026/experiment/train.py",),
            edits=(
                ImplementationEdit(
                    relative_path="src/tiktok2026/experiment/train.py", content="VALUE = 2\n"
                ),
            ),
            checks=("model-claimed-check",),
        )

    def run_check(
        _repository: ScopedWorktreeRepository,
        command: tuple[str, ...],
        timeout_seconds: int,
    ) -> str:
        calls.append((command, timeout_seconds))
        return "passed"

    monkeypatch.setattr(adapters, "invoke_agentic", fake_invoke_agentic)
    monkeypatch.setattr(ScopedWorktreeRepository, "run_check", run_check)
    agent = RoleSpecificAgentClient(
        client(monkeypatch, {}),
        AgentRole.IMPLEMENTOR,
        "implement",
        scoped_repository=ScopedWorktreeRepository(
            repository, ("src/tiktok2026/experiment",)
        ),
    )

    result = await agent.invoke(_implementation_request())

    assert isinstance(result, ImplementationResult)
    assert len(calls) == len(IMPLEMENTOR_CHECK_NAMES)
    assert all(command for command, _ in calls)
    assert [timeout for _, timeout in calls] == [30] * len(IMPLEMENTOR_CHECK_NAMES)
    assert result.checks == IMPLEMENTOR_CHECK_NAMES


async def test_agentic_implementor_check_failure_blocks_submission(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    repository = _git_repo_with_entrypoint(tmp_path)
    failed_check = "ruff_entrypoint"
    calls: list[str] = []

    async def fake_invoke_agentic(*args: object, **kwargs: object) -> ImplementationSubmission:
        handler = args[7]
        assert callable(handler)
        handler(
            "write_file",
            {"path": "src/tiktok2026/experiment/train.py", "content": "VALUE = 2\n"},
        )  # type: ignore[operator]
        return ImplementationSubmission(
            experiment_id="exp-1",
            patch_artifact_id="patch-1",
            changed_files=("src/tiktok2026/experiment/train.py",),
            edits=(
                ImplementationEdit(
                    relative_path="src/tiktok2026/experiment/train.py", content="VALUE = 2\n"
                ),
            ),
            checks=IMPLEMENTOR_CHECK_NAMES,
        )

    def run_check(
        _repository: ScopedWorktreeRepository,
        command: tuple[str, ...],
        timeout_seconds: int,
    ) -> str:
        del timeout_seconds
        check = IMPLEMENTOR_CHECK_NAMES[len(calls)]
        calls.append(check)
        if check == failed_check:
            raise ValueError("synthetic check failure")
        return "passed"

    monkeypatch.setattr(adapters, "invoke_agentic", fake_invoke_agentic)
    monkeypatch.setattr(ScopedWorktreeRepository, "run_check", run_check)
    agent = RoleSpecificAgentClient(
        client(monkeypatch, {}),
        AgentRole.IMPLEMENTOR,
        "implement",
        scoped_repository=ScopedWorktreeRepository(
            repository, ("src/tiktok2026/experiment",)
        ),
    )

    result = await agent.invoke(_implementation_request())

    assert isinstance(result, AgentFailure)
    assert result.role == AgentRole.IMPLEMENTOR
    assert failed_check in result.message
    failed_index = IMPLEMENTOR_CHECK_NAMES.index(failed_check)
    assert calls == list(IMPLEMENTOR_CHECK_NAMES[: failed_index + 1])


async def test_agentic_implementor_repairs_after_guarded_submission(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    repository = _git_repo_with_entrypoint(tmp_path)
    submission = ImplementationSubmission(
        experiment_id="exp-1",
        patch_artifact_id="patch-1",
        changed_files=("src/tiktok2026/experiment/train.py",),
        edits=(
            ImplementationEdit(
                relative_path="src/tiktok2026/experiment/train.py", content="VALUE = 2\n"
            ),
        ),
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "write-1", "type": "function", "function": {
                    "name": "write_file", "arguments": json.dumps({
                        "path": "src/tiktok2026/experiment/train.py", "content": "def broken(:\n"
                    })
                }}
            ]}}]},
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "submit-1", "type": "function", "function": {
                    "name": "submit_result", "arguments": json.dumps(submission)
                }}
            ]}}]},
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "write-2", "type": "function", "function": {
                    "name": "write_file", "arguments": json.dumps({
                        "path": "src/tiktok2026/experiment/train.py", "content": "VALUE = 2\n"
                    })
                }}
            ]}}]},
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "submit-2", "type": "function", "function": {
                    "name": "submit_result", "arguments": json.dumps(submission)
                }}
            ]}}]},
        ]
    )
    checks: list[str] = []

    def run_check(
        _repository: ScopedWorktreeRepository,
        command: tuple[str, ...],
        timeout_seconds: int,
    ) -> str:
        del timeout_seconds
        checks.append(command[0] if command[0] != "git" else command[1])
        if len(checks) == 1:
            raise ValueError("synthetic compile failure")
        return "passed"

    monkeypatch.setattr(ScopedWorktreeRepository, "run_check", run_check)
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "implement",
        scoped_repository=ScopedWorktreeRepository(
            repository, ("src/tiktok2026/experiment",)
        ),
    )

    result = await agent.invoke(_implementation_request())

    assert isinstance(result, ImplementationResult)
    assert len(transport.requests) == 4
    assert "compile_entrypoint" in str(transport.requests[2][1]["messages"])
    assert len(checks) == 1 + len(IMPLEMENTOR_CHECK_NAMES) * 2


async def test_agentic_implementor_permanent_guard_failure_exhausts_loop(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    repository = _git_repo_with_entrypoint(tmp_path)
    submission = ImplementationSubmission(
        experiment_id="exp-1",
        patch_artifact_id="patch-1",
        changed_files=("src/tiktok2026/experiment/train.py",),
        edits=(
            ImplementationEdit(
                relative_path="src/tiktok2026/experiment/train.py", content="VALUE = 2\n"
            ),
        ),
    ).model_dump(mode="json")
    transport = RecordingTransport(
        [
            {"choices": [{"message": {"content": None, "tool_calls": [
                {"id": "write-1", "type": "function", "function": {
                    "name": "write_file", "arguments": json.dumps({
                        "path": "src/tiktok2026/experiment/train.py", "content": "VALUE = 2\n"
                    })
                }}
            ]}}]},
            *[
                {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": f"submit-{index}", "type": "function", "function": {
                        "name": "submit_result", "arguments": json.dumps(submission)
                    }}
                ]}}]}
                for index in range(31)
            ],
        ]
    )

    def run_check(
        _repository: ScopedWorktreeRepository,
        _command: tuple[str, ...],
        _timeout_seconds: int,
    ) -> str:
        raise ValueError("permanent check failure")

    monkeypatch.setattr(ScopedWorktreeRepository, "run_check", run_check)
    agent = RoleSpecificAgentClient(
        OpenAICompatibleClient(ModelSettings(), transport),
        AgentRole.IMPLEMENTOR,
        "implement",
        scoped_repository=ScopedWorktreeRepository(
            repository, ("src/tiktok2026/experiment",)
        ),
    )

    result = await agent.invoke(_implementation_request())

    assert isinstance(result, AgentFailure)
    assert "exceeded 32 turns" in result.message
    assert len(transport.requests) == 32


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
