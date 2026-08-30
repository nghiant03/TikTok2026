from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar, cast

from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.common.structured import invoke_agentic, invoke_structured
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    AuditEvent,
    BaselineCalibrationRecord,
    ContractModel,
    DatasetManifestIdentity,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionResult,
    ExperimentProposalDecision,
    ExperimentRegistrySnapshot,
    ExperimentSpec,
    FailureRecord,
    FinalizationBundleRequest,
    FinalizationRecord,
    ImplementationEdit,
    ImplementationRequest,
    ImplementationResult,
    ImplementationSubmission,
    OrchestrationDecision,
    OrchestrationRequest,
    PolicyDecisionModel,
    PredictionArtifactRegistration,
    ProvenanceRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceReservation,
    ResourceState,
    RunRecord,
    ScopedRepository,
    SourceRegistration,
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    ValidationVerdict,
    WorktreeAssignment,
)
from tiktok2026.observability.exports import export_records
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.persistence.resources import ResourceLedger
from tiktok2026.policies.lifecycle import can_repair, convergence_reason
from tiktok2026.policies.paths import check_changed_paths

ModelT = TypeVar("ModelT", bound=ContractModel)


def _tool(
    name: str,
    description: str,
    properties: dict[str, object],
    required: tuple[str, ...] = (),
) -> dict[str, object]:
    parameters: dict[str, object] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = list(required)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


_READ_FILE_TOOL = _tool(
    "read_file",
    "Read a file from the assigned worktree. The path must be within the allowed scope.",
    {
        "path": {"type": "string", "description": "Repository-relative POSIX path"},
        "max_characters": {
            "type": "integer",
            "description": "Maximum characters to return (default 20000)",
            "default": 20000,
        },
    },
    ("path",),
)

_WRITE_FILE_TOOL = _tool(
    "write_file",
    "Write content to a file in the assigned worktree. The path must be within the allowed scope.",
    {
        "path": {"type": "string", "description": "Repository-relative POSIX path"},
        "content": {"type": "string", "description": "Full file content to write"},
    },
    ("path", "content"),
)

_RUN_CHECK_TOOL = _tool(
    "run_check",
    "Run a command in the worktree (e.g. python -c 'import ...'). Returns stdout. "
    "Fails on non-zero exit or timeout.",
    {
        "command": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command and arguments as a list",
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Timeout in seconds (default 30)",
            "default": 30,
        },
    },
    ("command",),
)

_VALIDATOR_CHECK_TOOL = _tool(
    "run_check",
    "Run one controller-owned, non-mutating implementation check.",
    {
        "check": {
            "type": "string",
            "enum": [
                "compile_entrypoint",
                "import_entrypoint",
                "ruff_entrypoint",
                "pyright_entrypoint",
            ],
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Timeout in seconds (default 30)",
            "default": 30,
        },
    },
    ("check",),
)

_DIFF_TOOL = _tool(
    "diff",
    "Return the current git diff of all changes in the worktree.",
    {},
)


def _validator_check_commands() -> dict[str, tuple[str, ...]]:
    entrypoint = "src/tiktok2026/experiment/train.py"
    return {
        "compile_entrypoint": (
            "python",
            "-c",
            (
                "from pathlib import Path; "
                f"source=Path('{entrypoint}').read_text(); "
                f"compile(source, '{entrypoint}', 'exec')"
            ),
        ),
        "import_entrypoint": (
            "python",
            "-c",
            "import sys; sys.path.insert(0, 'src'); import tiktok2026.experiment.train",
        ),
        "ruff_entrypoint": ("ruff", "check", entrypoint),
        "pyright_entrypoint": ("pyright", entrypoint),
    }


def _submit_tool(model_type: type[ContractModel]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "submit_result",
            "description": (
                "Submit the final result. The arguments must match "
                "response_json_schema exactly."
            ),
            "parameters": model_type.model_json_schema(),
        },
    }


def _implementor_tools() -> list[dict[str, object]]:
    """OpenAI function-calling tool definitions for the implementor role."""
    return [
        _READ_FILE_TOOL,
        _WRITE_FILE_TOOL,
        _RUN_CHECK_TOOL,
        _DIFF_TOOL,
        _submit_tool(ImplementationSubmission),
    ]


def _validator_tools() -> list[dict[str, object]]:
    """Read-only tool definitions for the validator role."""
    return [
        _READ_FILE_TOOL,
        _VALIDATOR_CHECK_TOOL,
        _DIFF_TOOL,
        _submit_tool(ValidationReport),
    ]


class RoleSpecificAgentClient:
    """Typed adapter for one role, including one bounded schema repair."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        role: AgentRole,
        prompt: str,
        capabilities: tuple[str, ...] = (),
        scoped_repository: ScopedRepository | None = None,
    ) -> None:
        self._client = client
        self.role = role
        self.prompt = prompt
        self.capabilities = capabilities
        self.scoped_repository = scoped_repository

    def bind_worktree(self, path: Path, allowed_scopes: tuple[str, ...]) -> None:
        if self.role in {AgentRole.IMPLEMENTOR, AgentRole.VALIDATOR}:
            self.scoped_repository = ScopedWorktreeRepository(path, allowed_scopes)

    async def invoke(self, request: ContractModel) -> ContractModel:
        request_id = str(getattr(request, "request_id", ""))
        model_type: type[ContractModel]
        if self.role == AgentRole.ORCHESTRATION:
            model_type = OrchestrationDecision
        elif self.role == AgentRole.RESEARCH:
            model_type = (
                ExperimentProposalDecision
                if isinstance(request, ResearchRequest)
                and request.objective.startswith("propose next experiment")
                else ResearchDecision
            )
        elif self.role == AgentRole.IMPLEMENTOR:
            model_type = ImplementationSubmission
        elif self.role == AgentRole.VALIDATOR:
            model_type = ValidationReport
        else:  # pragma: no cover - AgentRole is exhaustive
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="capability",
                message="unsupported agent role",
                repair_attempts=0,
            )
        if self.role == AgentRole.IMPLEMENTOR and isinstance(
            self.scoped_repository, ScopedWorktreeRepository
        ):
            result = await self._invoke_implementor_agentic(request_id, request)
        elif (
            self.role == AgentRole.VALIDATOR
            and isinstance(request, ValidationRequest)
            and request.stage == ValidationStage.IMPLEMENTATION
            and isinstance(self.scoped_repository, ScopedWorktreeRepository)
        ):
            result = await self._invoke_validator_agentic(request_id, request)
        else:
            result = await invoke_structured(
                self._client,
                self.role,
                request_id,
                model_type,
                self.prompt,
                request.model_dump(mode="json"),
            )
        if isinstance(result, ExperimentProposalDecision):
            result = ResearchDecision.model_validate(result.model_dump(mode="json"))
        if isinstance(result, ImplementationSubmission):
            result = ImplementationResult.model_validate(result.model_dump(mode="json"))
        if isinstance(result, AgentFailure):
            return result
        if isinstance(request, OrchestrationRequest) and isinstance(
            result, OrchestrationDecision
        ):
            if result.action not in request.allowed_actions:
                return AgentFailure(
                    request_id=request.request_id,
                    role=AgentRole.ORCHESTRATION,
                    kind="policy",
                    message=f"orchestration selected disallowed action: {result.action.value}",
                    repair_attempts=0,
                )
            if (
                result.target_experiment_id is not None
                and result.target_experiment_id != request.current_experiment_id
            ):
                return AgentFailure(
                    request_id=request.request_id,
                    role=AgentRole.ORCHESTRATION,
                    kind="policy",
                    message="orchestration selected an unauthorized experiment identity",
                    repair_attempts=0,
                )
        # The implementor agentic path writes files via tools; edits are applied
        # by the controller only for the single-shot (non-agentic) path.
        implementor_agentic = self.role == AgentRole.IMPLEMENTOR and isinstance(
            self.scoped_repository, ScopedWorktreeRepository
        )
        if (
            not implementor_agentic
            and isinstance(request, ImplementationRequest)
            and isinstance(result, ImplementationResult)
        ):
            repository = self.scoped_repository
            if repository is None:
                return AgentFailure(
                    request_id=request_id,
                    role=self.role,
                    kind="capability",
                    message="implementor worktree capability is not bound",
                    repair_attempts=0,
                )
            try:
                apply_edits = getattr(repository, "apply_edits", None)
                if apply_edits is None:
                    for edit in result.edits:
                        repository.write(edit.relative_path, edit.content)
                else:
                    apply_edits(result.edits)
                changed_files = result.changed_files
                changed_files_method = getattr(repository, "changed_files", None)
                if callable(changed_files_method):
                    changed_files = cast(
                        Callable[[], tuple[str, ...]], changed_files_method
                    )()
                if not changed_files or not repository.diff():
                    raise ValueError("implementor edits produced no diff")
            except (OSError, PermissionError, ValueError, subprocess.CalledProcessError) as error:
                return AgentFailure(
                    request_id=request_id,
                    role=self.role,
                    kind="policy",
                    message=str(error),
                    repair_attempts=0,
                )
            result = result.model_copy(update={"changed_files": tuple(changed_files)})
        if (
            isinstance(request, ResearchRequest)
            and isinstance(result, ResearchDecision)
            and result.request_id != request.request_id
        ):
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="schema",
                message="research request ID mismatch",
                repair_attempts=0,
            )
        if (
            isinstance(request, ImplementationRequest)
            and isinstance(result, ImplementationResult)
            and result.experiment_id != request.experiment_id
        ):
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="policy",
                message="implementation experiment ID mismatch",
                repair_attempts=0,
            )
        if (
            isinstance(request, ValidationRequest)
            and isinstance(result, ValidationReport)
            and (result.experiment_id != request.experiment_id or result.stage != request.stage)
        ):
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="policy",
                message="validation subject identity mismatch",
                repair_attempts=0,
            )
        return result

    async def _invoke_implementor_agentic(
        self, request_id: str, request: ContractModel
    ) -> ContractModel:
        """Multi-turn tool-use loop for the implementor role."""
        repository = self.scoped_repository
        if not isinstance(repository, ScopedWorktreeRepository):
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="capability",
                message="implementor worktree capability is not bound",
                repair_attempts=0,
            )
        tools = _implementor_tools()

        def _handle(tool_name: str, arguments: dict[str, object]) -> str:
            if tool_name == "read_file":
                path = str(arguments.get("path", ""))
                max_chars = int(str(arguments.get("max_characters", "20000")))
                return repository.read(path, max_chars)
            if tool_name == "write_file":
                path = str(arguments.get("path", ""))
                content = str(arguments.get("content", ""))
                repository.write(path, content)
                return f"written: {path}"
            if tool_name == "run_check":
                command_raw = arguments.get("command", [])
                timeout = int(str(arguments.get("timeout_seconds", "30")))
                if isinstance(command_raw, list):
                    command = tuple(str(c) for c in cast(list[object], command_raw))
                else:
                    command = (str(command_raw),)
                return repository.run_check(command, timeout)
            if tool_name == "diff":
                return repository.diff()
            return f"error: unknown tool {tool_name!r}"

        result = await invoke_agentic(
            self._client,
            self.role,
            request_id,
            ImplementationSubmission,
            self.prompt,
            request.model_dump(mode="json"),
            tools,
            _handle,
            terminal_tool="submit_result",
        )
        if isinstance(result, AgentFailure):
            return result
        # The model has already written files via tools; skip re-applying edits.
        # Verify a real diff exists.
        changed_files = repository.changed_files()
        if not changed_files or not repository.diff():
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="policy",
                message="implementor produced no real diff",
                repair_attempts=0,
            )
        return ImplementationResult(
            experiment_id=result.experiment_id,
            patch_artifact_id=result.patch_artifact_id,
            changed_files=tuple(changed_files),
            edits=result.edits,
            changed_symbols=result.changed_symbols,
            checks=result.checks,
            assumptions=result.assumptions,
            unresolved_issues=result.unresolved_issues,
        )

    async def _invoke_validator_agentic(
        self, request_id: str, request: ContractModel
    ) -> ContractModel:
        """Multi-turn read-only verification loop for implementation validation."""
        repository = self.scoped_repository
        if not isinstance(repository, ScopedWorktreeRepository):
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="capability",
                message="validator worktree capability is not bound",
                repair_attempts=0,
            )

        check_commands = _validator_check_commands()
        check_results: dict[str, str] = {}
        failed_checks: list[str] = []
        for check, command in check_commands.items():
            try:
                check_results[check] = repository.run_check(command, 30)[-4_000:]
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                check_results[check] = str(error)[-4_000:]
                failed_checks.append(check)

        def _handle(tool_name: str, arguments: dict[str, object]) -> str:
            if tool_name == "read_file":
                path = str(arguments.get("path", ""))
                max_chars = int(str(arguments.get("max_characters", "20000")))
                return repository.read(path, max_chars)
            if tool_name == "run_check":
                check = str(arguments.get("check", ""))
                timeout = int(str(arguments.get("timeout_seconds", "30")))
                command = check_commands.get(check)
                if command is None:
                    raise ValueError(f"unsupported validator check: {check}")
                return repository.run_check(command, timeout)
            if tool_name == "diff":
                return repository.diff()
            return f"error: unknown tool {tool_name!r}"

        subject = request.model_dump(mode="json")
        subject["controller_check_results"] = check_results
        result = await invoke_agentic(
            self._client,
            self.role,
            request_id,
            ValidationReport,
            self.prompt,
            subject,
            _validator_tools(),
            _handle,
            max_turns=8,
            terminal_tool="submit_result",
        )
        if not failed_checks or not isinstance(result, ValidationReport):
            return result
        blockers = result.blockers + tuple(
            f"controller-owned check failed: {check}: {check_results[check]}"
            for check in failed_checks
        )
        verdict = (
            ValidationVerdict.REPAIRABLE
            if result.verdict == ValidationVerdict.APPROVED
            else result.verdict
        )
        return result.model_copy(
            update={"verdict": verdict, "blockers": blockers}
        )


class OpenAICompatibleAgentClient(RoleSpecificAgentClient):
    """Backward-compatible research-role adapter used by older callers."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        role: AgentRole = AgentRole.RESEARCH,
        prompt: str = "Return one typed research response.",
        capabilities: tuple[str, ...] = (),
        scoped_repository: ScopedRepository | None = None,
    ) -> None:
        super().__init__(client, role, prompt, capabilities, scoped_repository)


class ScopedWorktreeRepository:
    """Least-privilege repository capability bound to one experiment worktree."""

    def __init__(self, root: Path, allowed_scopes: tuple[str, ...]) -> None:
        self.root = root.resolve()
        self.allowed_scopes = tuple(scope.rstrip("/") for scope in allowed_scopes)

    def _path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise PermissionError("path escapes the assigned worktree") from error
        if not any(
            relative == scope or relative.startswith(scope + "/") for scope in self.allowed_scopes
        ):
            raise PermissionError(
                f"path is outside the approved implementation scope: {relative}"
            )
        if relative.startswith("baseline/"):
            raise PermissionError("protected path cannot be modified")
        return path

    def read(self, relative_path: str, max_characters: int = 20_000) -> str:
        content = self._path(relative_path).read_text(encoding="utf-8")
        if len(content) > max_characters:
            raise ValueError(f"scoped source exceeds {max_characters} characters")
        return content

    def read_base(self, relative_path: str, max_characters: int = 20_000) -> str:
        path = self._path(relative_path)
        relative = path.relative_to(self.root).as_posix()
        content = subprocess.run(
            ("git", "show", f"HEAD:{relative}"),
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if len(content) > max_characters:
            raise ValueError(f"base source exceeds {max_characters} characters")
        return content

    def search(self, query: str, max_results: int = 20) -> tuple[str, ...]:
        matches: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(self.root).as_posix()
                if not any(
                    relative == scope or relative.startswith(scope + "/")
                    for scope in self.allowed_scopes
                ):
                    continue
                if query in path.read_text(encoding="utf-8"):
                    matches.append(relative)
            except (OSError, UnicodeDecodeError):
                continue
            if len(matches) >= max_results:
                break
        return tuple(matches)

    def write(self, relative_path: str, content: str) -> None:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def apply_edits(self, edits: tuple[ImplementationEdit, ...]) -> None:
        paths = tuple(self._path(edit.relative_path) for edit in edits)
        for edit, path in zip(edits, paths, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit.content, encoding="utf-8")

    def changed_files(self) -> tuple[str, ...]:
        output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        paths: list[str] = []
        for line in output:
            if len(line) < 4:
                continue
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            paths.append(path)
        return tuple(paths)

    def diff(self) -> str:
        tracked = subprocess.run(
            ["git", "diff", "--", *self.allowed_scopes],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        untracked_parts: list[str] = []
        tracked_files = self._tracked_files()
        for relative_path in self.changed_files():
            path = self._path(relative_path)
            if not path.is_file() or path.relative_to(self.root).as_posix() not in tracked_files:
                result = subprocess.run(
                    ["git", "diff", "--no-index", "--no-ext-diff", "/dev/null", str(path)],
                    cwd=self.root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                untracked_parts.append(result.stdout)
        return tracked + "".join(untracked_parts)

    def _tracked_files(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in subprocess.run(
                ["git", "ls-files", "--", *self.allowed_scopes],
                cwd=self.root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if path
        )

    def run_check(self, command: tuple[str, ...], timeout_seconds: int) -> str:
        if not command or any(not part for part in command):
            raise ValueError("check command must be a non-empty argument tuple")
        result = subprocess.run(
            list(command),
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise ValueError(
                f"check exited with status {result.returncode}: {output[-20_000:]}"
            )
        return output


# ---------------------------------------------------------------------------
# RepositoryRunStore — wraps ApplicationRepository as a RunStore
# ---------------------------------------------------------------------------


class RepositoryRunStore:
    """RunStore over ApplicationRepository with typed persistence.

    Uses the generic ``put_json``/``list_json`` for records that lack
    dedicated authority methods (evaluations, worktree assignments).
    Experiment, run, audit, source registration, and finalization go
    through the repository's typed authority methods.
    """

    def __init__(self, repository: ApplicationRepository) -> None:
        self._repo = repository
        # Compatibility/debug view; authority remains the repository below.
        self.persisted: list[tuple[str, str, int, dict[str, object]]] = []

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        """Persist a transition and audit event through one repository CAS."""
        json_updates = cast(dict[str, object], _jsonable(updates))
        self._repo.persist_transition(run_id, operation, state_version, json_updates)
        self.persisted.append((run_id, operation, state_version, updates))

    def load_transition(self, run_id: str, state_version: int) -> dict[str, object] | None:
        for item in self._repo.list_json("transition"):
            value = json.loads(item)
            if value["run_id"] == run_id and value["state_version"] == state_version:
                return value["updates"]
        return None

    # --- typed authority methods ---

    def put_experiment(
        self,
        spec: ExperimentSpec,
        status: str,
        run_id: str,
        transition_id: str,
        expected_predecessor: str | None = None,
        audit_event: ContractModel | None = None,
    ) -> None:
        self._repo.put_experiment(
            spec=spec,
            status=status,
            run_id=run_id,
            transition_id=transition_id,
            expected_predecessor=expected_predecessor,
            audit_event=audit_event,  # type: ignore[arg-type]
        )

    def put_run(
        self,
        record: RunRecord,
        transition_id: str,
        expected_predecessor: str | None = None,
    ) -> None:
        self._repo.put_run(
            run=record,
            transition_id=transition_id,
            expected_predecessor=expected_predecessor,
        )

    def put_audit_event(self, event: ContractModel) -> None:
        if isinstance(event, AuditEvent):
            self._repo.put_audit_event(event)

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None:
        return self._repo.get_source_registration(experiment_id)

    def persist_provisional_finalization(self, request: ContractModel) -> FinalizationRecord:
        from tiktok2026.contracts import ProvisionalFinalizationRequest

        return self._repo.persist_provisional_finalization(
            request
            if isinstance(request, ProvisionalFinalizationRequest)
            else ProvisionalFinalizationRequest(**request.model_dump())
        )

    def get_finalization(self, finalization_id: str) -> FinalizationRecord | None:
        return self._repo.get_finalization(finalization_id)

    # --- generic record methods ---

    def put_evaluation(self, result: EvaluationResult, provenance: ProvenanceRequest) -> None:
        self._repo.put_json(
            "evaluation",
            result.evaluation_id,
            json.dumps(
                {
                    **result.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                    "provenance": provenance.model_dump(mode="json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def put_failure(self, record: FailureRecord, run_id: str) -> None:
        payload = record.model_dump_json()
        existing = {
            json.loads(value).get("failure_id") for value in self._repo.list_json("failure")
        }
        self._repo.put_json("failure", record.failure_id, payload)
        if record.failure_id not in existing:
            self._repo.put_audit_event(
                AuditEvent(
                    event_id=f"failure-{record.failure_id}",
                    run_id=run_id,
                    experiment_id=record.experiment_id,
                    event_type="failure_persisted",
                    actor_type="controller",
                    actor_id="production-controller",
                    payload=record.model_dump(mode="json"),
                )
            )

    def put_worktree_assignment(self, assignment: WorktreeAssignment) -> None:
        self._repo.put_json(
            "worktree_assignment", assignment.experiment_id, assignment.model_dump_json()
        )

    def put_source_registration(self, registration: SourceRegistration) -> None:
        self._repo.put_source_registration(registration)

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        return self._repo.get_experiment(experiment_id)

    def get_experiment_registry(self, limit: int = 50) -> ExperimentRegistrySnapshot:
        experiments, total = self._repo.list_experiments(limit)
        evaluations_by_experiment: dict[str, list[EvaluationResult]] = {}
        for raw in self._repo.list_json("evaluation"):
            value = json.loads(raw)
            result = EvaluationResult.model_validate(value.get("result", value))
            evaluations_by_experiment.setdefault(result.experiment_id, []).append(result)
        entries = tuple(
            entry.model_copy(
                update={
                    "evaluation_ids": tuple(
                        sorted(
                            result.evaluation_id
                            for result in evaluations_by_experiment.get(
                                entry.experiment_id, ()
                            )
                        )
                    ),
                    "evaluator_sha256s": tuple(
                        sorted(
                            {
                                result.evaluator_sha256
                                for result in evaluations_by_experiment.get(
                                    entry.experiment_id, ()
                                )
                            }
                        )
                    ),
                }
            )
            for entry in experiments
        )
        payload = json.dumps(
            {
                "entries": [entry.model_dump(mode="json") for entry in entries],
                "total_experiments": total,
                "complete": total <= limit,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return ExperimentRegistrySnapshot(
            evidence_id=f"experiment-registry-{hashlib.sha256(payload.encode()).hexdigest()}",
            entries=entries,
            total_experiments=total,
            complete=total <= limit,
        )

    def put_execution_result(self, result: ExecutionResult) -> None:
        self._repo.put_json("execution", result.execution_id, result.model_dump_json())

    def get_execution_result(self, execution_id: str) -> ExecutionResult | None:
        for item in self._repo.list_json("execution"):
            result = ExecutionResult.model_validate_json(item)
            if result.execution_id == execution_id:
                return result
        return None

    def get_evaluation_result(self, evaluation_id: str) -> EvaluationResult | None:
        for result in self.list_evaluation_results():
            if result.evaluation_id == evaluation_id:
                return result
        return None

    def list_evaluation_results(self) -> tuple[EvaluationResult, ...]:
        results: list[EvaluationResult] = []
        for item in self._repo.list_json("evaluation"):
            value = json.loads(item)
            payload = value.get("result", value)
            results.append(EvaluationResult.model_validate(payload))
        return tuple(results)

    def list_baseline_calibrations(self) -> tuple[BaselineCalibrationRecord, ...]:
        return tuple(
            BaselineCalibrationRecord.model_validate_json(item)
            for item in self._repo.list_json("baseline_calibration")
        )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        return self._repo.get_artifact(artifact_id)

    def put_artifact(self, record: ArtifactRecord) -> None:
        self._repo.register_artifact(record)

    def put_evaluator_identity(self, identity: EvaluatorIdentity) -> None:
        self._repo.put_evaluator_identity(identity)
        self._repo.put_json("evaluator_identity", identity.evaluator_id, identity.model_dump_json())

    def get_evaluator_identity(self, evaluator_id: str) -> EvaluatorIdentity | None:
        # ApplicationRepository has no reader for this record; retain a typed
        # cache in the generic authority namespace for composition adapters.
        for item in self._repo.list_json("evaluator_identity"):
            identity = EvaluatorIdentity.model_validate_json(item)
            if identity.evaluator_id == evaluator_id:
                return identity
        return None

    def put_dataset_manifest_identity(self, identity: DatasetManifestIdentity) -> None:
        self._repo.put_json(
            "dataset_manifest_identity", identity.manifest_id, identity.model_dump_json()
        )

    def get_dataset_manifest_identity(self) -> DatasetManifestIdentity | None:
        values = self._repo.list_json("dataset_manifest_identity")
        return DatasetManifestIdentity.model_validate_json(values[0]) if values else None

    def get_prediction_artifact(self, artifact_id: str) -> PredictionArtifactRegistration | None:
        for item in self._repo.list_json("prediction_artifact"):
            registration = PredictionArtifactRegistration.model_validate_json(item)
            if registration.artifact_id == artifact_id:
                return registration
        return None

    def get_worktree_assignment(self, experiment_id: str) -> WorktreeAssignment | None:
        for record_json in self._repo.list_json("worktree_assignment"):
            record = WorktreeAssignment.model_validate_json(record_json)
            if record.experiment_id == experiment_id:
                return record
        return None

    def put_json(self, kind: str, record_id: str, payload_json: str) -> None:
        self._repo.put_json(kind, record_id, payload_json)

    def list_json(self, kind: str) -> tuple[str, ...]:
        return self._repo.list_json(kind)


def _jsonable(value: object) -> object:
    if isinstance(value, ContractModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _jsonable(item) for key, item in mapping.items()}
    if isinstance(value, (tuple, list)):
        sequence = cast(tuple[object, ...] | list[object], value)
        return [_jsonable(item) for item in sequence]
    if isinstance(value, Path):
        return str(value)
    return value


class RepositoryTransitionStore(RepositoryRunStore):
    """Named composition seam for durable controller transitions."""


class RepositoryFinalizationBundleService:
    """Materialize a checksum-addressed bundle from persisted authority records."""

    def __init__(self, repository: ApplicationRepository, runtime_root: Path) -> None:
        self.repository = repository
        self.runtime_root = runtime_root

    def create(self, request: FinalizationBundleRequest) -> ArtifactRecord:
        store = RepositoryRunStore(self.repository)
        evaluation = store.get_evaluation_result(request.evaluation_id)
        source = self.repository.get_source_registration(request.experiment_id)
        evaluator = store.get_evaluator_identity(request.evaluator_id)
        prediction = (
            self.repository.get_artifact(evaluation.prediction_artifact_id)
            if evaluation is not None and evaluation.prediction_artifact_id is not None
            else None
        )
        if evaluation is None or source is None or evaluator is None:
            raise ValueError("finalization bundle authority records are incomplete")
        if (
            evaluation.experiment_id != request.experiment_id
            or evaluation.checkpoint_id != request.checkpoint_id
            or evaluation.run_id != request.run_id
            or source.run_id != request.run_id
            or source.source_commit != request.source_commit
            or evaluation.evaluator_artifact_id != request.evaluator_id
            or evaluation.evaluator_sha256 != evaluator.evaluator_sha256
            or prediction is None
            or prediction.kind != "prediction"
            or prediction.run_id != request.run_id
            or prediction.experiment_id != request.experiment_id
            or prediction.sha256 != evaluation.prediction_sha256
        ):
            raise ValueError("finalization bundle provenance does not match")
        payload = json.dumps(
            {
                "schema_version": "1",
                "run_id": request.run_id,
                "experiment_id": request.experiment_id,
                "source_commit": request.source_commit,
                "checkpoint_id": request.checkpoint_id,
                "evaluation_id": request.evaluation_id,
                "evaluator_id": request.evaluator_id,
                "evaluation": evaluation.model_dump(mode="json"),
                "source": source.model_dump(mode="json"),
                "evaluator": evaluator.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        import hashlib

        digest = hashlib.sha256(payload).hexdigest()
        artifact_id = f"bundle-{digest}"
        destination = (
            self.runtime_root
            / "artifacts"
            / request.run_id
            / request.experiment_id
            / artifact_id
            / "bundle.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            run_id=request.run_id,
            experiment_id=request.experiment_id,
            kind="finalization_bundle",
            uri=destination.as_uri(),
            sha256=digest,
            size_bytes=len(payload),
            producer="controller",
            retention=ArtifactRetention.PROVENANCE,
        )
        self.repository.register_artifact(record)
        return record


class RepositoryFrontierService:
    """Persist evaluation cost/fidelity observations and apply plateau policy."""

    def __init__(
        self,
        repository: ApplicationRepository,
        *,
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> None:
        if epsilon < 0 or patience < 1:
            raise ValueError("invalid plateau policy")
        self.repository = repository
        self.epsilon = epsilon
        self.patience = patience

    def initialize(self, run_id: str) -> None:
        self.repository.put_json(
            "frontier_policy",
            f"policy-{run_id}",
            json.dumps(
                {
                    "run_id": run_id,
                    "epsilon": self.epsilon,
                    "patience": self.patience,
                    "decision": "continue",
                    "reason": "awaiting persisted evaluation evidence",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def update(self, experiment_id: str, score: float) -> str | None:
        from tiktok2026.contracts import ExecutionResult

        raw_evaluations = [
            EvaluationResult.model_validate(json.loads(raw).get("result", json.loads(raw)))
            for raw in self.repository.list_json("evaluation")
        ]
        run_id = next(
            (result.run_id for result in raw_evaluations if result.experiment_id == experiment_id),
            None,
        )
        if run_id is None:
            return None
        evaluations: list[float] = []
        for result in raw_evaluations:
            if result.run_id != run_id:
                continue
            spec = self.repository.get_experiment(result.experiment_id)
            execution = None
            if result.execution_id:
                for execution_raw in self.repository.list_json("execution"):
                    candidate = ExecutionResult.model_validate_json(execution_raw)
                    if candidate.execution_id == result.execution_id:
                        execution = candidate
                        break
            if execution is None or spec is None:
                continue
            observed_score = result.validation_score
            self.repository.put_json(
                "frontier_observation",
                result.evaluation_id,
                json.dumps(
                    {
                        "evaluation_id": result.evaluation_id,
                        "run_id": result.run_id,
                        "experiment_id": result.experiment_id,
                        "score": observed_score,
                        "gpu_hours": execution.gpu_hours,
                        "fidelity": spec.fidelity.value,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            evaluations.append(observed_score)
        if not evaluations or abs(evaluations[-1] - score) > self.epsilon:
            return None
        policy = [
            json.loads(raw)
            for raw in self.repository.list_json("frontier_policy")
            if json.loads(raw).get("run_id") == run_id
        ]
        if not policy or policy[-1].get("decision") not in {"continue", "converge"}:
            return None
        configured_epsilon = float(policy[-1].get("epsilon", self.epsilon))
        configured_patience = int(policy[-1].get("patience", self.patience))
        reason = convergence_reason(evaluations, configured_epsilon, configured_patience)
        decision = "converge" if reason is not None else "continue"
        self.repository.put_json(
            "frontier_decision",
            f"{run_id}-{experiment_id}-{len(evaluations)}",
            json.dumps(
                {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "observation_count": len(evaluations),
                    "decision": decision,
                    "reason": reason or "configured policy requires more evidence",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return reason


# ---------------------------------------------------------------------------
# DeterministicPolicyGate — wraps pure policy functions
# ---------------------------------------------------------------------------


class DeterministicPolicyGate:
    """PolicyGate backed by the pure policy functions in ``policies/``."""

    def check_paths(
        self, changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
    ) -> PolicyDecisionModel:
        decision = check_changed_paths(changed_paths, allowed_scopes)
        return PolicyDecisionModel(allowed=decision.allowed, reason=decision.reason)

    def can_repair(self, repair_attempts: int) -> PolicyDecisionModel:
        decision = can_repair(repair_attempts)
        return PolicyDecisionModel(allowed=decision.allowed, reason=decision.reason)


# ---------------------------------------------------------------------------
# LedgerResourceAccountant — wraps ResourceLedger
# ---------------------------------------------------------------------------


class LedgerResourceAccountant:
    """ResourceAccountant backed by ResourceLedger."""

    def __init__(self, ledger: ResourceLedger) -> None:
        self._ledger = ledger

    def state(self) -> ResourceState:
        return self._ledger.state()

    def reserve(self, reservation: ContractModel) -> bool:
        if isinstance(reservation, ResourceReservation):
            return self._ledger.reserve(reservation)
        return False

    def consume(self, reservation_id: str, **usage: float | int) -> bool:
        return self._ledger.consume(
            reservation_id,
            gpu_hours=usage.get("gpu_hours"),  # type: ignore[arg-type]
            wall_seconds=usage.get("wall_seconds"),  # type: ignore[arg-type]
            tokens=usage.get("tokens"),  # type: ignore[arg-type]
            disk_bytes=usage.get("disk_bytes"),  # type: ignore[arg-type]
        )

    def reconcile(self, reservation_id: str, **usage: float | int) -> bool:
        return self._ledger.reconcile(
            reservation_id,
            gpu_hours=usage.get("gpu_hours"),
            wall_seconds=usage.get("wall_seconds"),
            tokens=int(usage["tokens"]) if "tokens" in usage else None,
            disk_bytes=int(usage["disk_bytes"]) if "disk_bytes" in usage else None,
        )


# ---------------------------------------------------------------------------
# RepositoryExportService — wraps export_records
# ---------------------------------------------------------------------------


class RepositoryExportService:
    """ExportService that reconstructs records from ApplicationRepository
    and writes deterministic JSONL + Markdown exports."""

    def __init__(self, repository: ApplicationRepository, runtime_root: Path) -> None:
        self._repo = repository
        self._runtime_root = runtime_root

    async def export_run(self, run_id: str, output_dir: Path | None = None) -> dict[str, Path]:
        events = self._repo.list_audit_events(run_id)
        records = tuple(event.model_dump(mode="json") for event in events)
        dest = output_dir or self._runtime_root / "exports" / run_id
        jsonl_path, md_path = export_records(run_id, records, dest)
        return {"jsonl": jsonl_path, "markdown": md_path}
