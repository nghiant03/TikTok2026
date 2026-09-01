from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, TypeVar, cast

from tiktok2026.agents.common.client import OpenAICompatibleClient
from tiktok2026.agents.common.structured import invoke_agentic, invoke_structured
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    AuditEvent,
    BaselineCalibrationRecord,
    BlockerResolution,
    ChampionBinding,
    ContractModel,
    DatasetManifestIdentity,
    DecisionAction,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionResult,
    ExperimentProposalDecision,
    ExperimentRegistrySnapshot,
    ExperimentSpec,
    FailureRecord,
    FinalizationBundleRequest,
    FinalizationRecord,
    FullAttemptClaimRequest,
    FullScientificAttemptClaim,
    ImplementationEdit,
    ImplementationRequest,
    ImplementationResult,
    ImplementationSubmission,
    OnlineResearchProvider,
    OnlineSearchRequest,
    OrchestrationDecision,
    OrchestrationRequest,
    PolicyDecisionModel,
    PredictionArtifactRegistration,
    ProvenanceRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceReservation,
    ResourceState,
    RunBaselineBinding,
    RunClosure,
    RunRecord,
    ScopedRepository,
    ScoredObservation,
    ScoredObservationRequest,
    SourceRegistration,
    ValidationBlocker,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    ValidationVerdict,
    WorktreeAssignment,
    validation_blocker_id,
)
from tiktok2026.observability.exports import export_records
from tiktok2026.persistence.repositories import ApplicationRepository, PersistenceConflictError
from tiktok2026.persistence.resources import ResourceLedger
from tiktok2026.policies.lifecycle import can_repair, convergence_reason
from tiktok2026.policies.paths import check_changed_paths

ModelT = TypeVar("ModelT", bound=ContractModel)

MAX_FAILURE_ID_LENGTH = 256
MAX_FAILURE_RUN_ID_LENGTH = 256
MAX_FAILURE_EXPERIMENT_ID_LENGTH = 256
MAX_FAILURE_EVIDENCE_REFS = 8
MAX_FAILURE_EVIDENCE_REF_LENGTH = 256
MAX_FAILURE_REPAIR_ATTEMPT = 3


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
    "Run one controller-owned, non-mutating implementation check.",
    {
        "check": {
            "type": "string",
            "enum": [
                "compile_entrypoint",
                "ruff_entrypoint",
                "pyright_entrypoint",
                "diff_check",
                "contract_smoke",
            ],
        },
    },
    ("check",),
)

_VALIDATOR_CHECK_TOOL = _tool(
    "run_check",
    "Run one controller-owned, non-mutating implementation check.",
    {
        "check": {
            "type": "string",
            "enum": [
                "compile_entrypoint",
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
    {
        "max_characters": {
            "type": "integer",
            "description": "Maximum characters to return (default 20000)",
            "default": 20_000,
        }
    },
)

_SEARCH_ONLINE_TOOL = _tool(
    "search_online",
    (
        "Search current public sources for scientific evidence. Never include repository "
        "source, credentials, private paths, dataset rows, or test information in the query."
    ),
    {
        "query": {"type": "string", "minLength": 3, "maxLength": 500},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
    },
    ("query",),
)



def _static_contract_check_command() -> tuple[str, ...]:
    entrypoint = "src/tiktok2026/experiment/train.py"
    code = (
        "import sys; "
        "from pathlib import Path; "
        "sys.path.insert(0, 'src'); "
        "from tiktok2026.policies.implementation import check_static_training_contract; "
        f"diagnostic=check_static_training_contract("
        f"Path({entrypoint!r}).read_text(encoding='utf-8')); "
        "sys.exit(diagnostic) if diagnostic else print('static contract check passed')"
    )
    return ("python", "-c", code)

def _implementation_check_commands() -> dict[str, tuple[str, ...]]:
    entrypoint = "src/tiktok2026/experiment/train.py"
    commands: dict[str, tuple[str, ...]] = {
        "compile_entrypoint": (
            "python",
            "-c",
            (
                "from pathlib import Path; "
                f"source=Path('{entrypoint}').read_text(); "
                f"compile(source, '{entrypoint}', 'exec')"
            ),
        ),
        "ruff_entrypoint": ("ruff", "check", entrypoint),
        "pyright_entrypoint": ("pyright", entrypoint),
    }
    commands["diff_check"] = ("git", "diff", "--check", "--", entrypoint)
    # Compatibility name retained for callers, but this command only parses
    # candidate source with a trusted AST checker.  It never imports train.py.
    commands["contract_smoke"] = _static_contract_check_command()
    return commands


IMPLEMENTOR_CHECK_NAMES = tuple(_implementation_check_commands())


def _validator_check_commands() -> dict[str, tuple[str, ...]]:
    commands = _implementation_check_commands()
    commands.pop("diff_check")
    commands.pop("contract_smoke")
    return commands


def _implementation_submission_failure(
    repository: ScopedWorktreeRepository,
    check_commands: Mapping[str, tuple[str, ...]],
) -> str | None:
    """Return a controller-owned submission diagnostic, if one exists."""
    try:
        changed_files = repository.changed_files()
        if not changed_files or not repository.diff():
            return "implementor produced no real diff"
        for check in IMPLEMENTOR_CHECK_NAMES:
            command = check_commands[check]
            try:
                repository.run_check(command, 30)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                label = "static contract check" if check == "contract_smoke" else check
                return f"controller-owned check failed: {label}: {error}"
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return str(error)
    return None


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
        online_research: OnlineResearchProvider | None = None,
        max_online_searches: int = 3,
        max_online_results: int = 5,
        online_allowed_domains: tuple[str, ...] = (),
    ) -> None:
        self._client = client
        self.role = role
        self.prompt = prompt
        self.capabilities = capabilities
        self.scoped_repository = scoped_repository
        self.online_research = online_research
        self.max_online_searches = max_online_searches
        self.max_online_results = max_online_results
        self.online_allowed_domains = online_allowed_domains

    def bind_worktree(
        self,
        path: Path,
        allowed_scopes: tuple[str, ...],
        read_scopes: tuple[str, ...] | None = None,
    ) -> None:
        if self.role == AgentRole.IMPLEMENTOR:
            self.scoped_repository = ScopedWorktreeRepository(
                path, allowed_scopes, read_scopes=read_scopes
            )
        elif self.role == AgentRole.VALIDATOR:
            # A validator may inspect the assigned source, but never receives a
            # writable repository capability, even when an old caller passes
            # implementation scopes to this method.
            self.scoped_repository = ScopedWorktreeRepository(
                path,
                read_scopes=self._merge_scopes(allowed_scopes, read_scopes or ()),
                write_scopes=(),
                inspection_scopes=allowed_scopes,
            )

    @staticmethod
    def _merge_scopes(*scope_sets: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(scope for scopes in scope_sets for scope in scopes))

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
        if (
            self.role == AgentRole.RESEARCH
            and isinstance(request, ResearchRequest)
            and self.online_research is not None
        ):
            result = await self._invoke_research_agentic(request, model_type)
        elif self.role == AgentRole.IMPLEMENTOR and isinstance(
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
            pending_targets: set[str] = {
                proposal.experiment_id
                for proposal in (
                    request.experiment_history.pending_proposals
                    if request.experiment_history is not None
                    else ()
                )
            }
            if result.action in (DecisionAction.RESEARCH, DecisionAction.STOP):
                target_is_invalid = result.target_experiment_id is not None
            elif result.action == DecisionAction.IMPLEMENT:
                target_is_invalid = result.target_experiment_id not in pending_targets
            else:
                target_is_invalid = result.target_experiment_id != request.current_experiment_id
            if target_is_invalid:
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
        if isinstance(request, ResearchRequest) and isinstance(result, ResearchDecision):
            spec = result.experiment_spec
            if spec is not None and not set(spec.implementation_scope).issubset(
                set(request.allowed_paths)
            ):
                return AgentFailure(
                    request_id=request_id,
                    role=self.role,
                    kind="policy",
                    message="experiment scope is not authorized",
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

    async def _invoke_research_agentic(
        self, request: ResearchRequest, model_type: type[ContractModel]
    ) -> ContractModel:
        provider = self.online_research
        if provider is None:
            raise AssertionError("online research provider is not bound")
        evidence_ids: set[str] = set()
        if request.source_context is not None:
            evidence_ids.add(request.source_context.evidence_id)
        if request.experiment_history is not None:
            evidence_ids.add(request.experiment_history.evidence_id)
        if (
            request.controller_context is not None
            and request.controller_context.experiment_registry is not None
        ):
            evidence_ids.add(request.controller_context.experiment_registry.evidence_id)
        online_ids: set[str] = set()
        searches = 0

        async def _handle(tool_name: str, arguments: dict[str, object]) -> str:
            nonlocal searches
            if tool_name != "search_online":
                raise ValueError(f"unsupported research tool: {tool_name}")
            if searches >= self.max_online_searches:
                raise ValueError("online research search budget exhausted")
            searches += 1
            requested_limit = int(str(arguments.get("max_results", self.max_online_results)))
            search_request = OnlineSearchRequest(
                request_id=f"{request.request_id}-online-{searches}",
                query=str(arguments.get("query", "")),
                max_results=min(requested_limit, self.max_online_results),
                allowed_domains=self.online_allowed_domains,
            )
            result = await provider.search(search_request)
            for source in result.sources:
                if source.source_id in evidence_ids:
                    raise ValueError(f"duplicate research evidence ID: {source.source_id}")
                evidence_ids.add(source.source_id)
                online_ids.add(source.source_id)
            return result.model_dump_json()

        def _guard(result: ContractModel) -> str | None:
            if not isinstance(result, (ResearchDecision, ExperimentProposalDecision)):
                return "research submitted an invalid result type"
            references = set(result.evidence_refs)
            if result.experiment_spec is not None:
                references.update(result.experiment_spec.evidence_refs)
            unknown = references - evidence_ids
            if unknown:
                return f"research cites unknown evidence IDs: {tuple(sorted(unknown))}"
            if online_ids and references.isdisjoint(online_ids):
                return "online search results were used but no online evidence ID was cited"
            if result.request_id != request.request_id:
                return "research request ID mismatch"
            return None

        return await invoke_agentic(
            self._client,
            self.role,
            request.request_id,
            model_type,
            self.prompt,
            request.model_dump(mode="json"),
            (_SEARCH_ONLINE_TOOL, _submit_tool(model_type)),
            _handle,
            max_turns=self.max_online_searches + 3,
            terminal_tool="submit_result",
            terminal_guard=_guard,
        )

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
        check_commands = _implementation_check_commands()

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
                check = str(arguments.get("check", ""))
                command = check_commands.get(check)
                if command is None:
                    raise ValueError(f"unsupported implementor check: {check}")
                return repository.run_check(command, 30)
            if tool_name == "diff":
                max_chars = int(str(arguments.get("max_characters", "20000")))
                return repository.diff(max_chars)
            return f"error: unknown tool {tool_name!r}"

        def _guard(_submission: ImplementationSubmission) -> str | None:
            return _implementation_submission_failure(repository, check_commands)

        result = await invoke_agentic(
            self._client,
            self.role,
            request_id,
            ImplementationSubmission,
            self.prompt,
            request.model_dump(mode="json"),
            tools,
            _handle,
            max_turns=32,
            terminal_tool="submit_result",
            terminal_guard=_guard,
        )
        if isinstance(result, AgentFailure):
            return result
        # The model has already written files via tools; skip re-applying edits.
        # Re-check after the loop: terminal guards are useful for repair, but
        # the controller must retain the final acceptance authority.
        diagnostic = _implementation_submission_failure(repository, check_commands)
        if diagnostic is not None:
            return AgentFailure(
                request_id=request_id,
                role=self.role,
                kind="policy",
                message=diagnostic,
                repair_attempts=0,
            )
        changed_files = repository.changed_files()
        return ImplementationResult(
            experiment_id=result.experiment_id,
            patch_artifact_id=result.patch_artifact_id,
            changed_files=tuple(changed_files),
            edits=result.edits,
            changed_symbols=result.changed_symbols,
            checks=IMPLEMENTOR_CHECK_NAMES,
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
                max_chars = int(str(arguments.get("max_characters", "20000")))
                return repository.diff(max_chars)
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
            ValidationBlocker(
                blocker_id=validation_blocker_id(
                    request_id,
                    result.stage,
                    f"controller-owned check failed: {check}: {check_results[check]}",
                ),
                experiment_id=result.experiment_id,
                stage=result.stage,
                text=f"controller-owned check failed: {check}: {check_results[check]}",
                report_id=result.report_id,
                evidence_refs=(f"controller-check-{check}",),
            )
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

    def __init__(
        self,
        root: Path,
        allowed_scopes: tuple[str, ...] = (),
        *,
        read_scopes: tuple[str, ...] | None = None,
        write_scopes: tuple[str, ...] | None = None,
        inspection_scopes: tuple[str, ...] | None = None,
    ) -> None:
        self.root = root.resolve()
        # ``allowed_scopes`` is the pre-split constructor argument.  Preserve
        # its meaning for existing callers while allowing the controller to
        # grant broader read access than write access.
        effective_write_scopes = allowed_scopes if write_scopes is None else write_scopes
        effective_read_scopes = (
            effective_write_scopes
            if read_scopes is None
            else (*effective_write_scopes, *read_scopes)
        )
        self.write_scopes = self._normalize_scopes(effective_write_scopes)
        self.read_scopes = self._normalize_scopes(effective_read_scopes)
        effective_inspection_scopes = (
            effective_write_scopes if inspection_scopes is None else inspection_scopes
        )
        self.inspection_scopes = self._normalize_scopes(effective_inspection_scopes)
        # Compatibility for callers that inspect the old public attribute.
        self.allowed_scopes = self.write_scopes

    @staticmethod
    def _normalize_scopes(scopes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(scope.rstrip("/") for scope in scopes)

    def _path(
        self, relative_path: str, scopes: tuple[str, ...], *, writable: bool = False
    ) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise PermissionError("path escapes the assigned worktree") from error
        if not any(relative == scope or relative.startswith(scope + "/") for scope in scopes):
            raise PermissionError(f"path is outside the approved repository scope: {relative}")
        if writable and relative.startswith("baseline/"):
            raise PermissionError("protected path cannot be modified")
        return path

    def _write_path(self, relative_path: str) -> Path:
        return self._path(relative_path, self.write_scopes, writable=True)

    def _read_path(self, relative_path: str) -> Path:
        return self._path(relative_path, self.read_scopes)

    def read(self, relative_path: str, max_characters: int = 20_000) -> str:
        content = self._read_path(relative_path).read_text(encoding="utf-8")
        if len(content) > max_characters:
            raise ValueError(f"scoped source exceeds {max_characters} characters")
        return content

    def read_base(self, relative_path: str, max_characters: int = 20_000) -> str:
        path = self._read_path(relative_path)
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
                    for scope in self.read_scopes
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
        path = self._write_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def apply_edits(self, edits: tuple[ImplementationEdit, ...]) -> None:
        paths = tuple(self._write_path(edit.relative_path) for edit in edits)
        for edit, path in zip(edits, paths, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit.content, encoding="utf-8")

    def changed_files(self) -> tuple[str, ...]:
        output = self._status_paths()
        paths: list[str] = []
        for path in output:
            if path.startswith("baseline/"):
                raise PermissionError("protected path cannot be modified")
            if not any(
                path == scope or path.startswith(scope + "/") for scope in self.write_scopes
            ):
                raise PermissionError(f"changed path is outside write scope: {path}")
            paths.append(path)
        return tuple(paths)

    def _status_paths(self) -> tuple[str, ...]:
        output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        paths: list[str] = []
        for line in output:
            if len(line) >= 4:
                path = line[3:]
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                paths.append(path)
        return tuple(paths)

    def diff(self, max_characters: int | None = None) -> str:
        """Return the authoritative diff, optionally bounded for an agent tool."""
        value = self._full_diff()
        if max_characters is not None:
            if max_characters < 0:
                raise ValueError("maximum diff characters cannot be negative")
            return value[:max_characters]
        return value

    def _full_diff(self) -> str:
        if not self.inspection_scopes:
            return ""
        tracked = subprocess.run(
            ["git", "diff", "--", *self.inspection_scopes],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        untracked_parts: list[str] = []
        tracked_files = self._tracked_files(self.inspection_scopes)
        for relative_path in self._status_paths():
            if not any(
                relative_path == scope
                or relative_path.startswith(scope + "/")
                for scope in self.inspection_scopes
            ):
                continue
            path = self._path(relative_path, self.inspection_scopes)
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

    def _tracked_files(self, scopes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            path
            for path in subprocess.run(
                ["git", "ls-files", "--", *scopes],
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

    def get_source_registration_by_id(
        self, registration_id: str
    ) -> SourceRegistration | None:
        return self._repo.get_source_registration_by_id(registration_id)

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

    def put_validation_report(
        self,
        report: ValidationReport,
        run_id: str,
        operation: ValidationOperationIdentity,
        subject: dict[str, object],
    ) -> None:
        self._repo.put_validation_report(report, run_id, operation, subject)

    def get_validation_report(self, report_id: str) -> ValidationReport | None:
        return self._repo.get_validation_report(report_id)

    def get_validation_report_by_operation(
        self, operation_id: str
    ) -> ValidationReport | None:
        return self._repo.get_validation_report_by_operation(operation_id)

    def get_validation_report_for_attempt(
        self, run_id: str, experiment_id: str, stage: ValidationStage, repair_attempt: int
    ) -> ValidationReport | None:
        return self._repo.get_validation_report_for_attempt(
            run_id, experiment_id, stage, repair_attempt
        )

    def get_validation_operation(
        self, operation_id: str
    ) -> ValidationOperationIdentity | None:
        return self._repo.get_validation_operation(operation_id)

    def list_validation_reports(
        self, experiment_id: str | None = None
    ) -> tuple[ValidationReport, ...]:
        return self._repo.list_validation_reports(experiment_id)

    def list_validation_blockers(
        self, experiment_id: str | None = None
    ) -> tuple[ValidationBlocker, ...]:
        return self._repo.list_validation_blockers(experiment_id)

    def get_validation_blocker(self, blocker_id: str) -> ValidationBlocker | None:
        return self._repo.get_validation_blocker(blocker_id)

    def put_blocker_resolution(self, resolution: BlockerResolution, run_id: str) -> None:
        self._repo.put_blocker_resolution(resolution, run_id)

    def list_blocker_resolutions(
        self, experiment_id: str | None = None
    ) -> tuple[BlockerResolution, ...]:
        return self._repo.list_blocker_resolutions(experiment_id)

    def get_blocker_resolution(self, resolution_id: str) -> BlockerResolution | None:
        return self._repo.get_blocker_resolution(resolution_id)

    def get_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        return self._repo.get_unresolved_blockers(experiment_id)

    def get_unresolved_blocker_ids(self, experiment_id: str) -> tuple[str, ...]:
        return self._repo.get_unresolved_blocker_ids(experiment_id)

    def list_unresolved_blockers(self, experiment_id: str) -> tuple[ValidationBlocker, ...]:
        return self._repo.list_unresolved_blockers(experiment_id)

    def put_failure(self, record: FailureRecord, run_id: str) -> None:
        if record.run_id is not None and record.run_id != run_id:
            raise ValueError("failure record run_id does not match its persistence run")
        get_json = getattr(self._repo, "get_json", None)
        if callable(get_json):
            existing_payload = cast(Callable[[str, str], str | None], get_json)(
                "failure", record.failure_id
            )
        else:
            # Compatibility fallback for older repository doubles.  Do not
            # validate unrelated rows while locating the requested identity.
            existing_payload = None
            for value in self._repo.list_json("failure"):
                try:
                    value_id = json.loads(value).get("failure_id")
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if value_id == record.failure_id:
                    existing_payload = value
                    break
        existing_record = (
            FailureRecord.model_validate_json(existing_payload)
            if existing_payload is not None
            else None
        )
        if existing_record is not None:
            if existing_record.run_id is not None and existing_record.run_id != run_id:
                raise ValueError("persisted failure record belongs to another run")
            if existing_record.run_id is None:
                # Legacy records are immutable and intentionally remain unbound.
                if existing_record != record.model_copy(update={"run_id": None}):
                    raise PersistenceConflictError(
                        f"failure {record.failure_id} legacy content changed"
                    )
                return
            if existing_record != record:
                raise PersistenceConflictError(
                    f"failure {record.failure_id} content changed"
                )
            return
        self._validate_new_failure_record(record, run_id)
        bound = record.model_copy(update={"run_id": run_id})
        payload = bound.model_dump_json()
        self._repo.put_json("failure", bound.failure_id, payload)
        self._repo.put_audit_event(
            AuditEvent(
                event_id=f"failure-{bound.failure_id}",
                run_id=run_id,
                experiment_id=bound.experiment_id,
                event_type="failure_persisted",
                actor_type="controller",
                actor_id="production-controller",
                payload=bound.model_dump(mode="json"),
            )
        )

    @staticmethod
    def _validate_new_failure_record(record: FailureRecord, run_id: str) -> None:
        """Apply current write limits without narrowing the v1 read contract."""
        values: list[tuple[str, str, int]] = [
            ("failure_id", record.failure_id, MAX_FAILURE_ID_LENGTH),
            ("run_id", run_id, MAX_FAILURE_RUN_ID_LENGTH),
        ]
        if record.experiment_id is not None:
            values.append(
                ("experiment_id", record.experiment_id, MAX_FAILURE_EXPERIMENT_ID_LENGTH)
            )
        for name, value, limit in values:
            if not value or len(value) > limit:
                raise ValueError(f"new failure {name} exceeds its write boundary")
        if len(record.evidence_refs) > MAX_FAILURE_EVIDENCE_REFS:
            raise ValueError("new failure evidence_refs exceeds its write boundary")
        if any(
            not reference or len(reference) > MAX_FAILURE_EVIDENCE_REF_LENGTH
            for reference in record.evidence_refs
        ):
            raise ValueError("new failure evidence reference exceeds its write boundary")
        if not 0 <= record.repair_attempt <= MAX_FAILURE_REPAIR_ATTEMPT:
            raise ValueError("new failure repair attempt exceeds its write boundary")

    def list_failure_records(self, run_id: str) -> tuple[FailureRecord, ...]:
        """Read only failures explicitly bound to one run."""
        records: list[FailureRecord] = []
        for payload in self._repo.list_json("failure"):
            try:
                record = FailureRecord.model_validate_json(payload)
            except ValueError:
                # A malformed historical record must not hide valid run-local
                # failures from replay and planning.
                continue
            if record.run_id == run_id:
                records.append(record)
        return tuple(records)

    def put_worktree_assignment(self, assignment: WorktreeAssignment) -> None:
        self._repo.put_json(
            "worktree_assignment", assignment.experiment_id, assignment.model_dump_json()
        )

    def put_source_registration(self, registration: SourceRegistration) -> None:
        self._repo.put_source_registration(registration)

    def get_experiment(self, experiment_id: str) -> ExperimentSpec | None:
        return self._repo.get_experiment(experiment_id)

    def list_experiments_by_status(
        self, run_id: str, status: str
    ) -> tuple[ExperimentSpec, ...]:
        """Expose the existing run-local proposal ledger to planning contexts."""
        return self._repo.list_experiments_by_status(run_id, status)

    def get_experiment_registry(
        self, limit: int = 50, exclude_experiment_id: str | None = None
    ) -> ExperimentRegistrySnapshot:
        experiments, total = self._repo.list_experiments(limit, exclude_experiment_id)
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
        return self._repo.list_baseline_calibrations()

    def get_baseline_calibration(self, calibration_id: str) -> BaselineCalibrationRecord | None:
        return self._repo.get_baseline_calibration(calibration_id)

    def put_baseline_calibration(
        self,
        record: BaselineCalibrationRecord,
        actor_type: Literal["agent", "controller", "human"],
        actor_id: str,
    ) -> None:
        self._repo.put_baseline_calibration(record, actor_type, actor_id)

    def put_run_baseline(self, binding: RunBaselineBinding) -> None:
        calibration = self.get_baseline_calibration(binding.calibration_id)
        if calibration is None:
            raise ValueError("run baseline binding references an unknown calibration")
        evaluation = calibration.evaluation
        calibration_metrics = {metric.name: metric.value for metric in evaluation.metrics}
        binding_metrics = {metric.name: metric.value for metric in binding.metrics}
        if (
            evaluation.evaluation_id != binding.baseline_evaluation_id
            or calibration.dataset_manifest_id != binding.dataset_manifest_id
            or calibration.dataset_manifest_sha256 != binding.dataset_manifest_sha256
            or calibration.evaluator_id != binding.evaluator_id
            or calibration.evaluator_sha256 != binding.evaluator_sha256
            or calibration.split != binding.split
            or calibration_metrics != binding_metrics
        ):
            raise ValueError("run baseline binding does not match its calibration")
        self._repo.put_run_baseline(binding)

    def get_run_baseline(self, run_id: str) -> RunBaselineBinding | None:
        return self._repo.get_run_baseline(run_id)

    def claim_full_attempt(
        self, request: FullAttemptClaimRequest
    ) -> FullScientificAttemptClaim | None:
        return self._repo.claim_full_attempt(request)

    def list_full_attempt_claims(
        self, run_id: str | None = None
    ) -> tuple[FullScientificAttemptClaim, ...]:
        return self._repo.list_full_attempt_claims(run_id)

    def count_full_attempt_claims(self, run_id: str) -> int:
        return self._repo.count_full_attempt_claims(run_id)

    def put_scored_observation(self, request: ScoredObservationRequest) -> ScoredObservation:
        return self._repo.put_scored_observation(request)

    def get_scored_observation(self, observation_id: str) -> ScoredObservation | None:
        return self._repo.get_scored_observation(observation_id)

    def list_scored_observations(
        self, run_id: str | None = None
    ) -> tuple[ScoredObservation, ...]:
        return self._repo.list_scored_observations(run_id)

    def close_run(
        self,
        run_id: str,
        reason: Literal["plateau", "attempt_cap"],
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> RunClosure:
        return self._repo.close_run(run_id, reason, epsilon, patience)

    def get_run_closure(self, run_id: str) -> RunClosure | None:
        return self._repo.get_run_closure(run_id)

    def get_run_champion(self, run_id: str) -> ChampionBinding | None:
        return self._repo.get_run_champion(run_id)

    def close_run_if_ready(
        self,
        run_id: str,
        *,
        after_failure: bool = False,
        epsilon: float = 0.002,
        patience: int = 3,
    ) -> RunClosure | None:
        return self._repo.close_run_if_ready(
            run_id,
            after_failure=after_failure,
            epsilon=epsilon,
            patience=patience,
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
        source = self.repository.get_source_registration_by_id(f"source-{request.source_commit}")
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
        current_metric_pair = frozenset(("GAUC", "nDCG@5"))
        current = next(
            (
                result
                for result in reversed(raw_evaluations)
                if result.experiment_id == experiment_id
                and result.validation_score == score
                and frozenset(metric.name for metric in result.metrics) == current_metric_pair
                and result.run_id is not None
                and result.dataset_manifest_sha256 is not None
                and result.split is not None
            ),
            None,
        )
        if current is None:
            return None
        run_id = current.run_id
        evaluations: list[float] = []
        for result in raw_evaluations:
            if (
                result.run_id != run_id
                or result.dataset_manifest_sha256 != current.dataset_manifest_sha256
                or result.split != current.split
                or result.evaluator_artifact_id != current.evaluator_artifact_id
                or result.evaluator_sha256 != current.evaluator_sha256
                or result.validity != current.validity
                or frozenset(metric.name for metric in result.metrics) != current_metric_pair
            ):
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

    def release(self, reservation_id: str) -> bool:
        return self._ledger.release(reservation_id)


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
