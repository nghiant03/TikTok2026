from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, cast

from loguru import logger

from tiktok2026.contracts import (
    AgentClient,
    AgentFailure,
    AgentRole,
    ControllerContext,
    DatasetViewProvenance,
    DecisionAction,
    EvaluationContext,
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
    ExecutionRequest,
    Executor,
    ExperimentSpec,
    ExportService,
    FailureKind,
    FailureRecord,
    FinalizationBundleRequest,
    FinalizationBundleService,
    FrontierService,
    ImplementationAttemptRecord,
    ImplementationRequest,
    ImplementationResult,
    ImplementationValidationAuthority,
    OrchestrationDecision,
    OrchestrationRequest,
    PolicyGate,
    ProvenanceRequest,
    ProvisionalFinalizationRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceAccountant,
    ResourceReservation,
    ResourceState,
    RunPhase,
    RunRecord,
    RunStore,
    ValidationBlocker,
    ValidationBlockerContext,
    ValidationOperationIdentity,
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    WorktreeManager,
    validation_blocker_id,
)
from tiktok2026.controller import Transition
from tiktok2026.graph.routes import (
    route_after_failure,
    route_after_orchestration,
    route_after_validation,
)
from tiktok2026.graph.state import ProductionState
from tiktok2026.policies.resources import check_smoke_feasibility


class MissingAuthorityError(RuntimeError):
    """A transition cannot proceed without a persisted authority record."""

    terminal = True


class TerminalLifecycleError(RuntimeError):
    """A typed terminal failure that must not continue to export or complete."""

    terminal = True


class ModelUnavailableError(RuntimeError):
    """A model call failed before an authoritative agent judgment was produced."""


def _agent_failure(state: ProductionState, failure: AgentFailure) -> dict[str, object]:
    if failure.kind == "model":
        raise ModelUnavailableError(failure.message)
    return _failure(state, FailureKind.SCHEMA_MISMATCH, failure.message)


IMPLEMENTATION_ROOTS = ("src/tiktok2026/experiment",)
EXPERIMENT_ENTRYPOINT = "src/tiktok2026/experiment/train.py"
MAX_TERMINAL_TEXT = 2_000
MAX_EVIDENCE_REFS = 8
MAX_EVIDENCE_REF_LENGTH = 256


def _agent(s: ServiceTransitions, role: AgentRole) -> AgentClient | None:
    return s.agent_clients.get(role) or s.agent_client


def _exp_id(state: ProductionState) -> str:
    value = state.get("current_experiment_id")
    if not value:
        raise MissingAuthorityError("current experiment identity is absent")
    return value


def _failure(
    state: ProductionState,
    kind: FailureKind,
    message: str,
    evidence: tuple[str, ...] = (),
) -> dict[str, object]:
    # The bounded graph state carries only an ID-sized summary.  Persist_failure
    # turns this into the typed FailureRecord exactly once.
    bounded_message = message[:MAX_TERMINAL_TEXT]
    bounded_evidence = tuple(
        item[:MAX_EVIDENCE_REF_LENGTH] for item in evidence[:MAX_EVIDENCE_REFS]
    )
    detail = json.dumps(
        {"kind": kind.value, "message": bounded_message, "evidence": bounded_evidence},
        sort_keys=True,
    )
    return {"terminal_reason": f"failure:{detail}", "pending_route": "persist_failure"}


def _blocker_context(
    blockers: tuple[ValidationBlocker, ...],
) -> tuple[ValidationBlockerContext, ...]:
    return tuple(
        ValidationBlockerContext(
            blocker_id=blocker.blocker_id,
            text=blocker.text[:MAX_TERMINAL_TEXT],
            evidence_refs=tuple(
                ref[:MAX_EVIDENCE_REF_LENGTH]
                for ref in blocker.evidence_refs[:MAX_EVIDENCE_REFS]
            ),
        )
        for blocker in blockers
    )


def _failure_details(state: ProductionState) -> tuple[FailureKind, str, tuple[str, ...]]:
    reason = state.get("terminal_reason") or ""
    if reason.startswith("failure:"):
        try:
            payload = json.loads(reason.removeprefix("failure:"))
            return FailureKind(payload["kind"]), str(payload["message"]), tuple(payload["evidence"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return FailureKind.SCHEMA_MISMATCH, "failure classification was absent", ()


def _validation_operation(
    state: ProductionState, stage: ValidationStage, subject: dict[str, object]
) -> ValidationOperationIdentity:
    subject_json = json.dumps(subject, sort_keys=True, separators=(",", ":"))
    subject_sha256 = hashlib.sha256(subject_json.encode()).hexdigest()
    authority = subject.get("implementation_authority")
    diff_sha256 = (
        str(cast(dict[str, object], authority)["diff_sha256"])
        if isinstance(authority, dict) and "diff_sha256" in authority
        else None
    )
    material = json.dumps(
        {
            "run_id": state["run_id"],
            "experiment_id": state.get("current_experiment_id"),
            "stage": stage.value,
            "repair_attempt": state["repair_attempts"],
            "subject_sha256": subject_sha256,
            "implementation_diff_sha256": diff_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ValidationOperationIdentity(
        operation_id=f"validation-operation-{hashlib.sha256(material.encode()).hexdigest()}",
        run_id=state["run_id"],
        experiment_id=_exp_id(state),
        stage=stage,
        repair_attempt=state["repair_attempts"],
        subject_sha256=subject_sha256,
        implementation_diff_sha256=diff_sha256,
    )


@dataclass
class ServiceTransitions:
    agent_client: AgentClient | None = None
    agent_clients: Mapping[AgentRole, AgentClient] = field(
        default_factory=lambda: dict[AgentRole, AgentClient]()
    )
    evaluator: Evaluator | None = None
    executor: Executor | None = None
    worktree_manager: WorktreeManager | None = None
    resource_accountant: ResourceAccountant | None = None
    policy_gate: PolicyGate | None = None
    run_store: RunStore | None = None
    frontier_service: FrontierService | None = None
    export_service: ExportService | None = None
    bundle_service: FinalizationBundleService | None = None
    repository_root: str | None = None
    runtime_root: str | None = None
    parent_commit: str | None = None
    docker_image: str | None = None
    dataset_root: str | None = None
    evaluator_id: str | None = None
    default_timeout_seconds: int = 300
    default_memory_bytes: int = 1 << 30
    default_cpus: float = 1.0
    default_gpu_count: int = 0
    smoke_timeout_seconds: int = 30
    smoke_memory_bytes: int = 512 * 1024 * 1024
    smoke_disk_bytes: int = 64 * 1024 * 1024
    dataset_view_provenance: Callable[[ExecutionRequest], DatasetViewProvenance] | None = None
    max_repairs: int = 3


def _unresolved_blockers(
    s: ServiceTransitions, experiment_id: str
) -> tuple[ValidationBlocker, ...]:
    if s.run_store is None:
        raise MissingAuthorityError("validation ledger authority is absent")
    return s.run_store.get_unresolved_blockers(experiment_id)


def _has_validation_ledger(s: ServiceTransitions) -> bool:
    return s.run_store is not None and all(
        callable(getattr(s.run_store, method, None))
        for method in (
            "put_validation_report",
            "get_validation_report_by_operation",
            "get_unresolved_blockers",
        )
    )


def _canonical_validation_report(
    report: ValidationReport, operation: ValidationOperationIdentity
) -> ValidationReport:
    report_id = f"validation-report-{operation.operation_id}"
    introduced_ids = {blocker.blocker_id for blocker in report.blockers}
    if introduced_ids & set(report.resolves_blocker_ids):
        raise ValueError("a validation report cannot resolve a blocker it introduces")
    blockers = tuple(
        ValidationBlocker(
            blocker_id=validation_blocker_id(report_id, operation.stage, blocker.text),
            experiment_id=operation.experiment_id,
            stage=operation.stage,
            text=blocker.text,
            report_id=report_id,
            evidence_refs=blocker.evidence_refs,
        )
        for blocker in report.blockers
    )
    return report.model_copy(
        update={
            "report_id": report_id,
            "experiment_id": operation.experiment_id,
            "stage": operation.stage,
            "blockers": blockers,
            "validation_operation_id": operation.operation_id,
        }
    )


def _replayed_validation_updates(
    s: ServiceTransitions, state: ProductionState, report: ValidationReport
) -> dict[str, object]:
    store = cast(RunStore, s.run_store)
    unresolved_blockers = store.get_unresolved_blockers(report.experiment_id)
    unresolved = tuple(blocker.blocker_id for blocker in unresolved_blockers)
    route = route_after_validation(state, report, unresolved)
    updates: dict[str, object] = {"latest_validation_report_id": report.report_id}
    if route not in {"repair", "persist_failure"}:
        return updates | {"pending_route": route}
    message = "; ".join(blocker.text for blocker in report.blockers)
    if unresolved:
        message = "unresolved validation blockers: " + "; ".join(
            f"{blocker.blocker_id}: {blocker.text}" for blocker in unresolved_blockers
        )
    if not message:
        message = f"{report.stage.value} validation rejected"
    return updates | _failure(
        state,
        FailureKind.SCHEMA_MISMATCH if route == "repair" else FailureKind.UNSTABLE_VALIDATION,
        message,
        report.evidence_refs,
    )


def make_service_transitions(
    *,
    agent_client: AgentClient | None = None,
    agent_clients: Mapping[AgentRole, AgentClient] | None = None,
    evaluator: Evaluator | None = None,
    executor: Executor | None = None,
    worktree_manager: WorktreeManager | None = None,
    resource_accountant: ResourceAccountant | None = None,
    policy_gate: PolicyGate | None = None,
    run_store: RunStore | None = None,
    frontier_service: FrontierService | None = None,
    export_service: ExportService | None = None,
    bundle_service: FinalizationBundleService | None = None,
    repository_root: str | None = None,
    runtime_root: str | None = None,
    parent_commit: str | None = None,
    docker_image: str | None = None,
    dataset_root: str | None = None,
    evaluator_id: str | None = None,
    default_timeout_seconds: int = 300,
    default_memory_bytes: int = 1 << 30,
    default_cpus: float = 1.0,
    default_gpu_count: int = 0,
    smoke_timeout_seconds: int = 30,
    smoke_memory_bytes: int = 512 * 1024 * 1024,
    smoke_disk_bytes: int = 64 * 1024 * 1024,
    dataset_view_provenance: Callable[[ExecutionRequest], DatasetViewProvenance] | None = None,
    max_repairs: int = 3,
) -> Mapping[str, Transition]:
    s = ServiceTransitions(
        agent_client=agent_client,
        agent_clients=agent_clients if agent_clients is not None else {},
        evaluator=evaluator,
        executor=executor,
        worktree_manager=worktree_manager,
        resource_accountant=resource_accountant,
        policy_gate=policy_gate,
        run_store=run_store,
        frontier_service=frontier_service,
        export_service=export_service,
        bundle_service=bundle_service,
        repository_root=repository_root,
        runtime_root=runtime_root,
        parent_commit=parent_commit,
        docker_image=docker_image,
        dataset_root=dataset_root,
        evaluator_id=evaluator_id,
        default_timeout_seconds=default_timeout_seconds,
        default_memory_bytes=default_memory_bytes,
        default_cpus=default_cpus,
        default_gpu_count=default_gpu_count,
        smoke_timeout_seconds=smoke_timeout_seconds,
        smoke_memory_bytes=smoke_memory_bytes,
        smoke_disk_bytes=smoke_disk_bytes,
        dataset_view_provenance=dataset_view_provenance,
        max_repairs=max_repairs,
    )
    return {
        "bootstrap": _bootstrap(s),
        "inspect": _inspect(s),
        "orchestrate": _orchestrate(s),
        "research": _research(s),
        "proposal_policy": _proposal_policy(s),
        "proposal_validation": _proposal_validation(s),
        "create_worktree": _create_worktree(s),
        "implement": _implement(s),
        "diff_policy": _diff_policy(s),
        "implementation_validation": _implementation_validation(s),
        "register_source": _register_source(s),
        "preflight": _preflight(s),
        "smoke": _smoke(s),
        "execute": _execute(s),
        "evaluate": _evaluate(s),
        "result_validation": _result_validation(s),
        "interpret": _interpret(s),
        "persist": _persist(s),
        "update_frontier": _update_frontier(s),
        "repair": _repair(s),
        "persist_failure": _persist_failure(s),
        "finalize": _finalize(s),
        "export": _export(s),
    }


def _bootstrap(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is None:
            return {"phase": RunPhase.BOOTSTRAP, "pending_route": "inspect"}
        s.run_store.put_run(
            RunRecord(run_id=state["run_id"], status="active"), f"{state['run_id']}-active"
        )
        if s.frontier_service is not None:
            initialize = getattr(s.frontier_service, "initialize", None)
            if initialize is not None:
                initialize(state["run_id"])
        return {"phase": RunPhase.BOOTSTRAP, "pending_route": "inspect"}

    return transition


def _inspect(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        _ = state, s
        return {"phase": RunPhase.RESEARCH, "pending_route": "orchestrate"}

    return transition


def _orchestrate(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.ORCHESTRATION)
        if client is None:
            return {"pending_route": "research"}
        finalization_ready = _finalization_ready(s, state)
        allowed_actions = (DecisionAction.RESEARCH,)
        if finalization_ready:
            allowed_actions = (
                DecisionAction.RESEARCH,
                DecisionAction.REPLICATE,
                DecisionAction.INCREASE_FIDELITY,
                DecisionAction.REVISIT_BRANCH,
                DecisionAction.STOP,
            )
        request = OrchestrationRequest(
            request_id=f"orchestration-{state['run_id']}-{state['state_version']}",
            run_id=state["run_id"],
            phase=state["phase"],
            allowed_actions=allowed_actions,
            resource_state=_resource_state(s),
            current_experiment_id=state.get("current_experiment_id"),
            latest_evaluation_result_id=state.get("latest_evaluation_result_id"),
            finalization_ready=finalization_ready,
            failure_summary=state.get("terminal_reason"),
            controller_context=_controller_context(s),
        )
        response = await client.invoke(request)
        if isinstance(response, AgentFailure):
            return _agent_failure(state, response)
        if not isinstance(response, OrchestrationDecision):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid orchestration response")
        if response.action not in request.allowed_actions:
            return _failure(
                state,
                FailureKind.SCHEMA_MISMATCH,
                f"orchestration selected disallowed action: {response.action.value}",
            )
        if (
            response.target_experiment_id is not None
            and response.target_experiment_id != request.current_experiment_id
        ):
            return _failure(
                state,
                FailureKind.SCHEMA_MISMATCH,
                "orchestration selected an unauthorized experiment identity",
            )
        return {
            "orchestration_decision_id": response.decision_id,
            "pending_route": route_after_orchestration(response),
        }

    return transition


def _finalization_ready(s: ServiceTransitions, state: ProductionState) -> bool:
    experiment_id = state.get("current_experiment_id")
    evaluation_id = state.get("latest_evaluation_result_id")
    if (
        s.run_store is None
        or s.bundle_service is None
        or s.evaluator_id is None
        or experiment_id is None
        or evaluation_id is None
    ):
        return False
    evaluation = s.run_store.get_evaluation_result(evaluation_id)
    source = (
        s.run_store.get_source_registration_by_id(f"source-{evaluation.source_commit}")
        if evaluation is not None and evaluation.source_commit is not None
        else s.run_store.get_source_registration(experiment_id)
    )
    return (
        source is not None
        and evaluation is not None
        and evaluation.experiment_id == experiment_id
    )


def _research(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.RESEARCH)
        if client is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "research role is not configured")
        objective = "propose next experiment"
        unresolved_context: tuple[ValidationBlockerContext, ...] = ()
        if state.get("terminal_reason"):
            _, message, _ = _failure_details(state)
            objective += f"; address validator feedback: {message}"
            try:
                unresolved_context = _blocker_context(
                    _unresolved_blockers(s, _exp_id(state))
                )
            except MissingAuthorityError as error:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        request = ResearchRequest(
            request_id=f"research-{state['run_id']}-{state['state_version']}",
            objective=objective,
            resource_state=_resource_state(s),
            allowed_paths=IMPLEMENTATION_ROOTS,
            controller_context=_controller_context(s),
            unresolved_blockers=unresolved_context,
        )
        response = await client.invoke(request)
        if isinstance(response, AgentFailure):
            return _agent_failure(state, response)
        if not isinstance(response, ResearchDecision) or response.experiment_spec is None:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "research returned no experiment spec"
            )
        spec = response.experiment_spec
        is_new_experiment = spec.experiment_id != state.get("current_experiment_id")
        if s.run_store is not None:
            s.run_store.put_experiment(
                spec, "proposed", state["run_id"], f"proposed-{spec.experiment_id}"
            )
        return {
            "phase": RunPhase.RESEARCH,
            "current_experiment_id": spec.experiment_id,
            "current_hypothesis_id": spec.hypothesis_id,
            "repair_attempts": 0 if is_new_experiment else state["repair_attempts"],
            "terminal_reason": None,
            "pending_route": "proposal_policy",
        }

    return transition


def _resource_state(s: ServiceTransitions) -> ResourceState:
    if s.resource_accountant is not None:
        return s.resource_accountant.state()
    return ResourceState(
        remaining_gpu_hours=0,
        accumulated_gpu_hours=0,
        remaining_wall_seconds=0,
        used_tokens=0,
        remaining_tokens=0,
        disk_bytes_available=0,
        reserved_final_gpu_hours=0,
    )


def _controller_context(s: ServiceTransitions) -> ControllerContext:
    dataset = s.run_store.get_dataset_manifest_identity() if s.run_store is not None else None
    evaluator = (
        s.run_store.get_evaluator_identity(s.evaluator_id)
        if s.run_store is not None and s.evaluator_id is not None
        else None
    )
    return ControllerContext(
        dataset_manifest_identity=dataset,
        evaluator_identity=evaluator,
        parent_commit=s.parent_commit,
        docker_image=s.docker_image,
        experiment_registry=(
            s.run_store.get_experiment_registry() if s.run_store is not None else None
        ),
    )


def _spec(s: ServiceTransitions, state: ProductionState) -> ExperimentSpec:
    if s.run_store is None:
        raise MissingAuthorityError("run store is required for experiment authority")
    value = s.run_store.get_experiment(_exp_id(state))
    if value is None:
        raise MissingAuthorityError("persisted experiment spec was not found")
    return value


def _deterministic_seed(run_id: str, experiment_id: str) -> int:
    material = f"tiktok2026:{run_id}:{experiment_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _proposal_policy(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        try:
            spec = _spec(s, state)
        except MissingAuthorityError as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        if s.policy_gate is not None:
            decision = s.policy_gate.check_paths(
                spec.implementation_scope, IMPLEMENTATION_ROOTS
            )
            if not decision.allowed:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, decision.reason)
        return {"pending_route": "proposal_validation"}

    return transition


def _proposal_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.VALIDATOR)
        if client is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "validator role is not configured")
        if not _has_validation_ledger(s):
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validation ledger authority is absent"
            )
        try:
            spec = _spec(s, state)
            store = cast(RunStore, s.run_store)
            bound_attempt = store.get_validation_report_for_attempt(
                state["run_id"],
                spec.experiment_id,
                ValidationStage.PROPOSAL,
                state["repair_attempts"],
            )
            if bound_attempt is not None:
                return _replayed_validation_updates(s, state, bound_attempt)
            unresolved = _unresolved_blockers(s, spec.experiment_id)
        except MissingAuthorityError as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        subject: dict[str, object] = {
            "experiment_spec": spec.model_dump(mode="json"),
            "controller_context": _controller_context(s).model_dump(mode="json"),
            "unresolved_blockers": [
                item.model_dump(mode="json") for item in _blocker_context(unresolved)
            ],
        }
        operation = _validation_operation(state, ValidationStage.PROPOSAL, subject)
        subject["validation_operation"] = operation.model_dump(mode="json")
        bound = store.get_validation_report_by_operation(operation.operation_id)
        if bound is not None:
            return _replayed_validation_updates(s, state, bound)
        response = await client.invoke(
            ValidationRequest(
                request_id=operation.operation_id,
                experiment_id=spec.experiment_id,
                stage=ValidationStage.PROPOSAL,
                validation_operation=operation,
                subject=subject,
            )
        )
        if isinstance(response, AgentFailure):
            return _agent_failure(state, response)
        if not isinstance(response, ValidationReport):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid proposal validation")
        if (
            response.experiment_id != spec.experiment_id
            or response.stage != ValidationStage.PROPOSAL
        ):
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validation response identity mismatch"
            )
        return _validation_updates(s, state, response, operation, subject)

    return transition


def _validation_updates(
    s: ServiceTransitions,
    state: ProductionState,
    report: ValidationReport,
    operation: ValidationOperationIdentity,
    subject: dict[str, object],
) -> dict[str, object]:
    if not _has_validation_ledger(s):
        return _failure(
            state, FailureKind.SCHEMA_MISMATCH, "validation ledger authority is absent"
        )
    report = _canonical_validation_report(report, operation)
    store = cast(RunStore, s.run_store)
    store.put_validation_report(report, state["run_id"], operation, subject)
    unresolved_blockers = store.get_unresolved_blockers(report.experiment_id)
    unresolved = tuple(blocker.blocker_id for blocker in unresolved_blockers)
    route = route_after_validation(state, report, unresolved)
    updates: dict[str, object] = {"latest_validation_report_id": report.report_id}
    if route not in {"repair", "persist_failure"}:
        return updates | {"pending_route": route}
    message = "; ".join(blocker.text for blocker in report.blockers)
    if unresolved:
        message = "unresolved validation blockers: " + "; ".join(
            f"{blocker.blocker_id}: {blocker.text}" for blocker in unresolved_blockers
        )
    if not message:
        message = f"{report.stage.value} validation rejected"
    return updates | _failure(
        state,
        (
            FailureKind.SCHEMA_MISMATCH
            if route == "repair"
            else FailureKind.UNSTABLE_VALIDATION
        ),
        message,
        report.evidence_refs,
    )


def _create_worktree(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.worktree_manager is None:
            return _failure(state, FailureKind.MISSING_PATH, "worktree capability is absent")
        try:
            spec = _spec(s, state)
            if s.parent_commit is None:
                raise MissingAuthorityError("approved parent commit is absent")
            assignment = s.worktree_manager.create(state["run_id"], spec, s.parent_commit)
            if s.run_store is not None:
                s.run_store.put_worktree_assignment(assignment)
            implementor = _agent(s, AgentRole.IMPLEMENTOR)
            binder = getattr(implementor, "bind_worktree", None)
            if binder is not None:
                binder(assignment.path, spec.implementation_scope)
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.MISSING_PATH, str(error))
        return {
            "phase": RunPhase.IMPLEMENT,
            "active_worktree_id": assignment.worktree_id,
            "pending_route": "implement",
        }

    return transition


def _implement(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.IMPLEMENTOR)
        if client is None:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "implementor role is not configured"
            )
        if not _has_validation_ledger(s):
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validation ledger authority is absent"
            )
        try:
            spec = _spec(s, state)
        except MissingAuthorityError as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        repository = getattr(client, "scoped_repository", None)
        prior_diff_sha256: str | None = None
        try:
            source_context: dict[str, str] = {}
            base_source_context: dict[str, str] = {}
            if repository is not None:
                prior_diff_sha256 = hashlib.sha256(repository.diff().encode()).hexdigest()
                current_source = repository.read(EXPERIMENT_ENTRYPOINT, 100_000)
                base_source = repository.read_base(EXPERIMENT_ENTRYPOINT, 100_000)
                source_context[EXPERIMENT_ENTRYPOINT] = current_source
                if current_source != base_source:
                    base_source_context[EXPERIMENT_ENTRYPOINT] = base_source
        except (OSError, PermissionError, ValueError, RuntimeError) as error:
            return _failure(state, FailureKind.MISSING_PATH, str(error))
        unresolved = _unresolved_blockers(s, spec.experiment_id)
        unresolved_blocker_ids = tuple(blocker.blocker_id for blocker in unresolved)
        unresolved_blocker_context = _blocker_context(unresolved)
        response = await client.invoke(
            ImplementationRequest(
                request_id=f"implementation-{state['run_id']}-{state['state_version']}",
                experiment_id=spec.experiment_id,
                experiment_spec=spec,
                allowed_scopes=spec.implementation_scope,
                capabilities=("scoped_read", "scoped_write", "diff", "checks"),
                repair_feedback=(
                    _failure_details(state)[1] if state["repair_attempts"] > 0 else None
                ),
                unresolved_blocker_ids=unresolved_blocker_ids,
                unresolved_blockers=unresolved_blocker_context,
                source_context=source_context,
                base_source_context=base_source_context,
                execution_timeout_seconds=s.default_timeout_seconds,
                execution_memory_bytes=s.default_memory_bytes,
                execution_cpus=s.default_cpus,
                execution_gpu_count=s.default_gpu_count,
                prior_diff_sha256=prior_diff_sha256,
            )
        )
        if isinstance(response, AgentFailure):
            return _agent_failure(state, response)
        if not isinstance(response, ImplementationResult):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid implementation response")
        if repository is not None:
            try:
                changed_files = tuple(repository.changed_files())
                diff = repository.diff()
                if not changed_files or not diff:
                    return _failure(
                        state, FailureKind.SCHEMA_MISMATCH, "implementation produced no real diff"
                    )
                result_diff_sha256 = hashlib.sha256(diff.encode()).hexdigest()
                if (
                    state["repair_attempts"] > 0
                    and result_diff_sha256 == prior_diff_sha256
                ):
                    return _failure(
                        state,
                        FailureKind.SCHEMA_MISMATCH,
                        "repair attempt did not change the authoritative implementation diff",
                    )
            except (OSError, PermissionError, ValueError, RuntimeError) as error:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
            response = response.model_copy(update={"changed_files": changed_files})
        else:
            result_diff_sha256 = None
        if s.run_store is not None:
            record = ImplementationAttemptRecord(
                experiment_id=response.experiment_id,
                repair_attempt=state["repair_attempts"],
                result=response,
                prior_diff_sha256=prior_diff_sha256,
                result_diff_sha256=result_diff_sha256,
            )
            s.run_store.put_json(
                "implementation",
                f"{response.experiment_id}:attempt:{state['repair_attempts']}",
                record.model_dump_json(),
            )
        return {
            "phase": RunPhase.IMPLEMENT,
            "terminal_reason": None,
            "pending_route": "diff_policy",
        }

    return transition


def _implementation_result(
    s: ServiceTransitions, state: ProductionState
) -> ImplementationResult | None:
    if s.run_store is None:
        return None
    values = s.run_store.list_json("implementation")
    for value in values:
        try:
            record = ImplementationAttemptRecord.model_validate_json(value)
            attempt = record.repair_attempt
            result = record.result
        except ValueError:
            # Existing pre-versioning records represent the initial attempt.
            attempt = 0
            result = ImplementationResult.model_validate_json(value)
        if (
            result.experiment_id == state.get("current_experiment_id")
            and attempt == state["repair_attempts"]
        ):
            return result
    return None


def _diff_policy(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        result = _implementation_result(s, state)
        repository = getattr(_agent(s, AgentRole.IMPLEMENTOR), "scoped_repository", None)
        if repository is not None:
            try:
                changed_files = tuple(repository.changed_files())
                if not changed_files or not repository.diff():
                    return _failure(
                        state, FailureKind.SCHEMA_MISMATCH, "implementation produced no real diff"
                    )
                result = (
                    result.model_copy(update={"changed_files": changed_files})
                    if result
                    else None
                )
            except (OSError, PermissionError, ValueError, RuntimeError) as error:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        if result is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "implementation result is absent")
        if EXPERIMENT_ENTRYPOINT not in result.changed_files:
            return _failure(
                state,
                FailureKind.SCHEMA_MISMATCH,
                f"implementation must integrate the execution entrypoint: {EXPERIMENT_ENTRYPOINT}",
            )
        if s.policy_gate is not None:
            spec = _spec(s, state)
            decision = s.policy_gate.check_paths(result.changed_files, spec.implementation_scope)
            if not decision.allowed:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, decision.reason)
        return {"pending_route": "implementation_validation"}

    return transition


def _validation(
    s: ServiceTransitions, state: ProductionState, stage: ValidationStage, route: str
) -> Transition:
    async def transition(_: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.VALIDATOR)
        if client is None:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validator role is not configured"
            )
        if s.run_store is None:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validation ledger authority is absent"
            )
        if not _has_validation_ledger(s):
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validation ledger authority is absent"
            )
        try:
            if stage != ValidationStage.IMPLEMENTATION:
                bound_attempt = s.run_store.get_validation_report_for_attempt(
                    state["run_id"],
                    _exp_id(state),
                    stage,
                    state["repair_attempts"],
                )
                if bound_attempt is not None:
                    return _replayed_validation_updates(s, state, bound_attempt)
            subject = _validation_subject(s, state, stage)
        except MissingAuthorityError as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        if stage == ValidationStage.IMPLEMENTATION:
            assignment = s.run_store.get_worktree_assignment(_exp_id(state))
            if assignment is None:
                return _failure(
                    state,
                    FailureKind.MISSING_PATH,
                    "validator worktree assignment is absent",
                )
            binder = getattr(client, "bind_worktree", None)
            if binder is not None:
                binder(assignment.path, _spec(s, state).implementation_scope)
        operation = _validation_operation(state, stage, subject)
        subject["validation_operation"] = operation.model_dump(mode="json")
        bound = s.run_store.get_validation_report_by_operation(operation.operation_id)
        if bound is not None:
            return _replayed_validation_updates(s, state, bound)
        response = await client.invoke(
            ValidationRequest(
                request_id=operation.operation_id,
                experiment_id=_exp_id(state),
                stage=stage,
                validation_operation=operation,
                subject=subject,
            )
        )
        if isinstance(response, AgentFailure):
            return _agent_failure(state, response)
        if not isinstance(response, ValidationReport):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid validation response")
        if response.experiment_id != _exp_id(state) or response.stage != stage:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "validation response identity mismatch"
            )
        return _validation_updates(s, state, response, operation, subject)

    return transition


def _validation_subject(
    s: ServiceTransitions, state: ProductionState, stage: ValidationStage
) -> dict[str, object]:
    subject: dict[str, object] = {
        "experiment_spec": _spec(s, state).model_dump(mode="json"),
        "controller_context": _controller_context(s).model_dump(mode="json"),
        "unresolved_blockers": [
            item.model_dump(mode="json")
            for item in _blocker_context(_unresolved_blockers(s, _exp_id(state)))
        ],
    }
    if stage == ValidationStage.IMPLEMENTATION:
        implementation = _implementation_result(s, state)
        if implementation is None:
            raise MissingAuthorityError("implementation result is absent")
        subject["implementation_result"] = implementation.model_dump(mode="json")
        if s.run_store is None:
            raise MissingAuthorityError("run store is required for implementation validation")
        assignment = s.run_store.get_worktree_assignment(_exp_id(state))
        repository = getattr(_agent(s, AgentRole.IMPLEMENTOR), "scoped_repository", None)
        if assignment is None or repository is None:
            raise MissingAuthorityError("implementation worktree authority is absent")
        try:
            diff = repository.diff()
            changed_files = tuple(repository.changed_files())
        except (OSError, PermissionError, ValueError, RuntimeError) as error:
            raise MissingAuthorityError(str(error)) from error
        if not diff or not changed_files:
            raise MissingAuthorityError("implementation diff authority is absent")
        digest = hashlib.sha256(diff.encode()).hexdigest()
        subject["implementation_authority"] = ImplementationValidationAuthority(
            evidence_id=f"implementation-diff-{digest}",
            worktree_id=assignment.worktree_id,
            parent_commit=assignment.parent_commit,
            diff_sha256=digest,
            changed_files=changed_files,
            allowed_scopes=_spec(s, state).implementation_scope,
        ).model_dump(mode="json")
        subject["execution_resources"] = {
            "timeout_seconds": s.default_timeout_seconds,
            "memory_bytes": s.default_memory_bytes,
            "cpus": s.default_cpus,
            "gpu_count": s.default_gpu_count,
        }
    elif stage == ValidationStage.RESULT:
        if s.run_store is None:
            raise MissingAuthorityError("run store is required for result validation")
        evaluation_id = state.get("latest_evaluation_result_id")
        execution_id = state.get("latest_execution_result_id")
        evaluation = s.run_store.get_evaluation_result(evaluation_id) if evaluation_id else None
        execution = s.run_store.get_execution_result(execution_id) if execution_id else None
        if evaluation is None or execution is None:
            raise MissingAuthorityError("result validation evidence is absent")
        subject["evaluation_result"] = evaluation.model_dump(mode="json")
        subject["execution_result"] = execution.model_dump(mode="json")
    return subject


def _implementation_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        return await _validation(s, state, ValidationStage.IMPLEMENTATION, "register_source")(state)

    return transition


def _register_source(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.worktree_manager is None or s.run_store is None:
            return _failure(
                state, FailureKind.MISSING_PATH, "source authority capability is absent"
            )
        try:
            assignment = s.run_store.get_worktree_assignment(_exp_id(state))
            if assignment is None:
                raise MissingAuthorityError("worktree assignment was not found")
            spec = _spec(s, state)
            previous = s.run_store.get_source_registration(spec.experiment_id)
            registration = s.worktree_manager.register_source(
                assignment, spec.implementation_scope, previous
            )
            s.run_store.put_source_registration(registration)
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        return {"pending_route": "preflight"}

    return transition


def _preflight(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is not None and s.run_store.get_source_registration(_exp_id(state)) is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "source registration was not found")
        return {"phase": RunPhase.EXECUTE, "pending_route": "smoke"}

    return transition


def _smoke(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if (
            s.executor is None
            or s.run_store is None
            or s.dataset_root is None
            or s.runtime_root is None
            or s.resource_accountant is None
        ):
            return _failure(
                state, FailureKind.MISSING_PATH, "smoke execution provenance is incomplete"
            )
        try:
            spec = _spec(s, state)
            registration = s.run_store.get_source_registration(spec.experiment_id)
            assignment = s.run_store.get_worktree_assignment(spec.experiment_id)
            manifest = s.run_store.get_dataset_manifest_identity()
            if registration is None or assignment is None or manifest is None:
                raise MissingAuthorityError(
                    "smoke source, worktree, or dataset authority is absent"
                )
            if s.dataset_view_provenance is None:
                raise MissingAuthorityError("current authorized smoke dataset view is absent")
            execution_id = f"smoke-{state['run_id']}-{spec.experiment_id}-{state['state_version']}"
            command = (
                "python",
                "-m",
                "tiktok2026.experiment.train",
                "--output-dir=/output",
                f"--seed={_deterministic_seed(state['run_id'], spec.experiment_id)}",
                "--fidelity=smoke",
                f"--source-commit={registration.source_commit}",
                f"--execution-id={execution_id}",
            )
            request = ExecutionRequest(
                run_id=state["run_id"],
                execution_id=execution_id,
                experiment_id=spec.experiment_id,
                source_registration_id=registration.registration_id,
                source_commit=registration.source_commit,
                command=command,
                image=s.docker_image or "",
                source_path=assignment.path,
                dataset_path=Path(s.dataset_root),
                dataset_manifest_sha256=manifest.manifest_sha256,
                output_path=Path(s.runtime_root) / "smoke-placeholder",
                timeout_seconds=s.smoke_timeout_seconds,
                memory_bytes=s.smoke_memory_bytes,
                cpus=s.default_cpus,
                gpu_count=s.default_gpu_count,
                execution_kind="smoke",
            )
            current_view = s.dataset_view_provenance(request)
            request = request.model_copy(update={"dataset_view_sha256": current_view.view_sha256})
            result = s.run_store.get_execution_result(execution_id)
            if result is not None and (
                result.execution_kind != "smoke"
                or result.experiment_id != spec.experiment_id
                or result.source_registration_id != registration.registration_id
                or result.source_commit != registration.source_commit
                or result.command != command
                or result.dataset_manifest_id != current_view.manifest_id
                or result.dataset_manifest_sha256 != current_view.manifest_sha256
                or result.dataset_view_sha256 != current_view.view_sha256
                or result.dataset_valid_rows != current_view.valid_rows
            ):
                raise MissingAuthorityError("persisted smoke result does not match the request")
            reservation_id = f"reservation-{execution_id}"
            reservation = ResourceReservation(
                reservation_id=reservation_id,
                run_id=state["run_id"],
                experiment_id=spec.experiment_id,
                gpu_hours=smoke_gpu_hours(s),
                wall_seconds=float(s.smoke_timeout_seconds + 5),
                tokens=0,
                disk_bytes=s.smoke_disk_bytes,
            )
            if not s.resource_accountant.reserve(reservation):
                return _failure(state, FailureKind.DISK, "smoke resource reservation was denied")
            if result is None:
                request = request.model_copy(
                    update={
                        "output_path": _fresh_execution_output_path(
                            s.runtime_root, execution_id
                        )
                    }
                )
                result = await s.executor.execute(request)
                s.run_store.put_execution_result(result)
            s.resource_accountant.consume(
                reservation_id,
                gpu_hours=result.gpu_hours,
                wall_seconds=result.elapsed_seconds,
                tokens=0,
                disk_bytes=result.artifact_output_bytes,
            )
            s.resource_accountant.reconcile(
                reservation_id,
                gpu_hours=result.gpu_hours,
                wall_seconds=result.elapsed_seconds,
                tokens=0,
                disk_bytes=result.artifact_output_bytes,
            )
            if result.failure_kind is not None:
                return _failure(
                    state,
                    result.failure_kind,
                    result.failure_message or "smoke execution failed",
                    (result.execution_id,),
                )
            if (
                result.dataset_manifest_id != current_view.manifest_id
                or result.dataset_manifest_sha256 != current_view.manifest_sha256
                or result.dataset_view_sha256 != current_view.view_sha256
                or result.dataset_valid_rows != current_view.valid_rows
            ):
                return _failure(
                    state,
                    FailureKind.SCHEMA_MISMATCH,
                    "smoke dataset provenance does not match current authorized view",
                    (result.execution_id,),
                )
            decision = check_smoke_feasibility(
                result,
                memory_limit_bytes=s.smoke_memory_bytes,
                timeout_seconds=s.smoke_timeout_seconds,
                gpu_requested=s.default_gpu_count > 0,
            )
            if not decision.allowed:
                return _failure(
                    state,
                    FailureKind.SCHEMA_MISMATCH,
                    decision.reason,
                    (result.execution_id,),
                )
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        return {"phase": RunPhase.EXECUTE, "pending_route": "execute"}

    return transition


def smoke_gpu_hours(s: ServiceTransitions) -> float:
    return s.smoke_timeout_seconds * s.default_gpu_count / 3600.0


def _fresh_execution_output_path(runtime_root: str, execution_id: str) -> Path:
    """Create the controller-owned, empty host directory for one execution."""

    root = Path(runtime_root).resolve()
    artifacts_root = root / "artifacts"
    execution_root = artifacts_root / ".execution"
    if artifacts_root.is_symlink() or execution_root.is_symlink():
        raise ValueError("execution output root must not be a symlink")
    try:
        execution_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        execution_root.chmod(0o755)
        output = execution_root / hashlib.sha256(execution_id.encode()).hexdigest()
        if output.exists():
            raise ValueError("execution output path already exists")
        output.mkdir(parents=True, mode=0o777)
        # The container runs as a fixed non-root UID.  This directory is
        # isolated per execution, so granting it writable access does not
        # expose other runtime state.
        output.chmod(0o777)
    except OSError as error:
        raise ValueError("execution output directory could not be prepared") from error
    return output


def _execute(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if (
            s.executor is None
            or s.run_store is None
            or s.dataset_root is None
            or s.runtime_root is None
        ):
            return _failure(state, FailureKind.MISSING_PATH, "execution provenance is incomplete")
        try:
            spec = _spec(s, state)
            registration = s.run_store.get_source_registration(spec.experiment_id)
            assignment = s.run_store.get_worktree_assignment(spec.experiment_id)
            manifest = s.run_store.get_dataset_manifest_identity()
            if registration is None or assignment is None or manifest is None:
                raise MissingAuthorityError("source, worktree, or dataset authority is absent")
            execution_id = (
                f"execution-{state['run_id']}-{spec.experiment_id}-{state['state_version']}"
            )
            command = (
                "python",
                "-m",
                "tiktok2026.experiment.train",
                "--output-dir=/output",
                f"--seed={_deterministic_seed(state['run_id'], spec.experiment_id)}",
                f"--fidelity={spec.fidelity.value}",
                f"--source-commit={registration.source_commit}",
                f"--execution-id={execution_id}",
            )
            result = s.run_store.get_execution_result(execution_id)
            if result is not None and (
                result.experiment_id != spec.experiment_id
                or result.source_registration_id != registration.registration_id
                or result.source_commit != registration.source_commit
                or result.command != command
            ):
                raise MissingAuthorityError(
                    "persisted execution result does not match the requested execution"
                )
            reservation_id = f"reservation-{execution_id}"
            reserved_wall_seconds = float(s.default_timeout_seconds + 5)
            reservation = ResourceReservation(
                reservation_id=reservation_id,
                run_id=state["run_id"],
                experiment_id=spec.experiment_id,
                gpu_hours=max(
                    spec.predicted_gpu_hours,
                    reserved_wall_seconds * s.default_gpu_count / 3600.0,
                ),
                wall_seconds=reserved_wall_seconds,
                tokens=0,
                disk_bytes=0,
            )
            if s.resource_accountant is not None and not s.resource_accountant.reserve(reservation):
                return _failure(state, FailureKind.DISK, "resource reservation was denied")
            if result is None:
                output_path = _fresh_execution_output_path(s.runtime_root, execution_id)
                request = ExecutionRequest(
                    run_id=state["run_id"],
                    execution_id=execution_id,
                    experiment_id=spec.experiment_id,
                    source_registration_id=registration.registration_id,
                    source_commit=registration.source_commit,
                    command=command,
                    image=s.docker_image or "",
                    source_path=assignment.path,
                    dataset_path=Path(s.dataset_root),
                    dataset_manifest_sha256=manifest.manifest_sha256,
                    output_path=output_path,
                    timeout_seconds=s.default_timeout_seconds,
                    memory_bytes=s.default_memory_bytes,
                    cpus=s.default_cpus,
                    gpu_count=s.default_gpu_count,
                )
                result = await s.executor.execute(request)
                s.run_store.put_execution_result(result)
            if s.resource_accountant is not None:
                s.resource_accountant.consume(
                    reservation_id,
                    gpu_hours=result.gpu_hours,
                    wall_seconds=result.elapsed_seconds,
                    tokens=0,
                    disk_bytes=0,
                )
                s.resource_accountant.reconcile(
                    reservation_id,
                    gpu_hours=result.gpu_hours,
                    wall_seconds=result.elapsed_seconds,
                    tokens=0,
                    disk_bytes=0,
                )
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        if result.failure_kind is not None:
            return _failure(
                state,
                result.failure_kind,
                result.failure_message or "execution failed",
                (result.execution_id,),
            )
        return {
            "phase": RunPhase.EXECUTE,
            "latest_execution_result_id": result.execution_id,
            "pending_route": "evaluate",
        }

    return transition


def _evaluate(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.evaluator is None or s.run_store is None or s.evaluator_id is None:
            return _failure(
                state, FailureKind.EVALUATOR_OUTPUT, "evaluation authority is incomplete"
            )
        try:
            execution_id = state.get("latest_execution_result_id")
            if not execution_id:
                raise MissingAuthorityError("execution result identity is absent")
            execution = s.run_store.get_execution_result(execution_id)
            if execution is None:
                raise MissingAuthorityError("execution provenance record is absent")
            registration = s.run_store.get_source_registration_by_id(
                execution.source_registration_id
            )
            manifest = s.run_store.get_dataset_manifest_identity()
            evaluator = s.run_store.get_evaluator_identity(s.evaluator_id)
            if registration is None or manifest is None or evaluator is None:
                raise MissingAuthorityError("evaluation provenance record is absent")
            if (
                execution.source_registration_id != registration.registration_id
                or execution.source_commit != registration.source_commit
            ):
                raise MissingAuthorityError("execution source does not match registered source")
            checkpoint_id = execution.checkpoint_id
            if checkpoint_id is None:
                raise MissingAuthorityError("execution did not return a checkpoint identity")
            prediction = next(
                (
                    artifact
                    for artifact_id in execution.artifact_ids
                    if (artifact := s.run_store.get_prediction_artifact(artifact_id)) is not None
                ),
                None,
            )
            if prediction is None:
                raise MissingAuthorityError(
                    "execution did not return a registered prediction artifact"
                )
            get_artifact = getattr(s.run_store, "get_artifact", None)
            prediction_record = (
                get_artifact(prediction.artifact_id) if get_artifact is not None else None
            )
            checkpoint_artifacts = (
                tuple(
                    artifact
                    for artifact_id in execution.artifact_ids
                    if (artifact := get_artifact(artifact_id)) is not None
                    and artifact.kind == "checkpoint"
                )
                if get_artifact is not None
                else (True,)
            )
            artifact_invalid = False
            if get_artifact is not None:
                artifact_invalid = (
                    prediction_record is None
                    or prediction_record.kind != "prediction"
                    or prediction_record.run_id != state["run_id"]
                    or prediction_record.experiment_id != _exp_id(state)
                    or prediction_record.sha256 != prediction.sha256
                    or not checkpoint_artifacts
                )
            if (
                artifact_invalid
                or prediction.checkpoint_id != checkpoint_id
                or prediction.source_commit != execution.source_commit
                or prediction.execution_id != execution.execution_id
                or prediction.dataset_manifest_id != manifest.manifest_id
                or prediction.dataset_manifest_sha256 != manifest.manifest_sha256
                or prediction.split != "valid"
            ):
                raise MissingAuthorityError("prediction artifact provenance is invalid")
            context = EvaluationContext(
                run_id=state["run_id"],
                evaluation_id=f"evaluation-{execution.execution_id}",
                experiment_id=_exp_id(state),
                checkpoint_id=checkpoint_id,
                source_commit=execution.source_commit,
                execution_id=execution.execution_id,
                dataset_manifest_id=manifest.manifest_id,
                dataset_manifest_sha256=manifest.manifest_sha256,
                split=prediction.split,
                prediction_artifact_id=prediction.artifact_id,
                prediction_sha256=prediction.sha256,
                evaluator_id=evaluator.evaluator_id,
                evaluator_sha256=evaluator.evaluator_sha256,
            )
            result = s.evaluator.evaluate(
                EvaluationRequest(evaluation_id=context.evaluation_id, context=context)
            )
            provenance = ProvenanceRequest(
                run_id=state["run_id"],
                experiment_id=_exp_id(state),
                source_commit=execution.source_commit,
                execution_id=execution.execution_id,
                dataset_manifest_id=manifest.manifest_id,
                dataset_manifest_sha256=manifest.manifest_sha256,
                evaluator_id=evaluator.evaluator_id,
                evaluator_sha256=evaluator.evaluator_sha256,
            )
            s.run_store.put_evaluation(result, provenance)
            _log_evaluation_metrics(s, result)
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.EVALUATOR_OUTPUT, str(error))
        return {
            "phase": RunPhase.EVALUATE,
            "latest_evaluation_result_id": result.evaluation_id,
            "pending_route": "result_validation",
        }

    return transition


def _log_evaluation_metrics(s: ServiceTransitions, result: EvaluationResult) -> None:
    metrics = {metric.name: metric.value for metric in result.metrics}
    ndcg = metrics["NDCG@10"]
    recall = metrics["Recall@50"]
    calibrations = (
        s.run_store.list_baseline_calibrations() if s.run_store is not None else ()
    )
    matching_calibrations = tuple(
        calibration
        for calibration in calibrations
        if calibration.dataset_manifest_sha256 == result.dataset_manifest_sha256
        and calibration.evaluator_sha256 == result.evaluator_sha256
        and calibration.split == result.split
    )
    if matching_calibrations:
        baseline = matching_calibrations[0]
        baseline_metrics = {
            metric.name: metric.value for metric in baseline.evaluation.metrics
        }
        baseline_ndcg = baseline_metrics["NDCG@10"]
        baseline_recall = baseline_metrics["Recall@50"]
        logger.info(
            "Provisional pipeline metrics experiment_id={} evaluation_id={} "
            "NDCG@10={:.6f} Recall@50={:.6f} composite={:.6f} "
            "baseline=starter_kit_fm baseline_calibration_id={} "
            "baseline_NDCG@10={:.6f} baseline_Recall@50={:.6f} "
            "delta_NDCG@10={:+.6f} delta_Recall@50={:+.6f} delta_composite={:+.6f}",
            result.experiment_id,
            result.evaluation_id,
            ndcg,
            recall,
            result.validation_score,
            baseline.calibration_id,
            baseline_ndcg,
            baseline_recall,
            ndcg - baseline_ndcg,
            recall - baseline_recall,
            result.validation_score - baseline.evaluation.validation_score,
        )
        return
    candidates = (
        s.run_store.list_evaluation_results() if s.run_store is not None else ()
    )
    comparable = tuple(
        candidate
        for candidate in candidates
        if candidate.evaluation_id != result.evaluation_id
        and candidate.run_id == result.run_id
        and candidate.dataset_manifest_sha256 == result.dataset_manifest_sha256
        and candidate.evaluator_sha256 == result.evaluator_sha256
        and candidate.split == result.split
        and candidate.validity == result.validity
    )
    if not comparable:
        logger.info(
            "Provisional pipeline metrics experiment_id={} evaluation_id={} "
            "NDCG@10={:.6f} Recall@50={:.6f} composite={:.6f} baseline=unavailable",
            result.experiment_id,
            result.evaluation_id,
            ndcg,
            recall,
            result.validation_score,
        )
        return
    baseline = max(comparable, key=lambda candidate: candidate.validation_score)
    baseline_metrics = {metric.name: metric.value for metric in baseline.metrics}
    baseline_ndcg = baseline_metrics["NDCG@10"]
    baseline_recall = baseline_metrics["Recall@50"]
    logger.info(
        "Provisional pipeline metrics experiment_id={} evaluation_id={} "
        "NDCG@10={:.6f} Recall@50={:.6f} composite={:.6f} "
        "baseline=prior_champion baseline_evaluation_id={} "
        "baseline_NDCG@10={:.6f} baseline_Recall@50={:.6f} "
        "delta_NDCG@10={:+.6f} delta_Recall@50={:+.6f} delta_composite={:+.6f}",
        result.experiment_id,
        result.evaluation_id,
        ndcg,
        recall,
        result.validation_score,
        baseline.evaluation_id,
        baseline_ndcg,
        baseline_recall,
        ndcg - baseline_ndcg,
        recall - baseline_recall,
        result.validation_score - baseline.validation_score,
    )


def _result_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        return await _validation(s, state, ValidationStage.RESULT, "interpret")(state)

    return transition


def _interpret(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.RESEARCH)
        if client is not None:
            response = await client.invoke(
                ResearchRequest(
                    request_id=f"interpret-{state['run_id']}-{state['state_version']}",
                    objective="interpret evaluation result",
                    resource_state=_resource_state(s),
                )
            )
            if isinstance(response, AgentFailure):
                return _agent_failure(state, response)
        return {"pending_route": "persist"}

    return transition


def _persist(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        try:
            spec = _spec(s, state)
            if s.run_store is not None:
                s.run_store.put_experiment(
                    spec,
                    "completed",
                    state["run_id"],
                    f"completed-{spec.experiment_id}",
                    expected_predecessor=f"proposed-{spec.experiment_id}",
                )
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        return {"phase": RunPhase.PERSIST, "pending_route": "update_frontier"}

    return transition


def _update_frontier(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        evaluation_id = state.get("latest_evaluation_result_id")
        if s.run_store is None or not evaluation_id:
            return _failure(
                state, FailureKind.SCIENTIFIC_NON_IMPROVEMENT, "evaluation evidence is absent"
            )
        result = s.run_store.get_evaluation_result(evaluation_id)
        if result is None:
            return _failure(
                state, FailureKind.SCIENTIFIC_NON_IMPROVEMENT, "evaluation evidence is absent"
            )
        score = result.validation_score
        terminal_reason = (
            s.frontier_service.update(_exp_id(state), score) if s.frontier_service else None
        )
        if terminal_reason is None:
            return {"pending_route": "orchestrate"}
        try:
            spec = _spec(s, state)
            s.run_store.put_experiment(
                spec,
                "converged",
                state["run_id"],
                f"converged-{spec.experiment_id}",
                expected_predecessor=f"completed-{spec.experiment_id}",
            )
            s.run_store.put_run(
                RunRecord(run_id=state["run_id"], status="converged"),
                f"{state['run_id']}-converged",
                expected_predecessor=f"{state['run_id']}-active",
            )
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        return {"terminal_reason": terminal_reason, "pending_route": "finalize"}

    return transition


def _repair(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if state["repair_attempts"] >= s.max_repairs:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "repair limit reached") | {
                "pending_route": "persist_failure"
            }
        if (
            s.policy_gate is not None
            and not s.policy_gate.can_repair(state["repair_attempts"]).allowed
        ):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "repair limit reached") | {
                "pending_route": "persist_failure"
            }
        # RESEARCH failures (proposal/schema issues) return to research so the
        # same hypothesis can be re-formed. IMPLEMENT/EXECUTE/EVALUATE failures
        # return to implement so the same experiment is re-implemented rather
        # than replaced by a new experiment via research.
        route = "research" if state["phase"] == RunPhase.RESEARCH else "implement"
        return {"repair_attempts": state["repair_attempts"] + 1, "pending_route": route}

    return transition


def _persist_failure(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        kind, message, evidence = _failure_details(state)
        record = FailureRecord(
            failure_id=f"failure-{state['run_id']}-{state['state_version']}",
            experiment_id=state.get("current_experiment_id"),
            kind=kind,
            evidence_refs=evidence or (message,),
            repair_attempt=state["repair_attempts"],
        )
        if s.run_store is not None:
            s.run_store.put_failure(record, state["run_id"])
        route = route_after_failure(state, record, s.max_repairs)
        if route == "orchestrate" and s.run_store is not None:
            try:
                spec = _spec(s, state)
                s.run_store.put_experiment(
                    spec,
                    "failed",
                    state["run_id"],
                    f"failed-{spec.experiment_id}",
                    expected_predecessor=f"proposed-{spec.experiment_id}",
                )
            except (MissingAuthorityError, ValueError) as error:
                raise TerminalLifecycleError(str(error)) from error
        if route == "terminal":
            raise TerminalLifecycleError(message)
        return {"pending_route": route}

    return transition


def _finalize(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        try:
            if s.run_store is None or s.bundle_service is None:
                raise MissingAuthorityError("finalization bundle authority is absent")
            exp_id = _exp_id(state)
            registration = s.run_store.get_source_registration(exp_id)
            evaluation_id = state.get("latest_evaluation_result_id")
            evaluation = s.run_store.get_evaluation_result(evaluation_id) if evaluation_id else None
            if registration is None or evaluation is None or s.evaluator_id is None:
                raise MissingAuthorityError("finalization provenance is absent")
            bundle = s.bundle_service.create(
                FinalizationBundleRequest(
                    run_id=state["run_id"],
                    experiment_id=exp_id,
                    source_commit=registration.source_commit,
                    checkpoint_id=evaluation.checkpoint_id,
                    evaluation_id=evaluation.evaluation_id,
                    evaluator_id=s.evaluator_id,
                )
            )
            s.run_store.persist_provisional_finalization(
                ProvisionalFinalizationRequest(
                    finalization_id=f"finalization-{state['run_id']}",
                    run_id=state["run_id"],
                    experiment_id=exp_id,
                    source_commit=registration.source_commit,
                    checkpoint_id=evaluation.checkpoint_id,
                    evaluation_id=evaluation.evaluation_id,
                    bundle_artifact_id=bundle.artifact_id,
                    evaluator_id=s.evaluator_id,
                )
            )
        except Exception as error:
            _mark_terminal_failure(
                s, state, FailureKind.SCHEMA_MISMATCH, str(error)
            )
        return {"phase": RunPhase.FINALIZE, "pending_route": "export"}

    return transition


def _export(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is None or s.run_store.get_finalization(
            f"finalization-{state['run_id']}"
        ) is None:
            _mark_terminal_failure(
                s,
                state,
                FailureKind.SCHEMA_MISMATCH,
                "export requires a persisted provisional finalization",
            )
        if s.export_service is not None:
            if s.runtime_root is None:
                return _failure(
                    state, FailureKind.MISSING_PATH, "runtime export root is absent"
                ) | {"pending_route": "complete"}
            await s.export_service.export_run(
                state["run_id"], Path(s.runtime_root) / "exports" / state["run_id"]
            )
        return {"phase": RunPhase.COMPLETE, "pending_route": "complete"}

    return transition


def _mark_terminal_failure(
    s: ServiceTransitions,
    state: ProductionState,
    kind: FailureKind,
    message: str,
    evidence: tuple[str, ...] = (),
) -> NoReturn:
    record = FailureRecord(
        failure_id=f"failure-{state['run_id']}-{state['state_version']}-terminal",
        experiment_id=state.get("current_experiment_id"),
        kind=kind,
        evidence_refs=tuple(
            item[:MAX_EVIDENCE_REF_LENGTH]
            for item in (evidence or (message,))[:MAX_EVIDENCE_REFS]
        ),
        repair_attempt=state["repair_attempts"],
    )
    if s.run_store is not None:
        s.run_store.put_failure(record, state["run_id"])
    raise TerminalLifecycleError(message)
