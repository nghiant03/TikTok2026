from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NoReturn, cast

from loguru import logger

from tiktok2026.contracts import (
    DEFAULT_IMPLEMENTATION_CRITERIA,
    AgentClient,
    AgentFailure,
    AgentRole,
    ControllerContext,
    CriterionAssessmentStatus,
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
    FullAttemptClaimRequest,
    ImplementationAttemptRecord,
    ImplementationCriterion,
    ImplementationCriterionAssessment,
    ImplementationCriterionId,
    ImplementationRequest,
    ImplementationResult,
    ImplementationValidationAuthority,
    OrchestrationDecision,
    OrchestrationRequest,
    OutcomeSummary,
    PolicyGate,
    ProposalSummary,
    ProvenanceRequest,
    ProvisionalFinalizationRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceAccountant,
    ResourceReservation,
    ResourceState,
    RunClosure,
    RunPhase,
    RunRecord,
    RunStore,
    ScoredObservationRequest,
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
from tiktok2026.policies.lifecycle import check_implementation_resource_estimate
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
CONTROLLER_READ_SCOPES = (
    "src/tiktok2026/contracts",
    "src/tiktok2026/benchmark/kuaireand_pure/manifest.py",
    "tests/experiment/test_training_contract.py",
    EXPERIMENT_ENTRYPOINT,
)
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
            criterion_id=blocker.criterion_id,
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
    min_pending_proposals: int = 3
    requires_run_baseline: bool = False
    plateau_epsilon: float = 0.002
    plateau_patience: int = 3


def _unresolved_blockers(
    s: ServiceTransitions, experiment_id: str
) -> tuple[ValidationBlocker, ...]:
    if s.run_store is None:
        raise MissingAuthorityError("validation ledger authority is absent")
    return s.run_store.get_unresolved_blockers(experiment_id)


def _has_validation_ledger(s: ServiceTransitions) -> bool:
    return s.run_store is not None


def _canonical_validation_report(
    report: ValidationReport, operation: ValidationOperationIdentity
) -> ValidationReport:
    report_id = f"validation-report-{operation.operation_id}"
    introduced_ids = {blocker.blocker_id for blocker in report.blockers}
    claimed_ids = {
        blocker_id
        for claim in report.resolution_claims
        for blocker_id in claim.blocker_ids
    }
    if introduced_ids & (set(report.resolves_blocker_ids) | claimed_ids):
        raise ValueError("a validation report cannot resolve a blocker it introduces")
    blockers = tuple(
        ValidationBlocker(
            blocker_id=validation_blocker_id(report_id, operation.stage, blocker.text),
            experiment_id=operation.experiment_id,
            stage=operation.stage,
            text=blocker.text,
            report_id=report_id,
            evidence_refs=blocker.evidence_refs,
            criterion_id=blocker.criterion_id,
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
    if (
        report.stage == ValidationStage.IMPLEMENTATION
        and _resource_feasibility_escalated(
            store, report.experiment_id, unresolved_blockers
        )
    ):
        route = "orchestrate"
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
    requires_run_baseline: bool = False,
    plateau_epsilon: float = 0.002,
    plateau_patience: int = 3,
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
        requires_run_baseline=requires_run_baseline,
        plateau_epsilon=plateau_epsilon,
        plateau_patience=plateau_patience,
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
            s.frontier_service.initialize(state["run_id"])
        return {"phase": RunPhase.BOOTSTRAP, "pending_route": "inspect"}

    return transition


def _inspect(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        _ = state, s
        return {"phase": RunPhase.RESEARCH, "pending_route": "orchestrate"}

    return transition


def _orchestrate(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is not None:
            closure = s.run_store.get_run_closure(state["run_id"])
            if closure is not None:
                return closure_updates_without_agents(s, state, closure)
        client = _agent(s, AgentRole.ORCHESTRATION)
        if client is None:
            return {"pending_route": "research"}
        evaluated_candidate_ready = _evaluated_candidate_ready(s, state)
        pending = _pending_proposals(s, state["run_id"])
        history = _outcome_history(s, state["run_id"])
        allowed_actions = (DecisionAction.RESEARCH,)
        if evaluated_candidate_ready:
            allowed_actions = (
                DecisionAction.RESEARCH,
                DecisionAction.REPLICATE,
                DecisionAction.INCREASE_FIDELITY,
                DecisionAction.REVISIT_BRANCH,
            )
        if pending:
            allowed_actions = (*allowed_actions, DecisionAction.IMPLEMENT)
        request = OrchestrationRequest(
            request_id=f"orchestration-{state['run_id']}-{state['state_version']}",
            run_id=state["run_id"],
            phase=state["phase"],
            allowed_actions=allowed_actions,
            resource_state=_resource_state(s),
            current_experiment_id=state.get("current_experiment_id"),
            latest_evaluation_result_id=state.get("latest_evaluation_result_id"),
            finalization_ready=False,
            failure_summary=state.get("terminal_reason"),
            controller_context=_controller_context(s),
            pending_proposals=pending,
            outcome_history=history,
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
        pending_ids = {p.experiment_id for p in pending}
        if response.action == DecisionAction.IMPLEMENT:
            if s.run_store is None or response.target_experiment_id not in pending_ids:
                return _failure(
                    state,
                    FailureKind.SCHEMA_MISMATCH,
                    "orchestration selected an unknown proposal",
                )
            chosen = s.run_store.get_experiment(response.target_experiment_id)
            if chosen is None:
                return _failure(
                    state,
                    FailureKind.SCHEMA_MISMATCH,
                    "selected proposal is not persisted",
                )
            return {
                "orchestration_decision_id": response.decision_id,
                "current_experiment_id": chosen.experiment_id,
                "current_hypothesis_id": chosen.hypothesis_id,
                "repair_attempts": 0,
                "terminal_reason": None,
                "pending_route": "proposal_policy",
            }
        if (
            response.target_experiment_id is not None
            and response.target_experiment_id != request.current_experiment_id
            and response.target_experiment_id not in pending_ids
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


def _evaluated_candidate_ready(s: ServiceTransitions, state: ProductionState) -> bool:
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
        if s.run_store is not None:
            closure = s.run_store.get_run_closure(state["run_id"])
            if closure is not None:
                return closure_updates_without_agents(s, state, closure)
        client = _agent(s, AgentRole.RESEARCH)
        if client is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "research role is not configured")
        objective = "propose next experiment"
        unresolved_context: tuple[ValidationBlockerContext, ...] = ()
        prior_experiment_id = state.get("current_experiment_id")
        is_proposal_repair = bool(state.get("terminal_reason") and prior_experiment_id)
        if state.get("terminal_reason"):
            _, message, _ = _failure_details(state)
            objective += f"; address validator feedback: {message}"
            if prior_experiment_id is not None:
                try:
                    unresolved_context = _blocker_context(
                        _unresolved_blockers(s, prior_experiment_id)
                    )
                except MissingAuthorityError as error:
                    return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        request = ResearchRequest(
            request_id=f"research-{state['run_id']}-{state['state_version']}",
            objective=objective,
            resource_state=_resource_state(s),
            parent_experiment_id=(prior_experiment_id if is_proposal_repair else None),
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
        is_new_experiment = spec.experiment_id != prior_experiment_id
        if s.run_store is not None:
            existing = s.run_store.get_experiment(spec.experiment_id)
            if existing is not None and existing != spec:
                return _failure(
                    state,
                    FailureKind.SCHEMA_MISMATCH,
                    "experiment identities are immutable; return the revised proposal with a "
                    "new experiment_id and set parent_experiment_id to the proposal being repaired",
                )
            if (
                is_proposal_repair
                and is_new_experiment
                and spec.parent_experiment_id != prior_experiment_id
            ):
                return _failure(
                    state,
                    FailureKind.SCHEMA_MISMATCH,
                    "a revised proposal must set parent_experiment_id to the proposal "
                    "being repaired",
                )
            s.run_store.put_experiment(
                spec, "proposed", state["run_id"], f"proposed-{spec.experiment_id}"
            )
        return {
            "phase": RunPhase.RESEARCH,
            "current_experiment_id": spec.experiment_id,
            "current_hypothesis_id": spec.hypothesis_id,
            "repair_attempts": (
                state["repair_attempts"]
                if is_proposal_repair
                else 0 if is_new_experiment else state["repair_attempts"]
            ),
            "terminal_reason": None,
            "pending_route": (
                "orchestrate"
                if getattr(s, "min_pending_proposals", 1) > 1
                else "proposal_policy"
            ),
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


def _execution_resources(s: ServiceTransitions) -> dict[str, object]:
    return {
        "timeout_seconds": s.default_timeout_seconds,
        "memory_bytes": s.default_memory_bytes,
        "cpus": s.default_cpus,
        "gpu_count": s.default_gpu_count,
    }


def _implementation_criterion_requirements() -> tuple[ImplementationCriterion, ...]:
    return tuple(
        ImplementationCriterion(criterion_id=criterion, description=criterion.value)
        for criterion in DEFAULT_IMPLEMENTATION_CRITERIA
    )


def _bind_worktree(
    binder: Callable[..., object],
    path: Path,
    write_scopes: tuple[str, ...],
) -> None:
    """Bind modern clients with separate reads while tolerating legacy fakes."""
    try:
        supports_read_scopes = "read_scopes" in inspect.signature(binder).parameters
    except (TypeError, ValueError):
        supports_read_scopes = False
    if supports_read_scopes:
        binder(path, write_scopes, read_scopes=CONTROLLER_READ_SCOPES)
    else:
        binder(path, write_scopes)


def _complete_implementation_validation(
    report: ValidationReport,
    unresolved_before: tuple[ValidationBlocker, ...],
) -> ValidationReport:
    """Make the controller's criterion matrix total before persisting a report."""
    assessments = {str(item.criterion_id): item for item in report.criterion_assessments}
    blockers = list(report.blockers)
    prior_by_criterion = {
        str(blocker.criterion_id): blocker
        for blocker in unresolved_before
        if blocker.criterion_id is not None
    }
    claimed_ids = {
        blocker_id
        for claim in report.resolution_claims
        for blocker_id in claim.blocker_ids
    }
    for criterion in DEFAULT_IMPLEMENTATION_CRITERIA:
        key = str(criterion)
        assessment = assessments.get(key)
        if assessment is None:
            assessment = ImplementationCriterionAssessment(
                criterion_id=criterion,
                status=CriterionAssessmentStatus.FAIL,
                details=f"criterion was not assessed: {criterion.value}",
            )
            assessments[key] = assessment
        if assessment.status not in (
            CriterionAssessmentStatus.FAIL,
            CriterionAssessmentStatus.PARTIAL,
        ):
            continue
        has_blocker = any(
            blocker.criterion_id is not None and str(blocker.criterion_id) == key
            for blocker in blockers
        )
        # A partial claim may address an older, stable blocker.  Do not create
        # the same blocker in this report or turn that claim into self-resolution.
        prior = prior_by_criterion.get(key)
        if not has_blocker and (prior is None or prior.blocker_id not in claimed_ids):
            blockers.append(
                ValidationBlocker(
                    blocker_id="",
                    experiment_id=report.experiment_id,
                    stage=report.stage,
                    text=assessment.details
                    or f"implementation criterion failed: {criterion.value}",
                    report_id=report.report_id,
                    evidence_refs=assessment.evidence_refs,
                    criterion_id=criterion,
                )
            )
    return report.model_copy(
        update={
            "criterion_assessments": tuple(assessments.values()),
            "blockers": tuple(blockers),
        }
    )


def _resource_feasibility_escalated(
    store: RunStore,
    experiment_id: str,
    unresolved_blockers: tuple[ValidationBlocker, ...],
) -> bool:
    getter = getattr(store, "get_criterion_repeat_count", None)
    if not callable(getter):
        return False
    has_resource_blocker = any(
        blocker.criterion_id == ImplementationCriterionId.RESOURCE_FEASIBILITY
        for blocker in unresolved_blockers
    )
    typed_getter = cast(Callable[[str, ImplementationCriterionId], int], getter)
    return has_resource_blocker and int(
        typed_getter(experiment_id, ImplementationCriterionId.RESOURCE_FEASIBILITY)
    ) >= 2

def _pending_proposals(s: ServiceTransitions, run_id: str) -> tuple[ProposalSummary, ...]:
    if s.run_store is None:
        return ()
    lister = getattr(s.run_store, "list_experiments_by_status", None)
    if lister is None:
        return ()
    attempted: set[str] = set()
    obs_lister = getattr(s.run_store, "list_scored_observations", None)
    if obs_lister is not None:
        attempted = {o.experiment_id for o in obs_lister(run_id)}
    return tuple(
        ProposalSummary(
            experiment_id=x.experiment_id,
            hypothesis=x.hypothesis,
            mechanism=x.mechanism,
            implementation_scope=x.implementation_scope,
            parent_experiment_id=x.parent_experiment_id,
        )
        for x in lister(run_id, "proposed")
        if x.experiment_id not in attempted
    )


def _outcome_history(s: ServiceTransitions, run_id: str) -> tuple[OutcomeSummary, ...]:
    if s.run_store is None:
        return ()
    obs_lister = getattr(s.run_store, "list_scored_observations", None)
    if obs_lister is None:
        return ()
    out: list[OutcomeSummary] = []
    scores: dict[str, float] = {}
    for o in obs_lister(run_id):
        spec = s.run_store.get_experiment(o.experiment_id)
        scores[o.experiment_id] = o.primary_score
        parent = spec.parent_experiment_id if spec is not None else None
        out.append(
            OutcomeSummary(
                experiment_id=o.experiment_id,
                hypothesis=spec.hypothesis if spec is not None else "",
                primary_score=o.primary_score,
                delta_vs_parent=(
                    round(o.primary_score - scores[parent], 6)
                    if parent is not None and parent in scores
                    else None
                ),
            )
        )
    return tuple(out)

def _controller_context(
    s: ServiceTransitions,
    *,
    include_experiment_registry: bool = True,
    exclude_experiment_id: str | None = None,
) -> ControllerContext:
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
            s.run_store.get_experiment_registry(
                exclude_experiment_id=exclude_experiment_id
            )
            if s.run_store is not None
            else None
        )
        if include_experiment_registry
        else None,
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
        estimate = spec.implementation_resource_estimate
        if estimate is not None:
            decision = check_implementation_resource_estimate(
                estimate,
                execution_timeout_seconds=s.default_timeout_seconds,
                execution_memory_bytes=s.default_memory_bytes,
                resource_state=_resource_state(s),
            )
            if not decision.allowed:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, decision.reason)
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
            unresolved = _unresolved_blockers(s, spec.experiment_id)
        except MissingAuthorityError as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        subject: dict[str, object] = {
            "experiment_spec": spec.model_dump(mode="json"),
            "controller_context": _controller_context(
                s, exclude_experiment_id=spec.experiment_id
            ).model_dump(mode="json"),
            "execution_resources": _execution_resources(s),
            "implementation_resource_estimate": (
                spec.implementation_resource_estimate.model_dump(mode="json")
                if spec.implementation_resource_estimate is not None
                else None
            ),
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
    if (
        report.stage == ValidationStage.IMPLEMENTATION
        and _resource_feasibility_escalated(
            store, report.experiment_id, unresolved_blockers
        )
    ):
        route = "orchestrate"
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
                _bind_worktree(binder, assignment.path, spec.implementation_scope)
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
            if repository is not None:
                prior_diff_sha256 = hashlib.sha256(repository.diff().encode()).hexdigest()
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
                read_scopes=CONTROLLER_READ_SCOPES,
                implementation_criteria=DEFAULT_IMPLEMENTATION_CRITERIA,
                criterion_requirements=_implementation_criterion_requirements(),
                capabilities=("scoped_read", "scoped_write", "diff", "checks"),
                repair_feedback=(
                    _failure_details(state)[1] if state["repair_attempts"] > 0 else None
                ),
                unresolved_blocker_ids=unresolved_blocker_ids,
                unresolved_blockers=unresolved_blocker_context,
                source_context={},
                base_source_context={},
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
            subject = _validation_subject(s, state, stage)
            unresolved_before = _unresolved_blockers(s, _exp_id(state))
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
                _bind_worktree(
                    binder, assignment.path, _spec(s, state).implementation_scope
                )
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
        if stage == ValidationStage.IMPLEMENTATION:
            response = _complete_implementation_validation(response, unresolved_before)
        return _validation_updates(s, state, response, operation, subject)

    return transition


def _validation_subject(
    s: ServiceTransitions, state: ProductionState, stage: ValidationStage
) -> dict[str, object]:
    controller_context = _controller_context(
        s, include_experiment_registry=stage != ValidationStage.IMPLEMENTATION
    ).model_dump(mode="json")
    if stage == ValidationStage.IMPLEMENTATION:
        # The registry is proposal/research context, not implementation evidence.
        controller_context.pop("experiment_registry", None)
    subject: dict[str, object] = {
        "experiment_spec": _spec(s, state).model_dump(mode="json"),
        "controller_context": controller_context,
        "unresolved_blockers": [
            item.model_dump(mode="json")
            for item in _blocker_context(_unresolved_blockers(s, _exp_id(state)))
        ],
    }
    if stage == ValidationStage.IMPLEMENTATION:
        implementation = _implementation_result(s, state)
        if implementation is None:
            raise MissingAuthorityError("implementation result is absent")
        subject["implementation_result"] = implementation.model_dump(
            mode="json", exclude={"edits"}
        )
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
        subject["execution_resources"] = _execution_resources(s)
        subject["implementation_criteria"] = [
            criterion.value for criterion in DEFAULT_IMPLEMENTATION_CRITERIA
        ]
        subject["criterion_requirements"] = [
            item.model_dump(mode="json") for item in _implementation_criterion_requirements()
        ]
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
        subject["execution_result"] = execution.model_dump(
            mode="json", exclude={"dataset_valid_rows"}
        )
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
    output: Path | None = None
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
        if output is not None:
            _remove_empty_execution_output(output)
        raise ValueError("execution output directory could not be prepared") from error
    return output


def _remove_empty_execution_output(output_path: Path) -> None:
    try:
        if output_path.is_dir() and not output_path.is_symlink() and not any(output_path.iterdir()):
            output_path.rmdir()
    except OSError:
        logger.warning("could not remove empty execution output path {}", output_path)


def _execute(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if (
            s.executor is None
            or s.run_store is None
            or s.dataset_root is None
            or s.runtime_root is None
            or s.resource_accountant is None
        ):
            return _failure(state, FailureKind.MISSING_PATH, "execution provenance is incomplete")
        reservation_id = ""
        reservation_held = False
        dispatch_started = False
        settlement_replay = False
        output_path: Path | None = None
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
            if result is not None:
                claim = s.run_store.claim_full_attempt(
                    FullAttemptClaimRequest(
                        attempt_id=f"attempt-{execution_id}",
                        execution_id=execution_id,
                        run_id=state["run_id"],
                        experiment_id=spec.experiment_id,
                        source_registration_id=registration.registration_id,
                        source_commit=registration.source_commit,
                    )
                )
                if claim is None:
                    return _failure(
                        state,
                        FailureKind.SCIENTIFIC_NON_IMPROVEMENT,
                        "persisted execution cannot be adopted because the attempt cap "
                        "is exhausted",
                    )
                reservation_id = f"reservation-{execution_id}"
                reservation = ResourceReservation(
                    reservation_id=reservation_id,
                    run_id=state["run_id"],
                    experiment_id=spec.experiment_id,
                    gpu_hours=max(
                        spec.predicted_gpu_hours,
                        float(s.default_timeout_seconds + 5)
                        * s.default_gpu_count
                        / 3600.0,
                    ),
                    wall_seconds=float(s.default_timeout_seconds + 5),
                    tokens=0,
                    disk_bytes=0,
                )
                settlement_replay = True
                if not s.resource_accountant.reserve(reservation):
                    raise MissingAuthorityError(
                        "persisted execution resource reservation is unavailable"
                    )
                if not s.resource_accountant.consume(
                    reservation_id,
                    gpu_hours=result.gpu_hours,
                    wall_seconds=result.elapsed_seconds,
                    tokens=0,
                    disk_bytes=0,
                ):
                    raise MissingAuthorityError(
                        "persisted execution resource settlement is unavailable"
                    )
                if not s.resource_accountant.reconcile(
                    reservation_id,
                    gpu_hours=result.gpu_hours,
                    wall_seconds=result.elapsed_seconds,
                    tokens=0,
                    disk_bytes=0,
                ):
                    raise MissingAuthorityError(
                        "persisted execution resource reconciliation is unavailable"
                    )
            else:
                existing_claim = next(
                    (
                        claim
                        for claim in s.run_store.list_full_attempt_claims(state["run_id"])
                        if claim.execution_id == execution_id
                    ),
                    None,
                )
                if existing_claim is not None:
                    return _failure(
                        state,
                        FailureKind.SCHEMA_MISMATCH,
                        "full execution is already claimed but its result is absent",
                        (execution_id,),
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
                if not s.resource_accountant.reserve(reservation):
                    return _failure(state, FailureKind.DISK, "resource reservation was denied")
                reservation_held = True
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
                claim = s.run_store.claim_full_attempt(
                    FullAttemptClaimRequest(
                        attempt_id=f"attempt-{execution_id}",
                        execution_id=execution_id,
                        run_id=state["run_id"],
                        experiment_id=spec.experiment_id,
                        source_registration_id=registration.registration_id,
                        source_commit=registration.source_commit,
                    )
                )
                if claim is None:
                    s.resource_accountant.release(reservation_id)
                    reservation_held = False
                    _remove_empty_execution_output(output_path)
                    return _failure(
                        state,
                        FailureKind.SCIENTIFIC_NON_IMPROVEMENT,
                        "full attempt cap exhausted before dispatch",
                    )
                dispatch_started = True
                result = await s.executor.execute(request)
                s.run_store.put_execution_result(result)
                if not s.resource_accountant.consume(
                    reservation_id,
                    gpu_hours=result.gpu_hours,
                    wall_seconds=result.elapsed_seconds,
                    tokens=0,
                    disk_bytes=0,
                ):
                    raise MissingAuthorityError("execution resource settlement is unavailable")
                if not s.resource_accountant.reconcile(
                    reservation_id,
                    gpu_hours=result.gpu_hours,
                    wall_seconds=result.elapsed_seconds,
                    tokens=0,
                    disk_bytes=0,
                ):
                    raise MissingAuthorityError(
                        "execution resource reconciliation is unavailable"
                    )
                reservation_held = False
        except Exception as error:
            if reservation_held and not dispatch_started and not settlement_replay:
                s.resource_accountant.release(reservation_id)
            if output_path is not None and not dispatch_started and not settlement_replay:
                _remove_empty_execution_output(output_path)
            if dispatch_started or settlement_replay:
                raise
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
            prediction_record = s.run_store.get_artifact(prediction.artifact_id)
            checkpoint_artifacts = tuple(
                artifact
                for artifact_id in execution.artifact_ids
                if (artifact := s.run_store.get_artifact(artifact_id)) is not None
                and artifact.kind == "checkpoint"
            )
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
            if s.requires_run_baseline:
                binding = s.run_store.get_run_baseline(state["run_id"])
                if binding is None:
                    raise MissingAuthorityError(
                        f"run baseline authority binding is absent for production run "
                        f"{state['run_id']}"
                    )
                if (
                    binding.dataset_manifest_id != context.dataset_manifest_id
                    or binding.dataset_manifest_sha256 != context.dataset_manifest_sha256
                    or binding.evaluator_id != context.evaluator_id
                    or binding.evaluator_sha256 != context.evaluator_sha256
                    or binding.split != context.split
                ):
                    raise MissingAuthorityError(
                        f"run baseline binding does not match evaluation {context.evaluation_id}"
                    )
            result = s.evaluator.evaluate(
                EvaluationRequest(evaluation_id=context.evaluation_id, context=context)
            )
            if s.requires_run_baseline and {
                metric.name for metric in result.metrics
            } != {"GAUC", "nDCG@5"}:
                raise MissingAuthorityError(
                    "production evaluation does not use the current judging metric pair"
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
    if set(metrics) != {"GAUC", "nDCG@5"} or len(metrics) != 2:
        if s.requires_run_baseline:
            raise MissingAuthorityError(
                "production evaluation does not use the current judging metric pair"
            )
        logger.info(
            "Historical evaluation metrics experiment_id={} evaluation_id={} "
            "evaluator_id={} are not comparable to the current judging contract",
            result.experiment_id,
            result.evaluation_id,
            result.evaluator_artifact_id,
        )
        return
    gauc = metrics["GAUC"]
    ndcg = metrics["nDCG@5"]
    # Production evaluations are compared only against the immutable binding
    # selected for their run.  In particular, never select the first calibration
    # in the database: that can silently compare different dataset/evaluator
    # identities after a resume or an evaluator upgrade.
    if (
        s.requires_run_baseline
        and result.run_id is not None
        and s.run_store is not None
    ):
        binding = s.run_store.get_run_baseline(result.run_id)
        if binding is None:
            raise MissingAuthorityError(
                f"run baseline authority binding is absent for production run {result.run_id}"
            )
        if (
            binding.dataset_manifest_id != result.dataset_manifest_id
            or binding.dataset_manifest_sha256 != result.dataset_manifest_sha256
            or binding.evaluator_id != result.evaluator_artifact_id
            or binding.evaluator_sha256 != result.evaluator_sha256
            or binding.split != result.split
        ):
            raise MissingAuthorityError(
                f"run baseline binding does not match evaluation {result.evaluation_id}"
            )
        baseline_calibration = s.run_store.get_baseline_calibration(binding.calibration_id)
        if (
            baseline_calibration is None
            or baseline_calibration.evaluation.evaluation_id != binding.baseline_evaluation_id
        ):
            raise MissingAuthorityError(
                f"bound baseline calibration is unavailable for run {result.run_id}"
            )
        calibration_identity_matches = (
            baseline_calibration.dataset_manifest_id == binding.dataset_manifest_id
            and baseline_calibration.dataset_manifest_sha256 == binding.dataset_manifest_sha256
            and baseline_calibration.evaluator_id == binding.evaluator_id
            and baseline_calibration.evaluator_sha256 == binding.evaluator_sha256
            and baseline_calibration.split == binding.split
        )
        baseline_metrics = {metric.name: metric.value for metric in binding.metrics}
        if (
            not calibration_identity_matches
            or set(baseline_metrics) != {"GAUC", "nDCG@5"}
            or {
                metric.name: metric.value for metric in baseline_calibration.evaluation.metrics
            }
            != baseline_metrics
        ):
            raise MissingAuthorityError(
                f"bound baseline calibration does not match run {result.run_id}"
            )
        baseline_gauc = baseline_metrics["GAUC"]
        baseline_ndcg = baseline_metrics["nDCG@5"]
        logger.info(
            "Provisional pipeline metrics experiment_id={} evaluation_id={} "
            "GAUC={:.6f} nDCG@5={:.6f} composite={:.6f} "
            "baseline=starter_kit_fm baseline_calibration_id={} "
            "baseline_GAUC={:.6f} baseline_nDCG@5={:.6f} "
            "delta_GAUC={:+.6f} delta_nDCG@5={:+.6f} delta_composite={:+.6f}",
            result.experiment_id,
            result.evaluation_id,
            gauc,
            ndcg,
            result.validation_score,
            binding.calibration_id,
            baseline_gauc,
            baseline_ndcg,
            gauc - baseline_gauc,
            ndcg - baseline_ndcg,
            result.validation_score - (baseline_gauc + baseline_ndcg) / 2.0,
        )
        return

    if s.requires_run_baseline:
        raise MissingAuthorityError("production run baseline authority is unavailable")

    # Synthetic and unit-test evaluators may retain the prior-champion diagnostic path.
    calibrations = s.run_store.list_baseline_calibrations() if s.run_store is not None else ()
    matching_calibrations = tuple(
        calibration
        for calibration in calibrations
        if calibration.dataset_manifest_sha256 == result.dataset_manifest_sha256
        and calibration.evaluator_id == result.evaluator_artifact_id
        and calibration.evaluator_sha256 == result.evaluator_sha256
        and calibration.split == result.split
        and {metric.name for metric in calibration.evaluation.metrics}
        == {"GAUC", "nDCG@5"}
    )
    if matching_calibrations:
        baseline = matching_calibrations[0]
        baseline_metrics = {metric.name: metric.value for metric in baseline.evaluation.metrics}
        baseline_gauc = baseline_metrics["GAUC"]
        baseline_ndcg = baseline_metrics["nDCG@5"]
        logger.info(
            "Provisional pipeline metrics experiment_id={} evaluation_id={} "
            "GAUC={:.6f} nDCG@5={:.6f} composite={:.6f} "
            "baseline=starter_kit_fm baseline_calibration_id={} "
            "baseline_GAUC={:.6f} baseline_nDCG@5={:.6f} "
            "delta_GAUC={:+.6f} delta_nDCG@5={:+.6f} delta_composite={:+.6f}",
            result.experiment_id,
            result.evaluation_id,
            gauc,
            ndcg,
            result.validation_score,
            baseline.calibration_id,
            baseline_gauc,
            baseline_ndcg,
            gauc - baseline_gauc,
            ndcg - baseline_ndcg,
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
        and candidate.evaluator_artifact_id == result.evaluator_artifact_id
        and candidate.evaluator_sha256 == result.evaluator_sha256
        and candidate.split == result.split
        and candidate.validity == result.validity
        and {metric.name for metric in candidate.metrics} == {"GAUC", "nDCG@5"}
    )
    if not comparable:
        logger.info(
            "Provisional pipeline metrics experiment_id={} evaluation_id={} "
            "GAUC={:.6f} nDCG@5={:.6f} composite={:.6f} baseline=unavailable",
            result.experiment_id,
            result.evaluation_id,
            gauc,
            ndcg,
            result.validation_score,
        )
        return
    baseline = max(comparable, key=lambda candidate: candidate.validation_score)
    baseline_metrics = {metric.name: metric.value for metric in baseline.metrics}
    baseline_gauc = baseline_metrics["GAUC"]
    baseline_ndcg = baseline_metrics["nDCG@5"]
    logger.info(
        "Provisional pipeline metrics experiment_id={} evaluation_id={} "
        "GAUC={:.6f} nDCG@5={:.6f} composite={:.6f} "
        "baseline=prior_champion baseline_evaluation_id={} "
        "baseline_GAUC={:.6f} baseline_nDCG@5={:.6f} "
        "delta_GAUC={:+.6f} delta_nDCG@5={:+.6f} delta_composite={:+.6f}",
        result.experiment_id,
        result.evaluation_id,
        gauc,
        ndcg,
        result.validation_score,
        baseline.evaluation_id,
        baseline_gauc,
        baseline_ndcg,
        gauc - baseline_gauc,
        ndcg - baseline_ndcg,
        result.validation_score - baseline.validation_score,
    )


def _result_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        return await _validation(s, state, ValidationStage.RESULT, "persist")(state)

    return transition


def _interpret(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is not None and s.run_store.get_run_closure(state["run_id"]) is not None:
            return {"pending_route": "finalize"}
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
        return {"pending_route": "orchestrate"}

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


def _closure_champion_updates(
    s: ServiceTransitions, state: ProductionState, closure: RunClosure
) -> dict[str, object]:
    if s.run_store is None or closure.champion is None:
        raise MissingAuthorityError("run closure has no eligible champion")
    champion = closure.champion
    observation = s.run_store.get_scored_observation(champion.observation_id)
    if observation is None:
        raise MissingAuthorityError("run closure champion observation is absent")
    spec = s.run_store.get_experiment(observation.experiment_id)
    evaluation = s.run_store.get_evaluation_result(observation.evaluation_id)
    source = s.run_store.get_source_registration_by_id(f"source-{observation.source_commit}")
    attempt = next(
        (
            item
            for item in s.run_store.list_full_attempt_claims(state["run_id"])
            if item.attempt_id == champion.attempt_id
        ),
        None,
    )
    if spec is None or evaluation is None or source is None or attempt is None:
        raise MissingAuthorityError("run closure champion authority is incomplete")
    if (
        observation.run_id != state["run_id"]
        or observation.experiment_id != spec.experiment_id
        or observation.attempt_id != champion.attempt_id
        or observation.execution_id != champion.execution_id
        or observation.evaluation_id != champion.evaluation_id
        or observation.checkpoint_id != champion.checkpoint_id
        or observation.source_commit != champion.source_commit
        or observation.primary_score != champion.primary_score
        or attempt.execution_id != champion.execution_id
        or attempt.run_id != state["run_id"]
        or attempt.experiment_id != observation.experiment_id
        or attempt.source_registration_id != source.registration_id
        or attempt.source_commit != source.source_commit
        or attempt.attempt_sequence != champion.attempt_sequence
        or source.run_id != state["run_id"]
        or source.experiment_id != spec.experiment_id
        or source.source_commit != observation.source_commit
        or not source.eligible
        or evaluation.evaluation_id != observation.evaluation_id
        or evaluation.experiment_id != observation.experiment_id
        or evaluation.run_id != state["run_id"]
        or evaluation.execution_id != observation.execution_id
        or evaluation.checkpoint_id != observation.checkpoint_id
        or evaluation.source_commit != observation.source_commit
    ):
        raise MissingAuthorityError("run closure champion provenance is inconsistent")
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
    return {
        "phase": RunPhase.FINALIZE,
        "current_experiment_id": spec.experiment_id,
        "current_hypothesis_id": spec.hypothesis_id,
        "latest_execution_result_id": champion.execution_id,
        "latest_evaluation_result_id": evaluation.evaluation_id,
        "latest_validation_report_id": observation.validation_report_id,
        "terminal_reason": closure.reason,
        "pending_route": "finalize",
    }


def closure_updates_without_agents(
    s: ServiceTransitions, state: ProductionState, closure: RunClosure
) -> dict[str, object]:
    """Route an already authoritative closure without consulting a model."""
    if closure.champion is not None:
        return _closure_champion_updates(s, state, closure)
    if s.run_store is not None:
        s.run_store.put_run(
            RunRecord(run_id=state["run_id"], status="failed"),
            f"{state['run_id']}-terminal-no-observation",
            expected_predecessor=f"{state['run_id']}-active",
        )
    return {
        "phase": RunPhase.COMPLETE,
        "terminal_reason": f"{closure.reason}:no_eligible_observation",
        "pending_route": "complete",
    }


def _update_frontier(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        evaluation_id = state.get("latest_evaluation_result_id")
        report_id = state.get("latest_validation_report_id")
        execution_id = state.get("latest_execution_result_id")
        if s.run_store is None or not evaluation_id or not report_id or not execution_id:
            return _failure(
                state,
                FailureKind.SCIENTIFIC_NON_IMPROVEMENT,
                "scored observation evidence is absent",
            )
        result = s.run_store.get_evaluation_result(evaluation_id)
        execution = s.run_store.get_execution_result(execution_id)
        report = s.run_store.get_validation_report(report_id)
        attempts = s.run_store.list_full_attempt_claims(state["run_id"])
        attempt = next((item for item in attempts if item.execution_id == execution_id), None)
        if (
            result is None
            or execution is None
            or report is None
            or attempt is None
            or result.source_commit is None
            or result.dataset_manifest_id is None
            or result.dataset_manifest_sha256 is None
            or result.split != "valid"
            or result.validity not in {"provisional", "official"}
        ):
            return _failure(
                state,
                FailureKind.SCIENTIFIC_NON_IMPROVEMENT,
                "scored observation authority is absent",
            )
        try:
            observation = s.run_store.put_scored_observation(
                ScoredObservationRequest(
                    observation_id=f"observation-{result.evaluation_id}",
                    run_id=state["run_id"],
                    experiment_id=result.experiment_id,
                    attempt_id=attempt.attempt_id,
                    execution_id=execution.execution_id,
                    evaluation_id=result.evaluation_id,
                    checkpoint_id=result.checkpoint_id,
                    source_commit=result.source_commit,
                    evaluator_id=result.evaluator_artifact_id,
                    evaluator_sha256=result.evaluator_sha256,
                    dataset_manifest_id=result.dataset_manifest_id,
                    dataset_manifest_sha256=result.dataset_manifest_sha256,
                    split="valid",
                    validity=cast(Literal["provisional", "official"], result.validity),
                    primary_score=result.validation_score,
                    validation_report_id=report.report_id,
                    validation_evidence_refs=report.evidence_refs,
                )
            )
            del observation
            epsilon = (
                s.frontier_service.epsilon
                if s.frontier_service is not None
                else s.plateau_epsilon
            )
            patience = (
                s.frontier_service.patience
                if s.frontier_service is not None
                else s.plateau_patience
            )
            closure = s.run_store.close_run_if_ready(
                state["run_id"],
                epsilon=epsilon,
                patience=patience,
            )
            # Keep the synthetic diagnostic mirror for legacy fixture exports;
            # production routing is determined only by the typed closure above.
            if not s.requires_run_baseline and s.frontier_service is not None:
                s.frontier_service.update(_exp_id(state), result.validation_score)
            if closure is None:
                return {"pending_route": "orchestrate"}
            return _closure_champion_updates(s, state, closure)
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))

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
        if s.run_store is None:
            closure = None
        else:
            s.run_store.put_failure(record, state["run_id"])
            closure = s.run_store.close_run_if_ready(state["run_id"], after_failure=True)
        route = route_after_failure(state, record, s.max_repairs)
        champion_observation = (
            s.run_store.get_scored_observation(closure.champion.observation_id)
            if closure is not None and closure.champion is not None and s.run_store is not None
            else None
        )
        champion_experiment_id = (
            champion_observation.experiment_id if champion_observation is not None else None
        )
        if (route == "orchestrate" or closure is not None) and s.run_store is not None:
            try:
                spec = _spec(s, state)
                if spec.experiment_id != champion_experiment_id:
                    s.run_store.put_experiment(
                        spec,
                        "failed",
                        state["run_id"],
                        f"failed-{spec.experiment_id}",
                        expected_predecessor=f"proposed-{spec.experiment_id}",
                    )
            except (MissingAuthorityError, ValueError) as error:
                raise TerminalLifecycleError(str(error)) from error
        if closure is not None:
            try:
                return closure_updates_without_agents(s, state, closure)
            except (MissingAuthorityError, ValueError) as error:
                raise TerminalLifecycleError(str(error)) from error
        if route == "terminal":
            raise TerminalLifecycleError(message)
        updates: dict[str, object] = {"pending_route": route}
        if route == "orchestrate":
            updates |= {
                "current_experiment_id": None,
                "current_hypothesis_id": None,
                "repair_attempts": 0,
            }
        return updates

    return transition


def _finalize(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "run closure authority is absent")
        closure = s.run_store.get_run_closure(state["run_id"])
        if closure is None:
            return {
                "terminal_reason": None,
                "phase": RunPhase.PERSIST,
                "pending_route": "orchestrate",
            }
        if closure.champion is None:
            _mark_terminal_failure(
                s,
                state,
                FailureKind.SCHEMA_MISMATCH,
                "run closure has no eligible champion",
            )
        try:
            if s.bundle_service is None:
                raise MissingAuthorityError("finalization bundle authority is absent")
            champion = closure.champion
            observation = s.run_store.get_scored_observation(champion.observation_id)
            if observation is None:
                raise MissingAuthorityError("finalization provenance is absent")
            spec = s.run_store.get_experiment(observation.experiment_id)
            evaluation = s.run_store.get_evaluation_result(observation.evaluation_id)
            registration = s.run_store.get_source_registration_by_id(
                f"source-{observation.source_commit}"
            )
            if spec is None or registration is None or evaluation is None:
                raise MissingAuthorityError("finalization provenance is absent")
            bundle = s.bundle_service.create(
                FinalizationBundleRequest(
                    run_id=state["run_id"],
                    experiment_id=spec.experiment_id,
                    source_commit=registration.source_commit,
                    checkpoint_id=evaluation.checkpoint_id,
                    evaluation_id=evaluation.evaluation_id,
                    evaluator_id=evaluation.evaluator_artifact_id,
                )
            )
            s.run_store.persist_provisional_finalization(
                ProvisionalFinalizationRequest(
                    finalization_id=f"finalization-{state['run_id']}",
                    run_id=state["run_id"],
                    experiment_id=spec.experiment_id,
                    source_commit=registration.source_commit,
                    checkpoint_id=evaluation.checkpoint_id,
                    evaluation_id=evaluation.evaluation_id,
                    bundle_artifact_id=bundle.artifact_id,
                    evaluator_id=evaluation.evaluator_artifact_id,
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
