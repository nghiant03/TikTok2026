from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from tiktok2026.contracts import (
    AgentClient,
    AgentFailure,
    AgentRole,
    EvaluationContext,
    EvaluationRequest,
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
    ImplementationRequest,
    ImplementationResult,
    OrchestrationDecision,
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
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    WorktreeManager,
)
from tiktok2026.controller import Transition
from tiktok2026.graph.routes import (
    route_after_failure,
    route_after_orchestration,
    route_after_validation,
)
from tiktok2026.graph.state import ProductionState


class MissingAuthorityError(RuntimeError):
    """A transition cannot proceed without a persisted authority record."""

    terminal = True


class TerminalLifecycleError(RuntimeError):
    """A typed terminal failure that must not continue to export or complete."""

    terminal = True


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
    detail = json.dumps(
        {"kind": kind.value, "message": message, "evidence": evidence}, sort_keys=True
    )
    return {"terminal_reason": f"failure:{detail}", "pending_route": "persist_failure"}


def _failure_details(state: ProductionState) -> tuple[FailureKind, str, tuple[str, ...]]:
    reason = state.get("terminal_reason") or ""
    if reason.startswith("failure:"):
        try:
            payload = json.loads(reason.removeprefix("failure:"))
            return FailureKind(payload["kind"]), str(payload["message"]), tuple(payload["evidence"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return FailureKind.SCHEMA_MISMATCH, "failure classification was absent", ()


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
        request = ResearchRequest(
            request_id=f"orchestration-{state['run_id']}-{state['state_version']}",
            objective="orchestrate",
            resource_state=_resource_state(s),
        )
        response = await client.invoke(request)
        if isinstance(response, AgentFailure):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, response.message)
        if not isinstance(response, OrchestrationDecision):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid orchestration response")
        return {
            "orchestration_decision_id": response.decision_id,
            "pending_route": route_after_orchestration(response),
        }

    return transition


def _research(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.RESEARCH)
        if client is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "research role is not configured")
        request = ResearchRequest(
            request_id=f"research-{state['run_id']}-{state['state_version']}",
            objective="propose next experiment",
            resource_state=_resource_state(s),
        )
        response = await client.invoke(request)
        if isinstance(response, AgentFailure):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, response.message)
        if not isinstance(response, ResearchDecision) or response.experiment_spec is None:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "research returned no experiment spec"
            )
        spec = response.experiment_spec
        if s.run_store is not None:
            s.run_store.put_experiment(
                spec, "proposed", state["run_id"], f"proposed-{spec.experiment_id}"
            )
        return {
            "current_experiment_id": spec.experiment_id,
            "current_hypothesis_id": spec.hypothesis_id,
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
                spec.implementation_scope, spec.implementation_scope
            )
            if not decision.allowed:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, decision.reason)
        return {"pending_route": "proposal_validation"}

    return transition


def _proposal_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.VALIDATOR)
        if client is None:
            return {"pending_route": "create_worktree"}
        response = await client.invoke(
            ValidationRequest(
                request_id=f"proposal-validation-{state['run_id']}-{state['state_version']}",
                experiment_id=_exp_id(state),
                stage=ValidationStage.PROPOSAL,
            )
        )
        if isinstance(response, AgentFailure):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, response.message)
        if not isinstance(response, ValidationReport):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid proposal validation")
        return {
            "latest_validation_report_id": response.report_id,
            "pending_route": route_after_validation(state, response),
        }

    return transition


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
        return {"active_worktree_id": assignment.worktree_id, "pending_route": "implement"}

    return transition


def _implement(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        client = _agent(s, AgentRole.IMPLEMENTOR)
        if client is None:
            return _failure(
                state, FailureKind.SCHEMA_MISMATCH, "implementor role is not configured"
            )
        try:
            spec = _spec(s, state)
        except MissingAuthorityError as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        response = await client.invoke(
            ImplementationRequest(
                request_id=f"implementation-{state['run_id']}-{state['state_version']}",
                experiment_id=spec.experiment_id,
                allowed_scopes=spec.implementation_scope,
                capabilities=("scoped_read", "scoped_write", "diff", "checks"),
            )
        )
        if isinstance(response, AgentFailure):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, response.message)
        if not isinstance(response, ImplementationResult):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid implementation response")
        repository = getattr(client, "scoped_repository", None)
        if repository is not None:
            try:
                changed_files = tuple(repository.changed_files())
                if not changed_files or not repository.diff():
                    return _failure(
                        state, FailureKind.SCHEMA_MISMATCH, "implementation produced no real diff"
                    )
            except (OSError, PermissionError, ValueError, RuntimeError) as error:
                return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
            response = response.model_copy(update={"changed_files": changed_files})
        if s.run_store is not None:
            s.run_store.put_json(
                "implementation", response.experiment_id, response.model_dump_json()
            )
        return {"phase": RunPhase.IMPLEMENT, "pending_route": "diff_policy"}

    return transition


def _implementation_result(
    s: ServiceTransitions, state: ProductionState
) -> ImplementationResult | None:
    if s.run_store is None:
        return None
    values = s.run_store.list_json("implementation")
    for value in values:
        result = ImplementationResult.model_validate_json(value)
        if result.experiment_id == state.get("current_experiment_id"):
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
            return {"pending_route": route}
        response = await client.invoke(
            ValidationRequest(
                request_id=f"{stage.value}-validation-{state['run_id']}-{state['state_version']}",
                experiment_id=_exp_id(state),
                stage=stage,
            )
        )
        if isinstance(response, AgentFailure):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, response.message)
        if not isinstance(response, ValidationReport):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "invalid validation response")
        return {
            "latest_validation_report_id": response.report_id,
            "pending_route": route_after_validation(state, response),
        }

    return transition


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
            registration = s.worktree_manager.register_source(assignment, spec.implementation_scope)
            s.run_store.put_source_registration(registration)
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, str(error))
        return {"pending_route": "preflight"}

    return transition


def _preflight(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.run_store is not None and s.run_store.get_source_registration(_exp_id(state)) is None:
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "source registration was not found")
        return {"pending_route": "execute"}

    return transition


def _execute(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        if s.executor is None or s.run_store is None or s.dataset_root is None:
            return _failure(state, FailureKind.MISSING_PATH, "execution provenance is incomplete")
        try:
            spec = _spec(s, state)
            registration = s.run_store.get_source_registration(spec.experiment_id)
            assignment = s.run_store.get_worktree_assignment(spec.experiment_id)
            if registration is None or assignment is None:
                raise MissingAuthorityError("source or worktree authority is absent")
            reservation_id = f"reservation-{state['run_id']}-{spec.experiment_id}"
            reservation = ResourceReservation(
                reservation_id=reservation_id,
                run_id=state["run_id"],
                experiment_id=spec.experiment_id,
                gpu_hours=spec.predicted_gpu_hours,
                wall_seconds=float(s.default_timeout_seconds),
                tokens=0,
                disk_bytes=0,
            )
            if s.resource_accountant is not None and not s.resource_accountant.reserve(reservation):
                return _failure(state, FailureKind.DISK, "resource reservation was denied")
            execution_id = (
                f"execution-{state['run_id']}-{spec.experiment_id}-{state['state_version']}"
            )
            request = ExecutionRequest(
                run_id=state["run_id"],
                execution_id=execution_id,
                experiment_id=spec.experiment_id,
                source_commit=registration.source_commit,
                command=(
                    "python",
                    "-m",
                    "tiktok2026.experiment.train",
                    "--output-dir=/output",
                    f"--seed={_deterministic_seed(state['run_id'], spec.experiment_id)}",
                    f"--fidelity={spec.fidelity.value}",
                    f"--source-commit={registration.source_commit}",
                    f"--execution-id={execution_id}",
                ),
                image=s.docker_image or "",
                source_path=assignment.path,
                dataset_path=Path(s.dataset_root),
                output_path=Path(s.runtime_root or s.dataset_root) / "artifacts" / state["run_id"],
                timeout_seconds=s.default_timeout_seconds,
                memory_bytes=s.default_memory_bytes,
                cpus=s.default_cpus,
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
            return _failure(state, result.failure_kind, "execution failed", (result.execution_id,))
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
            registration = s.run_store.get_source_registration(_exp_id(state))
            manifest = s.run_store.get_dataset_manifest_identity()
            evaluator = s.run_store.get_evaluator_identity(s.evaluator_id)
            if execution is None or registration is None or manifest is None or evaluator is None:
                raise MissingAuthorityError("evaluation provenance record is absent")
            if execution.source_commit != registration.source_commit:
                raise MissingAuthorityError("execution source does not match registered source")
            checkpoint_id = execution.checkpoint_id
            if checkpoint_id is None:
                raise MissingAuthorityError("execution did not return a checkpoint identity")
            prediction = next(
                (s.run_store.get_prediction_artifact(item) for item in execution.artifact_ids), None
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
        except (MissingAuthorityError, ValueError) as error:
            return _failure(state, FailureKind.EVALUATOR_OUTPUT, str(error))
        return {
            "phase": RunPhase.EVALUATE,
            "latest_evaluation_result_id": result.evaluation_id,
            "pending_route": "result_validation",
        }

    return transition


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
                return _failure(state, FailureKind.SCHEMA_MISMATCH, response.message)
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
        if state["repair_attempts"] >= 2:
            return {"pending_route": "export"}
        if (
            s.policy_gate is not None
            and not s.policy_gate.can_repair(state["repair_attempts"]).allowed
        ):
            return _failure(state, FailureKind.SCHEMA_MISMATCH, "repair limit reached") | {
                "pending_route": "export"
            }
        return {"repair_attempts": state["repair_attempts"] + 1, "pending_route": "implement"}

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
        return {"pending_route": route_after_failure(state, record)}

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
        evidence_refs=evidence or (message,),
        repair_attempt=state["repair_attempts"],
    )
    if s.run_store is not None:
        s.run_store.put_failure(record, state["run_id"])
    raise TerminalLifecycleError(message)
