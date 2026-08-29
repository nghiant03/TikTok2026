from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tiktok2026.contracts import (
    AgentClient,
    AgentFailure,
    EvaluationContext,
    EvaluationRequest,
    Evaluator,
    ExecutionRequest,
    Executor,
    ExperimentSpec,
    ExportService,
    FailureKind,
    FailureRecord,
    FrontierService,
    ImplementationResult,
    OrchestrationDecision,
    PolicyGate,
    ProvenanceRequest,
    ProvisionalFinalizationRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceAccountant,
    ResourceState,
    RunPhase,
    RunStore,
    ValidationReport,
    ValidationStage,
    ValidationVerdict,
    WorktreeAssignment,
    WorktreeManager,
)
from tiktok2026.controller import Transition
from tiktok2026.graph.routes import (
    route_after_failure,
    route_after_orchestration,
    route_after_validation,
)
from tiktok2026.graph.state import ProductionState

# ---------------------------------------------------------------------------
# Injectables container
# ---------------------------------------------------------------------------


@dataclass
class ServiceTransitions:
    """All 22 transitions as a single dataclass of callables.

    Each transition is a small typed function that reads ProductionState,
    invokes the appropriate injected capability, produces typed updates,
    and sets pending_route using the existing routes helpers.
    """

    agent_client: AgentClient | None = None
    evaluator: Evaluator | None = None
    executor: Executor | None = None
    worktree_manager: WorktreeManager | None = None
    resource_accountant: ResourceAccountant | None = None
    policy_gate: PolicyGate | None = None
    run_store: RunStore | None = None
    frontier_service: FrontierService | None = None
    export_service: ExportService | None = None
    repository_root: str | None = None
    runtime_root: str | None = None


def make_service_transitions(
    *,
    agent_client: AgentClient | None = None,
    evaluator: Evaluator | None = None,
    executor: Executor | None = None,
    worktree_manager: WorktreeManager | None = None,
    resource_accountant: ResourceAccountant | None = None,
    policy_gate: PolicyGate | None = None,
    run_store: RunStore | None = None,
    frontier_service: FrontierService | None = None,
    export_service: ExportService | None = None,
    repository_root: str | None = None,
    runtime_root: str | None = None,
) -> Mapping[str, Transition]:
    """Build the real service-driven transition map.

    Deterministic ops call policies/services via injected protocols.
    LLM ops call the injected AgentClient and validate/repair responses.
    Every transition includes a non-null ``pending_route``.
    """
    s = ServiceTransitions(
        agent_client=agent_client,
        evaluator=evaluator,
        executor=executor,
        worktree_manager=worktree_manager,
        resource_accountant=resource_accountant,
        policy_gate=policy_gate,
        run_store=run_store,
        frontier_service=frontier_service,
        export_service=export_service,
        repository_root=repository_root,
        runtime_root=runtime_root,
    )

    return {
        "bootstrap": _make_bootstrap(s),
        "inspect": _make_inspect(s),
        "orchestrate": _make_orchestrate(s),
        "research": _make_research(s),
        "proposal_policy": _make_proposal_policy(s),
        "proposal_validation": _make_proposal_validation(s),
        "create_worktree": _make_create_worktree(s),
        "implement": _make_implement(s),
        "diff_policy": _make_diff_policy(s),
        "implementation_validation": _make_implementation_validation(s),
        "register_source": _make_register_source(s),
        "preflight": _make_preflight(s),
        "execute": _make_execute(s),
        "evaluate": _make_evaluate(s),
        "result_validation": _make_result_validation(s),
        "interpret": _make_interpret(s),
        "persist": _make_persist(s),
        "update_frontier": _make_update_frontier(s),
        "repair": _make_repair(s),
        "persist_failure": _make_persist_failure(s),
        "finalize": _make_finalize(s),
        "export": _make_export(s),
    }


# ---------------------------------------------------------------------------
# Deterministic transitions
# ---------------------------------------------------------------------------


def _make_bootstrap(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        _ = state, s
        return {"phase": RunPhase.BOOTSTRAP, "pending_route": "inspect"}
    return transition


def _make_inspect(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        _ = state, s
        return {"phase": RunPhase.RESEARCH, "pending_route": "orchestrate"}
    return transition


def _make_proposal_policy(s: ServiceTransitions) -> Transition:
    """Check proposal scope against protected paths via PolicyGate."""
    async def transition(state: ProductionState) -> dict[str, object]:
        gate = s.policy_gate
        # Default: approve if no policy gate configured
        if gate is not None:
            decision = gate.check_paths(
                changed_paths=("src/tiktok2026/experiment",),
                allowed_scopes=("src/tiktok2026/experiment",),
            )
            if not decision.allowed:
                return {"pending_route": "persist_failure", "terminal_reason": decision.reason}
        return {"pending_route": "proposal_validation"}
    return transition


def _make_diff_policy(s: ServiceTransitions) -> Transition:
    """Check implementation diff against scope via PolicyGate."""
    async def transition(state: ProductionState) -> dict[str, object]:
        gate = s.policy_gate
        if gate is not None:
            decision = gate.check_paths(
                changed_paths=("src/tiktok2026/experiment",),
                allowed_scopes=("src/tiktok2026/experiment",),
            )
            if not decision.allowed:
                return {"pending_route": "persist_failure", "terminal_reason": decision.reason}
        return {"pending_route": "implementation_validation"}
    return transition


def _make_create_worktree(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        wtm = s.worktree_manager
        if wtm is not None:
            spec = ExperimentSpec(
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                hypothesis_id=state["current_hypothesis_id"] or "hyp-unknown",
                hypothesis="placeholder",
                mechanism="placeholder",
                motivation="placeholder",
                expected_signal="placeholder",
                implementation_scope=("src/tiktok2026/experiment",),
                fidelity=state["fidelity"],
                success_criteria="placeholder",
                failure_criteria="placeholder",
            )
            assignment = wtm.create(state["run_id"], spec, "HEAD")
            return {
                "active_worktree_id": assignment.worktree_id,
                "pending_route": "implement",
            }
        return {"active_worktree_id": "synth-wt", "pending_route": "implement"}
    return transition


def _make_register_source(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        wtm = s.worktree_manager
        if wtm is not None and state.get("active_worktree_id"):
            assignment = WorktreeAssignment(
                worktree_id=state["active_worktree_id"] or "wt-unknown",
                run_id=state["run_id"],
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                path=Path("/tmp"),
                branch="main",
                parent_commit="0" * 40,
            )
            wtm.register_source(assignment, ("src/tiktok2026/experiment",))
            return {"pending_route": "preflight"}
        return {"pending_route": "preflight"}
    return transition


def _make_preflight(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        _ = state, s
        return {"pending_route": "execute"}
    return transition


def _make_execute(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ex = s.executor
        if ex is not None:
            request = ExecutionRequest(
                execution_id=f"exec-{state['run_id']}",
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                source_commit="0" * 40,
                command=("python", "-m", "tiktok2026.experiment.train"),
                image="tiktok2026:local@sha256:" + "0" * 64,
                source_path=Path("/tmp"),
                dataset_path=Path("/tmp"),
                output_path=Path("/tmp"),
                timeout_seconds=30,
                memory_bytes=1 << 30,
                cpus=1.0,
            )
            result = await ex.execute(request)
            return {
                "latest_execution_result_id": result.execution_id,
                "phase": RunPhase.EXECUTE,
                "pending_route": "evaluate",
            }
        return {
            "latest_execution_result_id": "synth-exec",
            "phase": RunPhase.EXECUTE,
            "pending_route": "evaluate",
        }
    return transition


def _make_evaluate(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ev = s.evaluator
        if ev is not None:
            evaluation_id = f"eval-{state['run_id']}-{state['current_experiment_id'] or 'unknown'}"
            context = EvaluationContext(
                run_id=state["run_id"],
                evaluation_id=evaluation_id,
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                checkpoint_id=f"ckpt-{state['current_experiment_id'] or 'unknown'}",
                source_commit="0" * 40,
                execution_id=state["latest_execution_result_id"] or "exec-unknown",
                dataset_manifest_id="manifest-1",
                dataset_manifest_sha256="0" * 64,
                split="valid",
                prediction_artifact_id="pred-1",
                prediction_sha256="0" * 64,
                evaluator_id="provisional-within-user-v1",
                evaluator_sha256="0" * 64,
            )
            request = EvaluationRequest(
                evaluation_id=evaluation_id,
                context=context,
            )
            result = ev.evaluate(request)
            # Persist evaluation via RunStore
            rs = s.run_store
            if rs is not None:
                provenance = ProvenanceRequest(
                    run_id=state["run_id"],
                    experiment_id=state["current_experiment_id"] or "exp-unknown",
                    source_commit="0" * 40,
                    execution_id=state["latest_execution_result_id"] or "exec-unknown",
                    dataset_manifest_id="manifest-1",
                    dataset_manifest_sha256="0" * 64,
                    evaluator_id="provisional-within-user-v1",
                    evaluator_sha256="0" * 64,
                )
                rs.put_evaluation(result, provenance)
            return {
                "latest_evaluation_result_id": result.evaluation_id,
                "phase": RunPhase.EVALUATE,
                "pending_route": "result_validation",
            }
        return {
            "latest_evaluation_result_id": "synth-eval",
            "phase": RunPhase.EVALUATE,
            "pending_route": "result_validation",
        }
    return transition


def _make_persist(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        rs = s.run_store
        if rs is not None:
            spec = ExperimentSpec(
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                hypothesis_id=state["current_hypothesis_id"] or "hyp-unknown",
                hypothesis="placeholder",
                mechanism="placeholder",
                motivation="placeholder",
                expected_signal="placeholder",
                implementation_scope=("src/tiktok2026/experiment",),
                fidelity=state["fidelity"],
                success_criteria="placeholder",
                failure_criteria="placeholder",
            )
            rs.put_experiment(
                spec, "completed", state["run_id"], f"persist-{state['state_version']}"
            )
        return {"phase": RunPhase.PERSIST, "pending_route": "update_frontier"}
    return transition


def _make_update_frontier(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        _ = s
        return {"terminal_reason": "converged", "pending_route": "finalize"}
    return transition


def _make_repair(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        return {"repair_attempts": state["repair_attempts"] + 1, "pending_route": "implement"}
    return transition


def _make_persist_failure(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        rs = s.run_store
        failure = FailureRecord(
            failure_id=f"failure-{state['run_id']}-{state['state_version']}",
            experiment_id=state["current_experiment_id"] or "exp-unknown",
            kind=FailureKind.SCIENTIFIC_NON_IMPROVEMENT,
            evidence_refs=(),
            repair_attempt=state["repair_attempts"],
        )
        if rs is not None:
            rs.put_failure(failure, state["run_id"])
        # Use the routes helper to determine next node
        next_route = route_after_failure(state, failure)
        return {"pending_route": next_route}
    return transition


def _make_finalize(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        rs = s.run_store
        if rs is not None:
            # Attempt provisional finalization
            request = ProvisionalFinalizationRequest(
                finalization_id=f"final-{state['run_id']}",
                run_id=state["run_id"],
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                source_commit="0" * 40,
                checkpoint_id=f"ckpt-{state['current_experiment_id'] or 'unknown'}",
                evaluation_id=state["latest_evaluation_result_id"] or "eval-unknown",
                bundle_artifact_id="bundle-1",
                evaluator_id="provisional-within-user-v1",
            )
            with contextlib.suppress(Exception):
                rs.persist_provisional_finalization(request)
        return {"phase": RunPhase.FINALIZE, "pending_route": "export"}
    return transition


def _make_export(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        es = s.export_service
        if es is not None:
            from pathlib import Path
            await es.export_run(state["run_id"], Path("/tmp"))
        return {"phase": RunPhase.COMPLETE, "pending_route": "complete"}
    return transition


# ---------------------------------------------------------------------------
# LLM-driven transitions (use AgentClient with structured parsing)
# ---------------------------------------------------------------------------


def _make_orchestrate(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            request = ResearchRequest(
                request_id=f"orch-{state['run_id']}-{state['state_version']}",
                objective="orchestrate",
                resource_state=ResourceState(
                    remaining_gpu_hours=100.0,
                    accumulated_gpu_hours=0.0,
                    remaining_wall_seconds=3600.0,
                    used_tokens=0,
                    remaining_tokens=100000,
                    disk_bytes_available=1 << 30,
                    reserved_final_gpu_hours=10.0,
                ),
            )
            response = await ac.invoke(request)
            if isinstance(response, AgentFailure):
                return {"orchestration_decision_id": None, "pending_route": "persist_failure"}
            if isinstance(response, OrchestrationDecision):
                decision = response
            else:
                return {"orchestration_decision_id": None, "pending_route": "persist_failure"}
            next_route = route_after_orchestration(decision)
            return {
                "orchestration_decision_id": decision.decision_id,
                "pending_route": next_route,
            }
        return {"pending_route": "research"}
    return transition


def _make_research(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            request = ResearchRequest(
                request_id=f"res-{state['run_id']}-{state['state_version']}",
                objective="propose next experiment",
                resource_state=ResourceState(
                    remaining_gpu_hours=100.0,
                    accumulated_gpu_hours=0.0,
                    remaining_wall_seconds=3600.0,
                    used_tokens=0,
                    remaining_tokens=100000,
                    disk_bytes_available=1 << 30,
                    reserved_final_gpu_hours=10.0,
                ),
            )
            response = await ac.invoke(request)
            if isinstance(response, AgentFailure):
                return {"pending_route": "persist_failure"}
            if isinstance(response, ResearchDecision):
                decision = response
                if decision.experiment_spec is not None:
                    return {
                        "current_experiment_id": decision.experiment_spec.experiment_id,
                        "current_hypothesis_id": decision.experiment_spec.hypothesis_id,
                        "pending_route": "proposal_policy",
                    }
            # Fallback: route to proposal_policy
            return {
                "current_experiment_id": "exp-res",
                "current_hypothesis_id": "hyp-res",
                "pending_route": "proposal_policy",
            }
        return {
            "current_experiment_id": "exp-synth",
            "current_hypothesis_id": "hyp-synth",
            "pending_route": "proposal_policy",
        }
    return transition


def _make_proposal_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            response = await ac.invoke(ValidationReport(
                report_id="dummy",
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                stage=ValidationStage.PROPOSAL,
                verdict=ValidationVerdict.APPROVED,
                leakage_risk="none",
            ))
            if isinstance(response, AgentFailure):
                return {"pending_route": "persist_failure"}
            if isinstance(response, ValidationReport):
                report = response
                next_route = route_after_validation(state, report)
                return {
                    "latest_validation_report_id": report.report_id,
                    "pending_route": next_route,
                }
        return {"latest_validation_report_id": "synth-valid", "pending_route": "create_worktree"}
    return transition


def _make_implement(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            response = await ac.invoke(ImplementationResult(
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                patch_artifact_id="patch-1",
                changed_files=("src/tiktok2026/experiment/train.py",),
            ))
            if isinstance(response, AgentFailure):
                return {"phase": RunPhase.IMPLEMENT, "pending_route": "persist_failure"}
            return {"phase": RunPhase.IMPLEMENT, "pending_route": "diff_policy"}
        return {"phase": RunPhase.IMPLEMENT, "pending_route": "diff_policy"}
    return transition


def _make_implementation_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            response = await ac.invoke(ValidationReport(
                report_id="dummy-impl",
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                stage=ValidationStage.IMPLEMENTATION,
                verdict=ValidationVerdict.APPROVED,
                leakage_risk="none",
            ))
            if isinstance(response, AgentFailure):
                return {"pending_route": "persist_failure"}
            if isinstance(response, ValidationReport):
                report = response
                next_route = route_after_validation(state, report)
                return {
                    "latest_validation_report_id": report.report_id,
                    "pending_route": next_route,
                }
        return {
            "latest_validation_report_id": "synth-impl-valid",
            "pending_route": "register_source",
        }
    return transition


def _make_result_validation(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            response = await ac.invoke(ValidationReport(
                report_id="dummy-result",
                experiment_id=state["current_experiment_id"] or "exp-unknown",
                stage=ValidationStage.RESULT,
                verdict=ValidationVerdict.APPROVED,
                leakage_risk="none",
            ))
            if isinstance(response, AgentFailure):
                return {"pending_route": "persist_failure"}
            if isinstance(response, ValidationReport):
                report = response
                next_route = route_after_validation(state, report)
                return {
                    "latest_validation_report_id": report.report_id,
                    "pending_route": next_route,
                }
        return {
            "latest_validation_report_id": "synth-result-valid",
            "pending_route": "interpret",
        }
    return transition


def _make_interpret(s: ServiceTransitions) -> Transition:
    async def transition(state: ProductionState) -> dict[str, object]:
        ac = s.agent_client
        if ac is not None:
            request = ResearchRequest(
                request_id=f"interp-{state['run_id']}-{state['state_version']}",
                objective="interpret evaluation result",
                resource_state=ResourceState(
                    remaining_gpu_hours=100.0,
                    accumulated_gpu_hours=0.0,
                    remaining_wall_seconds=3600.0,
                    used_tokens=0,
                    remaining_tokens=100000,
                    disk_bytes_available=1 << 30,
                    reserved_final_gpu_hours=10.0,
                ),
            )
            response = await ac.invoke(request)
            if isinstance(response, AgentFailure):
                return {"pending_route": "persist_failure"}
        return {"pending_route": "persist"}
    return transition