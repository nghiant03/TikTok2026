from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    BenchmarkManifest,
    verify_protected_files,
)
from tiktok2026.contracts import (
    AgentFailure,
    AgentRole,
    ContractModel,
    DecisionAction,
    EvaluationRequest,
    EvaluationResult,
    ExecutionRequest,
    ExecutionResult,
    ExperimentSpec,
    FailureRecord,
    Fidelity,
    FinalizationRecord,
    ImplementationResult,
    MetricValue,
    OrchestrationDecision,
    PolicyDecisionModel,
    ProvenanceRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    RunRecord,
    RuntimePaths,
    SourceRegistration,
    ValidationReport,
)
from tiktok2026.controller import (
    ControllerServices,
    ProductionController,
)
from tiktok2026.graph.build import build_production_graph
from tiktok2026.persistence.checkpointer import SqliteCheckpointer
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.use_cases import make_service_transitions


@dataclass(frozen=True)
class RuntimeServices:
    repository_root: Path
    paths: RuntimePaths
    repository: ApplicationRepository


def initialize_runtime(repository_root: Path, runtime_root: Path) -> RuntimeServices:
    repository_root = repository_root.resolve()
    paths = RuntimePaths.create(repository_root, runtime_root)
    MigrationRunner(paths.application_db, repository_root / "migrations" / "application").apply()
    MigrationRunner(paths.graph_db, repository_root / "migrations" / "graph").apply()
    return RuntimeServices(
        repository_root=repository_root,
        paths=paths,
        repository=ApplicationRepository(paths.application_db),
    )


def verify_manifests(repository_root: Path) -> BenchmarkManifest:
    manifest_path = (
        repository_root / "src" / "tiktok2026" / "benchmark" / "kuaireand_pure" / "manifest.json"
    )
    manifest = BenchmarkManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    verify_protected_files(repository_root, manifest.protected_reference_files)
    return manifest


# ---------------------------------------------------------------------------
# Synthetic composition — scripted agents INJECTED INTO real service-driven
# transitions.  Reuses persistence, resources, artifacts, graph, and exports.
# ---------------------------------------------------------------------------


class _SyntheticTransitionStore:
    """In-memory store that records every transition."""

    def __init__(self) -> None:
        self.persisted: list[tuple[str, str, int, dict[str, object]]] = []

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))


class _ScriptedAgentClient:
    """AgentClient that returns canned responses for synthetic tests."""

    def __init__(self) -> None:
        self.calls: list[ContractModel] = []

    async def invoke(self, request: ContractModel) -> ContractModel:

        self.calls.append(request)
        # Handle orchestration requests
        if isinstance(request, ResearchRequest) and request.objective == "orchestrate":
            return OrchestrationDecision(
                decision_id=f"synth-dec-{request.request_id}",
                action=DecisionAction.RESEARCH,
                rationale="Synthetic orchestration decision",
            )
        # Return a canned response based on the request type
        if isinstance(request, OrchestrationDecision):
            return request
        if isinstance(request, ValidationReport):
            return request
        if isinstance(request, ImplementationResult):
            return request
        if isinstance(request, ResearchRequest):
            return ResearchDecision(
                request_id=f"synth-{request.request_id}",
                kind="proposal",
                experiment_spec=ExperimentSpec(
                    experiment_id="synth-exp-1",
                    hypothesis_id="synth-hyp-1",
                    hypothesis="Synthetic test hypothesis",
                    mechanism="Deterministic test mechanism",
                    motivation="Test the synthetic pipeline",
                    expected_signal="Metrics remain valid",
                    implementation_scope=("src/tiktok2026/experiment",),
                    fidelity=Fidelity.SMOKE,
                    success_criteria="All transitions persist",
                    failure_criteria="Any transition fails",
                    source_provenance=("synthetic-fixture-v1",),
                ),
                message="Synthetic proposal",
            )
        return AgentFailure(
            request_id=getattr(request, "request_id", "unknown"),
            role=AgentRole.RESEARCH,
            kind="model",
            message="unknown request type",
            repair_attempts=0,
        )


class _FakeEvaluator:
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult(
            evaluation_id=request.evaluation_id,
            experiment_id=request.context.experiment_id,
            checkpoint_id=request.context.checkpoint_id,
            metrics=(
                MetricValue(name="NDCG@10", value=0.5),
                MetricValue(name="Recall@50", value=0.6),
            ),
            evaluator_artifact_id="provisional-within-user-v1",
            evaluator_sha256="0" * 64,
            prediction_sha256="1" * 64,
            validity="provisional",
        )


class _FakeExecutor:
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            execution_id=request.execution_id,
            experiment_id=request.experiment_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=0,
            elapsed_seconds=0.1,
            gpu_hours=0.0,
        )


class _FakePolicyGate:
    def check_paths(
        self, changed_paths: tuple[str, ...], allowed_scopes: tuple[str, ...]
    ) -> PolicyDecisionModel:
        return PolicyDecisionModel(allowed=True, reason="allowed")

    def can_repair(self, repair_attempts: int) -> PolicyDecisionModel:
        return PolicyDecisionModel(allowed=True, reason="allowed")


class _FakeRunStore:
    def __init__(self) -> None:
        self.experiments: list[ExperimentSpec] = []
        self.evaluations: list[EvaluationResult] = []
        self.failures: list[FailureRecord] = []

    def put_experiment(
        self,
        spec: ExperimentSpec,
        status: str,
        run_id: str,
        transition_id: str,
        expected_predecessor: str | None = None,
        audit_event: ContractModel | None = None,
    ) -> None:
        self.experiments.append(spec)

    def put_evaluation(self, result: EvaluationResult, provenance: ProvenanceRequest) -> None:
        self.evaluations.append(result)

    def put_failure(self, record: FailureRecord, run_id: str) -> None:
        self.failures.append(record)

    def put_run(self, record: RunRecord, transition_id: str) -> None:
        pass

    def put_audit_event(self, event: ContractModel) -> None:
        pass

    def get_source_registration(self, experiment_id: str) -> SourceRegistration | None:
        return None

    def persist_provisional_finalization(
        self, request: ContractModel
    ) -> FinalizationRecord:
        return FinalizationRecord(
            finalization_id=getattr(request, "finalization_id", "final-1"),
            run_id=getattr(request, "run_id", "run-1"),
            experiment_id=getattr(request, "experiment_id", "exp-1"),
            source_commit="0" * 40,
            checkpoint_id="ckpt-1",
            evaluation_id="eval-1",
            validity="provisional",
            bundle_artifact_id="bundle-1",
            consumed_test_access=False,
        )


class _FakeResourceAccountant:
    def state(self) -> ResourceState:
        return ResourceState(
            remaining_gpu_hours=100.0,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=3600.0,
            used_tokens=0,
            remaining_tokens=100000,
            disk_bytes_available=1 << 30,
            reserved_final_gpu_hours=10.0,
        )

    def reserve(self, reservation: ContractModel) -> bool:
        return True

    def consume(self, reservation_id: str, **usage: float | int) -> bool:
        return True


def build_synthetic_controller(
    repository_root: Path,
    runtime_root: Path,
) -> tuple[ProductionController, object, object]:
    """Build a synthetic composition: real service-driven transitions with
    scripted agents, fake executor/evaluator/policy, and real persistence.

    No network, Docker, or GPU resources are required.
    Returns (controller, transitions_store, compiled_graph).
    """
    runtime = initialize_runtime(repository_root, runtime_root)

    # Build service-driven transitions with scripted/fake implementations
    store = _SyntheticTransitionStore()
    agent = _ScriptedAgentClient()
    fake_run_store = _FakeRunStore()
    fake_policy = _FakePolicyGate()

    # Use the real service-driven transition factory, injecting scripted fakes
    transitions = make_service_transitions(
        agent_client=agent,
        evaluator=_FakeEvaluator(),
        executor=_FakeExecutor(),
        worktree_manager=None,
        resource_accountant=_FakeResourceAccountant(),
        policy_gate=fake_policy,
        run_store=fake_run_store,
        frontier_service=None,
        export_service=None,
    )

    services = ControllerServices(
        transitions=transitions,
        store=store,
    )
    controller = ProductionController(services)

    # Build the graph with a checkpointer backed by the real graph DB
    checkpointer = SqliteCheckpointer(runtime.paths.graph_db)
    graph = build_production_graph(controller, checkpointer=checkpointer)

    return controller, store, graph