from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tiktok2026.adapters import (
    DeterministicPolicyGate,
    LedgerResourceAccountant,
    RepositoryExportService,
    RepositoryRunStore,
)
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
    Fidelity,
    ImplementationResult,
    MetricValue,
    OrchestrationDecision,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    RuntimePaths,
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
from tiktok2026.persistence.resources import ResourceLedger
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
# Production composition root — all concrete privileged implementations
# instantiated HERE ONLY.
# ---------------------------------------------------------------------------


@dataclass
class ProductionServices:
    controller: ProductionController
    repository: ApplicationRepository
    graph: Any
    settings: Any = None


def build_production_services(settings: Any) -> ProductionServices:
    """Construct the full production composition.

    All concrete privileged implementations (SQLite, Git, Docker, evaluator,
    model clients, etc.) are instantiated here.  No network/Docker calls are
    made at construction time.
    """
    from tiktok2026.config import AppSettings
    from tiktok2026.persistence.artifacts import ArtifactStore

    app_settings: AppSettings
    if isinstance(settings, AppSettings):
        app_settings = settings
    elif isinstance(settings, dict):
        app_settings = AppSettings(**settings)  # type: ignore[arg-type]
    else:
        raise TypeError("settings must be AppSettings or dict")

    # Initialize runtime layout and migrations
    runtime = initialize_runtime(app_settings.repository_root, app_settings.runtime_root)
    repo = runtime.repository
    paths = runtime.paths

    # Persistence services
    ArtifactStore(paths, repo)
    ledger = ResourceLedger(
        paths.application_db,
        ResourceState(
            remaining_gpu_hours=app_settings.budget.gpu_hours,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=float(app_settings.budget.wall_clock_seconds),
            used_tokens=0,
            remaining_tokens=app_settings.budget.tokens,
            disk_bytes_available=app_settings.budget.disk_bytes,
            reserved_final_gpu_hours=app_settings.budget.reserved_final_gpu_hours,
        ),
    )

    # Adapters
    run_store = RepositoryRunStore(repo)
    policy_gate = DeterministicPolicyGate()
    resource_accountant = LedgerResourceAccountant(ledger)
    export_service = RepositoryExportService(repo, paths.root)

    # Build transitions
    transitions = make_service_transitions(
        policy_gate=policy_gate,
        run_store=run_store,
        resource_accountant=resource_accountant,
        export_service=export_service,
        runtime_root=str(paths.root),
        repository_root=str(app_settings.repository_root),
        docker_image=app_settings.docker_image,
        evaluator_id=app_settings.evaluator_id,
    )

    # Controller
    store = _ProductionTransitionStore()
    services = ControllerServices(transitions=transitions, store=store)
    controller = ProductionController(services)

    # Graph with checkpointer
    checkpointer = SqliteCheckpointer(paths.graph_db)
    graph = build_production_graph(controller, checkpointer=checkpointer)

    return ProductionServices(
        controller=controller,
        repository=repo,
        graph=graph,
        settings=settings,
    )


class _ProductionTransitionStore:
    """Persists transitions through the existing controller checkpoint store.

    This is a minimal in-memory store; the full production checkpoint
    store would write to the graph DB directly.
    """

    def __init__(self) -> None:
        self.persisted: list[tuple[str, str, int, dict[str, object]]] = []

    def persist_transition(
        self, run_id: str, operation: str, state_version: int, updates: dict[str, object]
    ) -> None:
        self.persisted.append((run_id, operation, state_version, updates))


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


def build_synthetic_controller(
    repository_root: Path,
    runtime_root: Path,
) -> tuple[ProductionController, object, Any]:
    """Build a synthetic composition: real service-driven transitions with
    scripted agents, fake executor, fixture evaluator, and real adapters
    for persistence, policy, resources, and exports.

    No network, Docker, or GPU resources are required.
    Returns (controller, transitions_store, compiled_graph).
    """
    runtime = initialize_runtime(repository_root, runtime_root)
    repo = runtime.repository
    paths = runtime.paths

    # Build real adapters over the real persistence
    artifact_store = _ArtifactStoreDummy()  # ArtifactStore needs paths
    from tiktok2026.persistence.artifacts import ArtifactStore

    real_artifact_store = ArtifactStore(paths, repo)
    _ = artifact_store, real_artifact_store  # keep for future use

    run_store = RepositoryRunStore(repo)
    policy_gate = DeterministicPolicyGate()
    ledger = ResourceLedger(
        paths.application_db,
        ResourceState(
            remaining_gpu_hours=100.0,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=3600.0,
            used_tokens=0,
            remaining_tokens=100000,
            disk_bytes_available=1 << 30,
            reserved_final_gpu_hours=10.0,
        ),
    )
    resource_accountant = LedgerResourceAccountant(ledger)
    export_service = RepositoryExportService(repo, paths.root)

    # Build service-driven transitions with scripted/fake implementations
    store = _SyntheticTransitionStore()
    agent = _ScriptedAgentClient()

    transitions = make_service_transitions(
        agent_client=agent,
        evaluator=_FakeEvaluator(),
        executor=_FakeExecutor(),
        worktree_manager=None,
        resource_accountant=resource_accountant,
        policy_gate=policy_gate,
        run_store=run_store,
        frontier_service=None,
        export_service=export_service,
        runtime_root=str(paths.root),
        repository_root=str(repository_root),
    )

    services = ControllerServices(transitions=transitions, store=store)
    controller = ProductionController(services)

    # Build the graph with a checkpointer backed by the real graph DB
    checkpointer = SqliteCheckpointer(paths.graph_db)
    graph = build_production_graph(controller, checkpointer=checkpointer)

    return controller, store, graph


class _ArtifactStoreDummy:
    """Placeholder until full ArtifactStore wiring is needed."""

    def publish_bytes(self, *args: Any, **kwargs: Any) -> Any:
        return None