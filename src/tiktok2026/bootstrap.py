from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tiktok2026.adapters import (
    DeterministicPolicyGate,
    LedgerResourceAccountant,
    OpenAICompatibleAgentClient,
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
    # Composed adapters
    worktree_manager: Any = None
    executor: Any = None
    evaluator: Any = None
    agent_clients: dict[AgentRole, OpenAICompatibleAgentClient] = field(
        default_factory=dict[AgentRole, OpenAICompatibleAgentClient]
    )


def _approved_parent_validator(commit: str) -> bool:
    """Default parent validator: accept any non-empty hex string."""
    return bool(commit) and len(commit) >= 7


def build_production_services(settings: Any) -> ProductionServices:
    """Construct the full production composition.

    All concrete privileged implementations (SQLite, Git, Docker, evaluator,
    model clients, etc.) are instantiated here.  No network/Docker calls are
    made at construction time.
    """
    from tiktok2026.agents.common.client import OpenAICompatibleClient
    from tiktok2026.config import AppSettings
    from tiktok2026.evaluation.registry import ProvisionalEvaluator
    from tiktok2026.execution.docker import (
        ArtifactStorePublisher,
        DockerExecutor,
        ExecutionPolicy,
    )
    from tiktok2026.persistence.artifacts import ArtifactStore
    from tiktok2026.repository.worktrees import GitWorktreeManager

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
    artifact_store = ArtifactStore(paths, repo)
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

    # Worktree manager
    worktree_manager = GitWorktreeManager(
        repository=app_settings.repository_root,
        runtime_root=paths.root,
        approved_parent_validator=_approved_parent_validator,
        artifact_registry=repo,
    )

    # Docker executor (offline-safe — no Docker calls at construction)
    publisher = ArtifactStorePublisher(
        store=artifact_store,
        run_id="",
        experiment_id="",
        producer="docker-executor",
    )
    executor = DockerExecutor(
        policy=ExecutionPolicy(),
        publisher=publisher,
    )

    # Evaluator — use the ProvisionalEvaluator directly (not the registry)
    evaluator = ProvisionalEvaluator(
        evaluator_id=app_settings.evaluator_id,
    )

    # Agent clients — one per role, wired at construction time
    # (no network calls, no credential validation at construction)
    agent_clients: dict[AgentRole, Any] = {}
    for role in AgentRole:
        model_settings = app_settings.models.get(role)
        if model_settings is not None:
            raw_client = OpenAICompatibleClient(model_settings)
            agent_clients[role] = OpenAICompatibleAgentClient(raw_client)

    # Use the orchestration agent client as the default
    default_agent: Any = agent_clients.get(AgentRole.ORCHESTRATION)

    # Build transitions
    transitions = make_service_transitions(
        agent_client=default_agent,
        evaluator=evaluator,
        executor=executor,
        worktree_manager=worktree_manager,
        resource_accountant=resource_accountant,
        policy_gate=policy_gate,
        run_store=run_store,
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
        settings=app_settings,
        worktree_manager=worktree_manager,
        executor=executor,
        evaluator=evaluator,
        agent_clients=agent_clients,
    )


class _ProductionTransitionStore:
    """Persists transitions through the existing controller checkpoint store."""

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
        if isinstance(request, ResearchRequest) and request.objective == "orchestrate":
            return OrchestrationDecision(
                decision_id=f"synth-dec-{request.request_id}",
                action=DecisionAction.RESEARCH,
                rationale="Synthetic orchestration decision",
            )
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

    from tiktok2026.persistence.artifacts import ArtifactStore

    ArtifactStore(paths, repo)

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

    checkpointer = SqliteCheckpointer(paths.graph_db)
    graph = build_production_graph(controller, checkpointer=checkpointer)

    return controller, store, graph