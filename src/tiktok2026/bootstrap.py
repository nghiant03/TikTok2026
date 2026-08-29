from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from tiktok2026.adapters import (
    DeterministicPolicyGate,
    LedgerResourceAccountant,
    RepositoryExportService,
    RepositoryFinalizationBundleService,
    RepositoryFrontierService,
    RepositoryRunStore,
    RepositoryTransitionStore,
    RoleSpecificAgentClient,
)
from tiktok2026.benchmark.kuaireand_pure.manifest import BenchmarkManifest, verify_protected_files
from tiktok2026.contracts import (
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    AuditEvent,
    ContractModel,
    DatasetManifestIdentity,
    DecisionAction,
    EvaluationResult,
    EvaluatorIdentity,
    ExecutionRequest,
    ExecutionResult,
    ExperimentSpec,
    Fidelity,
    FinalizationBundleRequest,
    ImplementationRequest,
    ImplementationResult,
    OperationResult,
    OrchestrationDecision,
    PredictionArtifactRegistration,
    ProvisionalFinalizationRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    RunPhase,
    RunRecord,
    RuntimePaths,
    SourceRegistration,
    ValidationReport,
    ValidationRequest,
    ValidationVerdict,
    WorktreeAssignment,
)
from tiktok2026.controller import ControllerServices, ProductionController
from tiktok2026.evaluation.registry import evaluator_implementation_sha256
from tiktok2026.graph.build import build_production_graph
from tiktok2026.persistence.checkpointer import SqliteCheckpointer
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.persistence.resources import ResourceLedger
from tiktok2026.recovery import (
    RecoveryCandidate,
    reconcile_recovery,
    validate_pre_registration_assignment,
)
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
    repository = ApplicationRepository(paths.application_db)
    repository.initialize()
    return RuntimeServices(repository_root, paths, repository)


def verify_manifests(repository_root: Path) -> BenchmarkManifest:
    manifest_path = repository_root / "src/tiktok2026/benchmark/kuaireand_pure/manifest.json"
    manifest = BenchmarkManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    verify_protected_files(repository_root, manifest.protected_reference_files)
    return manifest


@dataclass
class ProductionServices:
    controller: ProductionController
    repository: ApplicationRepository
    graph: Any
    settings: Any = None
    worktree_manager: Any = None
    executor: Any = None
    evaluator: Any = None
    agent_clients: dict[AgentRole, RoleSpecificAgentClient] = field(default_factory=lambda: {})


def _current_commit(repository: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "@^{commit}"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _approved_parent_validator(commit: str) -> bool:
    return bool(commit) and len(commit) >= 7


def _persist_finalization(
    repository: ApplicationRepository, runtime_root: Path, run_id: str
) -> Any:
    finalization_id = f"finalization-{run_id}"
    existing = repository.get_finalization(finalization_id)
    if existing is not None:
        return existing
    events = repository.list_audit_events(run_id)
    experiment_id = next(
        (event.experiment_id for event in reversed(events) if event.experiment_id), None
    )
    if experiment_id is None:
        raise ValueError("no experiment found for this run")
    store = RepositoryRunStore(repository)
    source = store.get_source_registration(experiment_id)
    evaluation_values: list[EvaluationResult] = []
    for raw in repository.list_json("evaluation"):
        value = json.loads(raw)
        evaluation = EvaluationResult.model_validate(value.get("result", value))
        if evaluation.experiment_id == experiment_id:
            evaluation_values.append(evaluation)
    evaluation = evaluation_values[-1] if evaluation_values else None
    if source is None or evaluation is None:
        raise ValueError("finalization provenance is unavailable")
    if evaluation.run_id != run_id:
        raise ValueError("evaluation provenance does not match this run")
    bundle = RepositoryFinalizationBundleService(repository, runtime_root).create(
        FinalizationBundleRequest(
            run_id=run_id,
            experiment_id=experiment_id,
            source_commit=source.source_commit,
            checkpoint_id=evaluation.checkpoint_id,
            evaluation_id=evaluation.evaluation_id,
            evaluator_id=evaluation.evaluator_artifact_id,
        )
    )
    return repository.persist_provisional_finalization(
        ProvisionalFinalizationRequest(
            finalization_id=finalization_id,
            run_id=run_id,
            experiment_id=experiment_id,
            source_commit=source.source_commit,
            checkpoint_id=evaluation.checkpoint_id,
            evaluation_id=evaluation.evaluation_id,
            bundle_artifact_id=bundle.artifact_id,
            evaluator_id=evaluation.evaluator_artifact_id,
        )
    )


class OperationalError(RuntimeError):
    """A typed operator-facing failure from a bootstrap-owned operation."""


def _initial_state(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "phase": RunPhase.BOOTSTRAP,
        "current_experiment_id": None,
        "current_hypothesis_id": None,
        "active_worktree_id": None,
        "latest_validation_report_id": None,
        "latest_execution_result_id": None,
        "latest_evaluation_result_id": None,
        "orchestration_decision_id": None,
        "repair_attempts": 0,
        "fidelity": Fidelity.SMOKE,
        "pending_route": None,
        "terminal_reason": None,
        "state_version": 0,
    }


def _checkpoint_state(checkpoint: dict[str, object]) -> dict[str, object]:
    channels = checkpoint.get("channel_values")
    if isinstance(channels, dict):
        typed_channels = cast(dict[object, object], channels)
        return {str(key): value for key, value in typed_channels.items()}
    return checkpoint


def _result(
    operation: str,
    *,
    run_id: str | None = None,
    phase: object = None,
    status: str,
    values: dict[str, object] | None = None,
) -> OperationResult:
    parsed_phase: RunPhase | None = None
    if phase is not None:
        try:
            parsed_phase = RunPhase(str(phase))
        except ValueError:
            parsed_phase = None
    return OperationResult(
        operation=operation,
        run_id=run_id,
        phase=parsed_phase,
        status=status,
        values=values or {},
    )


class ProductionOperations:
    """Single operator composition root for production and synthetic commands."""

    def __init__(
        self,
        repository_root: Path,
        runtime_root: Path,
        profile_path: Path | None = None,
        operator_config: Path | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.runtime_root = runtime_root.resolve()
        self.profile_path = profile_path
        self.operator_config = operator_config

    def runtime_init(self) -> OperationResult:
        services = initialize_runtime(self.repository_root, self.runtime_root)
        return _result(
            "runtime-init", status="initialized", values={"root": str(services.paths.root)}
        )

    def migrate(self) -> OperationResult:
        services = initialize_runtime(self.repository_root, self.runtime_root)
        return _result("migrate", status="migrated", values={"root": str(services.paths.root)})

    def verify_manifests(self) -> OperationResult:
        manifest = verify_manifests(self.repository_root)
        return _result(
            "verify-manifests", status="verified", values={"benchmark_id": manifest.benchmark_id}
        )

    def diagnostics(self) -> OperationResult:
        manifest = verify_manifests(self.repository_root)
        return _result(
            "diagnostics",
            status="verified",
            values={
                "protected_manifest": "verified",
                "evaluator_status": manifest.judging_evaluator_status,
            },
        )

    def synthetic_run(self, iterations: int) -> OperationResult:
        from tiktok2026.testing import run_synthetic_lifecycle

        result = asyncio.run(run_synthetic_lifecycle(iterations, runtime_root=self.runtime_root))
        return _result(
            "synthetic-run",
            run_id=result.run_id,
            status="completed",
            values={
                "experiment_ids": result.experiment_ids,
                "validity": result.finalization.validity,
                "jsonl": str(result.exports.jsonl),
                "markdown": str(result.exports.markdown),
            },
        )

    def run(self, *, synthetic: bool = False, run_id: str | None = None) -> OperationResult:
        actual_run_id = run_id or ("test-run" if synthetic else f"prod-{uuid.uuid4().hex[:8]}")
        if synthetic:
            _controller, _store, graph = build_synthetic_controller(
                self.repository_root, self.runtime_root
            )
            repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        else:
            settings = self._production_settings()
            missing = sorted(
                {
                    model.api_key_env
                    for model in settings.models.values()
                    if not os.getenv(model.api_key_env)
                }
            )
            if missing:
                raise OperationalError(
                    "production run requires configured credentials: " + ", ".join(missing)
                )
            services = build_production_services(settings)
            graph, repository = services.graph, services.repository
        if not synthetic:
            repository.put_run(
                RunRecord(run_id=actual_run_id, status="active"), f"{actual_run_id}-active", None
            )
        repository.put_audit_event(
            AuditEvent(
                event_id=f"run-{actual_run_id}-start",
                run_id=actual_run_id,
                event_type="run_started",
                actor_type="human",
                actor_id="cli-operator",
                payload={
                    "run_id": actual_run_id,
                    "mode": "synthetic" if synthetic else "production",
                },
            )
        )
        try:
            state = asyncio.run(
                graph.ainvoke(
                    _initial_state(actual_run_id), {"configurable": {"thread_id": actual_run_id}}
                )
            )
        except Exception as error:
            raise OperationalError(str(error)) from error
        return _result(
            "run",
            run_id=actual_run_id,
            phase=state.get("phase"),
            status="completed" if str(state.get("phase")) == RunPhase.COMPLETE.value else "running",
            values={
                "pending_route": state.get("pending_route"),
                "state_version": state.get("state_version", 0),
            },
        )

    def resume(self, run_id: str, *, synthetic: bool = False) -> OperationResult:
        checkpoint = self._load_checkpoint(run_id)
        if checkpoint is None:
            raise OperationalError(f"no durable checkpoint exists for run {run_id}")
        state = _checkpoint_state(checkpoint)
        phase = str(state.get("phase"))
        repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        if phase in {RunPhase.COMPLETE.value, str(RunPhase.COMPLETE)}:
            self._resume_audit(repository, run_id, True, "run is already complete")
            return _result(
                "resume",
                run_id=run_id,
                phase=RunPhase.COMPLETE,
                status="already_complete",
                values={"resumed": False},
            )
        if not synthetic:
            self._reconcile_resume_boundary(repository, run_id, state)
        if synthetic:
            _controller, _store, graph = build_synthetic_controller(
                self.repository_root, self.runtime_root
            )
        else:
            settings = self._production_settings()
            graph = build_production_services(settings).graph
        try:
            result = asyncio.run(graph.ainvoke(None, {"configurable": {"thread_id": run_id}}))
        except Exception as error:
            self._resume_audit(repository, run_id, False, str(error))
            raise OperationalError(str(error)) from error
        self._resume_audit(repository, run_id, True, "checkpoint resumed")
        return _result(
            "resume",
            run_id=run_id,
            phase=result.get("phase"),
            status="resumed",
            values={
                "pending_route": result.get("pending_route"),
                "state_version": result.get("state_version", 0),
            },
        )

    def inspect(self, run_id: str) -> OperationResult:
        repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        events = repository.list_audit_events(run_id)
        if not events:
            raise OperationalError(f"run {run_id} not found")
        return _result(
            "inspect",
            run_id=run_id,
            status="available",
            values={"events": [event.model_dump(mode="json") for event in events]},
        )

    def finalize(self, run_id: str) -> OperationResult:
        repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        try:
            finalization = _persist_finalization(repository, self.runtime_root, run_id)
        except Exception as error:
            raise OperationalError(str(error)) from error
        return _result(
            "finalize",
            run_id=run_id,
            status="finalized",
            values={
                "finalization_id": finalization.finalization_id,
                "experiment_id": finalization.experiment_id,
                "validity": finalization.validity,
            },
        )

    def export(self, run_id: str) -> OperationResult:
        repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        if not repository.list_audit_events(run_id):
            raise OperationalError(f"run {run_id} not found")
        if repository.get_finalization(f"finalization-{run_id}") is None:
            raise OperationalError("export requires a persisted finalization")
        result = asyncio.run(
            RepositoryExportService(repository, self.runtime_root).export_run(run_id)
        )
        return _result(
            "export",
            run_id=run_id,
            status="exported",
            values={"jsonl": str(result["jsonl"]), "markdown": str(result["markdown"])},
        )

    def _load_checkpoint(self, run_id: str) -> dict[str, object] | None:
        checkpointer = SqliteCheckpointer(self.runtime_root / "graph.sqlite3")
        value = asyncio.run(checkpointer.aget_tuple({"configurable": {"thread_id": run_id}}))
        return value.checkpoint if value is not None else None

    def _production_settings(self) -> Any:
        from tiktok2026.config import AppSettings

        profile_path = self.profile_path or (
            self.repository_root / "config" / "budgets" / "judged.toml"
        )
        if not profile_path.is_file():
            raise OperationalError(f"production profile does not exist: {profile_path}")
        try:
            return AppSettings.load(
                repository_root=self.repository_root,
                profile_path=profile_path,
                operator_path=self.operator_config,
                overrides={"runtime_root": self.runtime_root, "profile": "production"},
            )
        except (OSError, ValueError) as error:
            raise OperationalError(f"invalid production settings: {error}") from error

    def _resume_audit(
        self, repository: ApplicationRepository, run_id: str, accepted: bool, reason: str
    ) -> None:
        repository.put_audit_event(
            AuditEvent(
                event_id=(
                    f"resume-{run_id}-{uuid.uuid4().hex[:8]}-"
                    f"{'accepted' if accepted else 'rejected'}"
                ),
                run_id=run_id,
                event_type="resume_accepted" if accepted else "resume_rejected",
                actor_type="controller",
                actor_id="production-operations",
                payload={"reason": reason},
            )
        )

    def _reconcile_late_resume(
        self, repository: ApplicationRepository, run_id: str, state: dict[str, object]
    ) -> None:
        store = RepositoryRunStore(repository)
        experiment_id = str(state.get("current_experiment_id") or "")
        source = store.get_source_registration(experiment_id) if experiment_id else None
        assignment = store.get_worktree_assignment(experiment_id) if experiment_id else None
        patch = repository.get_artifact(source.patch_artifact_id) if source is not None else None
        if (
            source is None
            or assignment is None
            or patch is None
            or source.run_id != run_id
            or source.experiment_id != experiment_id
            or assignment.run_id != run_id
            or assignment.experiment_id != experiment_id
            or patch.run_id != run_id
            or patch.experiment_id != experiment_id
            or patch.artifact_id != source.patch_artifact_id
            or patch.sha256 != source.patch_sha256
            or patch.uri != source.patch_artifact_uri
        ):
            reason = "late resume provenance is unavailable"
            self._resume_audit(repository, run_id, False, reason)
            raise OperationalError(reason)
        candidate = RecoveryCandidate(
            run_id=run_id,
            experiment_id=experiment_id,
            database_source_commit=source.source_commit,
            worktree_source_commit=source.source_commit,
            database_artifact_sha256=source.patch_sha256,
            artifact_sha256=patch.sha256,
            stale_lock=self.runtime_root / "locks" / f"{run_id}.lock",
            worktree_path=assignment.path,
            artifact_uri=Path(patch.uri.removeprefix("file://")),
            stale_reservation_id=self._reservation_id(run_id),
        )
        result = reconcile_recovery(candidate, self._release_reservation)
        if not result.resumable:
            self._resume_audit(repository, run_id, False, result.reason)
            raise OperationalError(result.reason)

    def _reconcile_resume_boundary(
        self, repository: ApplicationRepository, run_id: str, state: dict[str, object]
    ) -> None:
        route = str(state.get("pending_route") or "")
        if route in {
            "",
            "bootstrap",
            "inspect",
            "orchestrate",
            "research",
            "proposal_policy",
            "proposal_validation",
        }:
            return
        if route == "create_worktree":
            experiment_id = str(state.get("current_experiment_id") or "")
            assignment = (
                RepositoryRunStore(repository).get_worktree_assignment(experiment_id)
                if experiment_id
                else None
            )
            if assignment is not None:
                result = validate_pre_registration_assignment(
                    assignment,
                    self.runtime_root,
                    lambda commit: self._approved_parent_for_resume(commit),
                )
                if not result.resumable:
                    self._resume_audit(repository, run_id, False, result.reason)
                    raise OperationalError(result.reason)
            return
        if route in {"implement", "diff_policy", "implementation_validation", "register_source"}:
            experiment_id = str(state.get("current_experiment_id") or "")
            assignment = (
                RepositoryRunStore(repository).get_worktree_assignment(experiment_id)
                if experiment_id
                else None
            )
            if assignment is None:
                reason = "pre-registration worktree assignment is unavailable"
                self._resume_audit(repository, run_id, False, reason)
                raise OperationalError(reason)
            result = validate_pre_registration_assignment(
                assignment,
                self.runtime_root,
                lambda commit: self._approved_parent_for_resume(commit),
            )
            if not result.resumable:
                self._resume_audit(repository, run_id, False, result.reason)
                raise OperationalError(result.reason)
            return
        self._reconcile_late_resume(repository, run_id, state)

    def _approved_parent_for_resume(self, commit: str) -> bool:
        return _current_commit(self.repository_root) == commit

    def _reservation_id(self, run_id: str) -> str | None:
        with sqlite3.connect(self.runtime_root / "application.sqlite3") as connection:
            rows = connection.execute(
                "SELECT reservation_id, reservation_json "
                "FROM authority_resource_reservations WHERE status = 'reserved'"
            ).fetchall()
        return next(
            (
                str(identifier)
                for identifier, payload in rows
                if json.loads(payload).get("run_id") == run_id
            ),
            None,
        )

    def _release_reservation(self, reservation_id: str) -> bool:
        ledger = ResourceLedger(
            self.runtime_root / "application.sqlite3",
            ResourceState(
                remaining_gpu_hours=0,
                accumulated_gpu_hours=0,
                remaining_wall_seconds=0,
                used_tokens=0,
                remaining_tokens=0,
                disk_bytes_available=0,
                reserved_final_gpu_hours=0,
            ),
        )
        return ledger.release(reservation_id)


def build_production_operations(
    repository_root: Path,
    runtime_root: Path,
    profile_path: Path | None = None,
    operator_config: Path | None = None,
) -> ProductionOperations:
    """Build the sole operator-facing composition used by the CLI."""
    return ProductionOperations(repository_root, runtime_root, profile_path, operator_config)


class _RunBoundDockerExecutor:
    """Bind dataset, source, and publication authority to each execution request."""

    def __init__(
        self,
        repository: ApplicationRepository,
        artifact_store: Any,
        policy: Any,
        dataset_provider: Any,
        evaluator: Any | None = None,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store
        self.policy = policy
        self.dataset_provider = dataset_provider
        self.evaluator = evaluator

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        from tiktok2026.execution.docker import (
            ArtifactStorePublisher,
            DockerExecutor,
            RegisteredGitSourceVerifier,
        )

        assignment = RepositoryRunStore(self.repository).get_worktree_assignment(
            request.experiment_id
        )
        if assignment is None:
            raise ValueError("execution worktree assignment is unavailable")
        if request.run_id is None:
            raise ValueError("execution run identity is unavailable")
        executor = DockerExecutor(
            policy=self.policy,
            publisher=ArtifactStorePublisher(
                store=self.artifact_store,
                run_id=request.run_id,
                experiment_id=request.experiment_id,
            ),
            dataset_provider=self.dataset_provider,
            source_verifier=RegisteredGitSourceVerifier(self.repository, assignment),
        )
        result = await executor.execute(request)
        if result.exit_code != 0:
            return result
        return self._register_training_artifacts(request, result)

    def _register_training_artifacts(
        self, request: ExecutionRequest, result: ExecutionResult
    ) -> ExecutionResult:
        if request.run_id is None:
            raise ValueError("execution run identity is unavailable")
        dataset_identity = RepositoryRunStore(self.repository).get_dataset_manifest_identity()
        if dataset_identity is None:
            raise ValueError("verified dataset manifest identity is unavailable")
        output = request.output_path.resolve()
        predictions_path = output / "predictions.json"
        checkpoint_path = output / "checkpoint_bundle.json"
        if not predictions_path.is_file() or not checkpoint_path.is_file():
            raise ValueError("execution did not produce prediction and checkpoint artifacts")
        try:
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("execution artifacts are not valid JSON") from error
        if not isinstance(predictions, dict) or not isinstance(checkpoint, dict):
            raise ValueError("execution artifacts must be JSON objects")
        prediction_payload = cast(dict[str, object], predictions)
        checkpoint_payload = cast(dict[str, object], checkpoint)
        prediction_bytes = predictions_path.read_bytes()
        prediction_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
        expected = {
            "manifest_id": dataset_identity.manifest_id,
            "manifest_sha256": dataset_identity.manifest_sha256,
            "source_commit": request.source_commit,
            "execution_id": request.execution_id,
        }
        if any(prediction_payload.get(key) != value for key, value in expected.items()):
            raise ValueError("prediction artifact provenance does not match execution")
        checkpoint_expected = {
            "data_manifest_id": dataset_identity.manifest_id,
            "source_commit": request.source_commit,
            "execution_id": request.execution_id,
            "prediction_artifact": predictions_path.name,
            "prediction_sha256": prediction_sha256,
        }
        if any(checkpoint_payload.get(key) != value for key, value in checkpoint_expected.items()):
            raise ValueError("checkpoint artifact provenance does not match execution")
        checkpoint_id = checkpoint_payload.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint artifact has no checkpoint identity")
        prediction = self.artifact_store.publish_bytes(
            request.run_id,
            request.experiment_id,
            "prediction",
            predictions_path.name,
            prediction_bytes,
            "docker-executor",
            ArtifactRetention.RUN,
        )
        checkpoint_artifact = self.artifact_store.publish_bytes(
            request.run_id,
            request.experiment_id,
            "checkpoint",
            checkpoint_path.name,
            checkpoint_path.read_bytes(),
            "docker-executor",
            ArtifactRetention.RUN,
        )
        prediction_registration = PredictionArtifactRegistration(
            artifact_id=prediction.artifact_id,
            path=Path(prediction.uri.removeprefix("file://")),
            sha256=prediction.sha256,
            checkpoint_id=checkpoint_id,
            source_commit=request.source_commit,
            execution_id=request.execution_id,
            dataset_manifest_id=dataset_identity.manifest_id,
            dataset_manifest_sha256=dataset_identity.manifest_sha256,
            split="valid",
        )
        self.repository.put_json(
            "prediction_artifact", prediction.artifact_id, prediction_registration.model_dump_json()
        )
        if self.evaluator is not None:
            self.evaluator.artifacts[prediction.artifact_id] = prediction_registration
        return result.model_copy(
            update={
                "artifact_ids": result.artifact_ids
                + (prediction.artifact_id, checkpoint_artifact.artifact_id),
                "checkpoint_id": checkpoint_id,
            }
        )


def build_production_services(settings: Any) -> ProductionServices:
    from tiktok2026.agents.common.client import OpenAICompatibleClient
    from tiktok2026.config import AppSettings
    from tiktok2026.evaluation.registry import ProvisionalEvaluator
    from tiktok2026.execution.docker import AuthorizedTrainingDatasetProvider, ExecutionPolicy
    from tiktok2026.persistence.artifacts import ArtifactStore
    from tiktok2026.repository.worktrees import GitWorktreeManager

    if isinstance(settings, AppSettings):
        app_settings = settings
    elif isinstance(settings, dict):
        app_settings = AppSettings.model_validate(settings)
    else:
        raise TypeError("settings must be AppSettings or dict")
    if app_settings.profile == "production":
        missing: list[str] = []
        if app_settings.dataset_root is None:
            missing.append("dataset_root")
        elif (
            not app_settings.dataset_root.is_dir()
            or not (app_settings.dataset_root / "manifest.json").is_file()
        ):
            missing.append("validated dataset manifest")
        if set(app_settings.models) != set(AgentRole):
            missing.append("models for all four roles")
        if "@sha256:" not in app_settings.docker_image:
            missing.append("immutable docker_image")
        if _current_commit(app_settings.repository_root) is None:
            missing.append("approved repository commit")
        missing_credentials = sorted(
            {
                model.api_key_env
                for model in app_settings.models.values()
                if not os.getenv(model.api_key_env)
            }
        )
        if missing_credentials:
            missing.append("credentials for " + ", ".join(missing_credentials))
        if missing:
            raise ValueError("incomplete production settings: " + ", ".join(missing))

    runtime = initialize_runtime(app_settings.repository_root, app_settings.runtime_root)
    repo, paths = runtime.repository, runtime.paths
    artifact_store = ArtifactStore(paths, repo)
    ledger = ResourceLedger(
        paths.application_db,
        ResourceState(
            remaining_gpu_hours=app_settings.budget.gpu_hours,
            accumulated_gpu_hours=0,
            remaining_wall_seconds=float(app_settings.budget.wall_clock_seconds),
            used_tokens=0,
            remaining_tokens=app_settings.budget.tokens,
            disk_bytes_available=app_settings.budget.disk_bytes,
            reserved_final_gpu_hours=app_settings.budget.reserved_final_gpu_hours,
        ),
    )
    run_store = RepositoryTransitionStore(repo)
    evaluator_hash = evaluator_implementation_sha256()
    run_store.put_evaluator_identity(
        EvaluatorIdentity(
            evaluator_id=app_settings.evaluator_id,
            evaluator_sha256=evaluator_hash,
            validity="provisional",
        )
    )
    dataset_registry: dict[str, Any] = {}
    dataset_provider: Any = None
    if app_settings.dataset_root is not None:
        manifest_path = app_settings.dataset_root / "manifest.json"
        if manifest_path.is_file():
            from tiktok2026.benchmark.kuaireand_pure.manifest import (
                canonical_manifest_sha256,
                load_dataset_manifest,
                verify_dataset_manifest,
            )

            manifest = load_dataset_manifest(manifest_path)
            verified = verify_dataset_manifest(
                manifest, app_settings.dataset_root, splits={"train", "valid"}
            )
            dataset_registry[manifest.manifest_id] = verified
            dataset_provider = AuthorizedTrainingDatasetProvider(verified.training_view())
            run_store.put_dataset_manifest_identity(
                DatasetManifestIdentity(
                    manifest_id=manifest.manifest_id,
                    manifest_sha256=canonical_manifest_sha256(manifest),
                )
            )
    worktree_manager = GitWorktreeManager(
        repository=app_settings.repository_root,
        runtime_root=paths.root,
        approved_parent_validator=_approved_parent_validator,
        artifact_registry=repo,
    )
    evaluator = ProvisionalEvaluator(
        evaluator_id=app_settings.evaluator_id, datasets=dataset_registry
    )
    executor = _RunBoundDockerExecutor(
        repository=repo,
        artifact_store=artifact_store,
        policy=ExecutionPolicy(),
        dataset_provider=dataset_provider,
        evaluator=evaluator,
    )
    prompts = {
        AgentRole.ORCHESTRATION: "Select one allowed orchestration action as JSON.",
        AgentRole.RESEARCH: "Return one evidence-backed ResearchDecision as JSON.",
        AgentRole.IMPLEMENTOR: (
            "Use the bound scoped worktree capability and return one faithful "
            "ImplementationResult as JSON with at least one bounded edit."
        ),
        AgentRole.VALIDATOR: "Return one adversarial ValidationReport as JSON.",
    }
    capabilities = {
        AgentRole.ORCHESTRATION: ("route", "budget", "frontier"),
        AgentRole.RESEARCH: ("repository_read", "dataset_summary", "memory", "literature"),
        AgentRole.IMPLEMENTOR: ("scoped_read", "scoped_write", "diff", "checks"),
        AgentRole.VALIDATOR: ("repository_read", "diff", "provenance", "evaluation_read"),
    }
    agents: dict[AgentRole, RoleSpecificAgentClient] = {
        role: RoleSpecificAgentClient(
            OpenAICompatibleClient(model), role, prompts[role], capabilities[role]
        )
        for role, model in app_settings.models.items()
    }
    transitions = make_service_transitions(
        agent_clients=agents,
        evaluator=evaluator,
        executor=executor,
        worktree_manager=worktree_manager,
        resource_accountant=LedgerResourceAccountant(ledger),
        policy_gate=DeterministicPolicyGate(),
        run_store=run_store,
        export_service=RepositoryExportService(repo, paths.root),
        bundle_service=RepositoryFinalizationBundleService(repo, paths.root),
        frontier_service=RepositoryFrontierService(
            repo,
            epsilon=float(getattr(app_settings, "plateau_epsilon", 0.002)),
            patience=int(getattr(app_settings, "plateau_patience", 3)),
        ),
        runtime_root=str(paths.root),
        repository_root=str(app_settings.repository_root),
        parent_commit=_current_commit(app_settings.repository_root),
        dataset_root=str(app_settings.dataset_root) if app_settings.dataset_root else None,
        evaluator_id=app_settings.evaluator_id,
        docker_image=app_settings.docker_image,
        default_timeout_seconds=300,
    )
    controller = ProductionController(ControllerServices(transitions=transitions, store=run_store))
    graph = build_production_graph(controller, checkpointer=SqliteCheckpointer(paths.graph_db))
    return ProductionServices(
        controller, repo, graph, app_settings, worktree_manager, executor, evaluator, agents
    )


class _SyntheticWorktreeManager:
    def __init__(self, root: Path, store: RepositoryRunStore) -> None:
        self.root, self.store = root, store

    def create(self, run_id: str, spec: ExperimentSpec, parent_commit: str) -> WorktreeAssignment:
        path = self.root / "worktrees" / run_id / spec.experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return WorktreeAssignment(
            worktree_id=f"worktree-{hashlib.sha256(path.as_posix().encode()).hexdigest()[:20]}",
            run_id=run_id,
            experiment_id=spec.experiment_id,
            path=path,
            branch=f"synthetic/{run_id}/{spec.experiment_id}",
            parent_commit=parent_commit,
        )

    def register_source(
        self, assignment: WorktreeAssignment, allowed_scopes: tuple[str, ...]
    ) -> SourceRegistration:
        from tiktok2026.repository.diffs import patch_signature

        content = f"synthetic source for {assignment.experiment_id}\n".encode()
        digest = patch_signature(content.decode())
        destination = (
            self.root
            / "artifacts"
            / assignment.run_id
            / assignment.experiment_id
            / f"patch-{digest}.diff"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        self.store.put_artifact(
            ArtifactRecord(
                artifact_id=f"patch-{digest}",
                run_id=assignment.run_id,
                experiment_id=assignment.experiment_id,
                kind="source_patch",
                uri=destination.as_uri(),
                sha256=digest,
                size_bytes=len(content),
                producer="synthetic-worktree",
                retention=ArtifactRetention.PROVENANCE,
            )
        )
        return SourceRegistration(
            experiment_id=assignment.experiment_id,
            run_id=assignment.run_id,
            parent_commit=assignment.parent_commit,
            source_commit=hashlib.sha1((assignment.worktree_id + digest).encode()).hexdigest(),
            patch_sha256=digest,
            patch_artifact_id=f"patch-{digest}",
            patch_artifact_uri=destination.as_uri(),
            allowed_scopes=allowed_scopes,
            eligible=True,
        )

    def remove(self, assignment: WorktreeAssignment) -> None:
        del assignment


class _SyntheticExecutor:
    def __init__(
        self,
        root: Path,
        store: RepositoryRunStore,
        manifest: DatasetManifestIdentity,
        evaluator_hash: str,
        artifact_store: Any,
    ) -> None:
        self.root, self.store, self.manifest, self.evaluator_hash, self.artifact_store = (
            root,
            store,
            manifest,
            evaluator_hash,
            artifact_store,
        )

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        payload = json.dumps(
            {
                "schema_version": "1",
                "manifest_id": self.manifest.manifest_id,
                "manifest_sha256": self.manifest.manifest_sha256,
                "split": "valid",
                "source_commit": request.source_commit,
                "execution_id": request.execution_id,
                "rows": [],
            },
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(payload).hexdigest()
        run_id = request.run_id or "synthetic-run"
        prediction_filename = f"prediction-{digest}.json"
        prediction = self.artifact_store.publish_bytes(
            run_id,
            request.experiment_id,
            "prediction",
            prediction_filename,
            payload,
            "synthetic-executor",
            ArtifactRetention.RUN,
        )
        self.store.put_json(
            "prediction_artifact",
            prediction.artifact_id,
            PredictionArtifactRegistration(
                artifact_id=prediction.artifact_id,
                path=Path(prediction.uri.removeprefix("file://")),
                sha256=prediction.sha256,
                checkpoint_id=f"checkpoint-{digest}",
                source_commit=request.source_commit,
                execution_id=request.execution_id,
                dataset_manifest_id=self.manifest.manifest_id,
                dataset_manifest_sha256=self.manifest.manifest_sha256,
                split="valid",
            ).model_dump_json(),
        )
        checkpoint_payload = json.dumps(
            {
                "schema_version": "1",
                "checkpoint_id": f"checkpoint-{digest}",
                "data_manifest_id": self.manifest.manifest_id,
                "source_commit": request.source_commit,
                "execution_id": request.execution_id,
                "prediction_artifact": prediction_filename,
            },
            sort_keys=True,
        ).encode()
        checkpoint_digest = hashlib.sha256(checkpoint_payload).hexdigest()
        checkpoint = self.artifact_store.publish_bytes(
            run_id,
            request.experiment_id,
            "checkpoint",
            f"checkpoint-{checkpoint_digest}.json",
            checkpoint_payload,
            "synthetic-executor",
            ArtifactRetention.RUN,
        )
        return ExecutionResult(
            execution_id=request.execution_id,
            experiment_id=request.experiment_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=0,
            elapsed_seconds=0.1,
            gpu_hours=0,
            artifact_ids=(prediction.artifact_id, checkpoint.artifact_id),
            checkpoint_id=f"checkpoint-{digest}",
        )


class _ScriptedAgent:
    def __init__(self) -> None:
        self._proposal_count = 0

    async def invoke(self, request: ContractModel) -> ContractModel:
        if isinstance(request, ResearchRequest) and request.objective == "orchestrate":
            return OrchestrationDecision(
                decision_id=f"decision-{request.request_id}",
                action=DecisionAction.RESEARCH,
                rationale="fixture",
            )
        if isinstance(request, ResearchRequest):
            self._proposal_count += 1
            experiment_id = f"synthetic-exp-{self._proposal_count}"
            return ResearchDecision(
                request_id=request.request_id,
                kind="proposal",
                message="fixture",
                experiment_spec=ExperimentSpec(
                    experiment_id=experiment_id,
                    hypothesis_id=f"synthetic-hyp-{self._proposal_count}",
                    hypothesis="fixture",
                    mechanism="fixture",
                    motivation="fixture",
                    expected_signal="fixture",
                    implementation_scope=("src/tiktok2026/experiment",),
                    fidelity=Fidelity.SMOKE,
                    success_criteria="fixture",
                    failure_criteria="fixture",
                    source_provenance=("fixture-provenance",),
                ),
            )
        if isinstance(request, ImplementationRequest):
            return ImplementationResult(
                experiment_id=request.experiment_id,
                patch_artifact_id="patch-requested-by-fixture",
                changed_files=(),
            )
        if isinstance(request, ValidationRequest):
            return ValidationReport(
                report_id=f"report-{request.request_id}",
                experiment_id=request.experiment_id,
                stage=request.stage,
                verdict=ValidationVerdict.APPROVED,
                leakage_risk="none",
            )
        raise ValueError("unsupported synthetic request")


def build_synthetic_controller(
    repository_root: Path, runtime_root: Path, iterations: int = 2
) -> tuple[ProductionController, object, Any]:
    if iterations < 2:
        raise ValueError("synthetic controller requires at least two iterations")
    runtime = initialize_runtime(repository_root, runtime_root)
    repo = RepositoryTransitionStore(runtime.repository)
    manifest = DatasetManifestIdentity(
        manifest_id="synthetic-manifest",
        manifest_sha256=hashlib.sha256(b"synthetic-manifest").hexdigest(),
    )
    repo.put_dataset_manifest_identity(manifest)
    evaluator_hash = hashlib.sha256(b"synthetic-evaluator").hexdigest()
    repo.put_evaluator_identity(
        EvaluatorIdentity(
            evaluator_id="synthetic-evaluator",
            evaluator_sha256=evaluator_hash,
            validity="provisional",
        )
    )
    agent_clients = {role: _ScriptedAgent() for role in AgentRole}
    from tiktok2026.persistence.artifacts import ArtifactStore
    from tiktok2026.testing.synthetic import FixtureEvaluator

    artifact_store = ArtifactStore(runtime.paths, runtime.repository)

    transitions = make_service_transitions(
        agent_clients=agent_clients,
        evaluator=FixtureEvaluator(),
        executor=_SyntheticExecutor(
            runtime.paths.root, repo, manifest, evaluator_hash, artifact_store
        ),
        worktree_manager=_SyntheticWorktreeManager(runtime.paths.root, repo),
        resource_accountant=LedgerResourceAccountant(
            ResourceLedger(
                runtime.paths.application_db,
                ResourceState(
                    remaining_gpu_hours=100,
                    accumulated_gpu_hours=0,
                    remaining_wall_seconds=3600,
                    used_tokens=0,
                    remaining_tokens=100000,
                    disk_bytes_available=1 << 30,
                    reserved_final_gpu_hours=10,
                ),
            )
        ),
        policy_gate=DeterministicPolicyGate(),
        run_store=repo,
        export_service=RepositoryExportService(runtime.repository, runtime.paths.root),
        bundle_service=RepositoryFinalizationBundleService(runtime.repository, runtime.paths.root),
        frontier_service=RepositoryFrontierService(
            runtime.repository, patience=max(1, iterations - 1)
        ),
        runtime_root=str(runtime.paths.root),
        repository_root=str(repository_root),
        parent_commit=hashlib.sha1(b"synthetic-parent").hexdigest(),
        dataset_root=str(runtime.paths.root),
        evaluator_id="synthetic-evaluator",
    )
    controller = ProductionController(ControllerServices(transitions=transitions, store=repo))
    graph = build_production_graph(
        controller, checkpointer=SqliteCheckpointer(runtime.paths.graph_db)
    )
    return controller, repo, graph
