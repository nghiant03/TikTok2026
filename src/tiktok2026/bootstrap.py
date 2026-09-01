from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from loguru import logger

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
from tiktok2026.agents.research.online import OpenAIWebSearchProvider
from tiktok2026.benchmark.kuaireand_pure.manifest import BenchmarkManifest, verify_protected_files
from tiktok2026.contracts import (
    DEFAULT_IMPLEMENTATION_CRITERIA,
    AgentRole,
    ArtifactRecord,
    ArtifactRetention,
    AuditEvent,
    BaselineCalibrationRecord,
    ContractModel,
    CriterionAssessmentStatus,
    DatasetManifestIdentity,
    DatasetViewProvenance,
    DatasetViewRow,
    DecisionAction,
    EvaluatorIdentity,
    ExecutionRequest,
    ExecutionResult,
    ExperimentSpec,
    FailureKind,
    Fidelity,
    FinalizationBundleRequest,
    ImplementationCriterionAssessment,
    ImplementationCriterionId,
    ImplementationRequest,
    ImplementationResourceEstimate,
    ImplementationResult,
    OperationResult,
    OrchestrationDecision,
    OrchestrationRequest,
    PredictionArtifactRegistration,
    ProvisionalFinalizationRequest,
    ResearchDecision,
    ResearchRequest,
    ResourceState,
    RunBaselineBinding,
    RunPhase,
    RunRecord,
    RunStore,
    RuntimePaths,
    SourceRegistration,
    ValidationReport,
    ValidationRequest,
    ValidationStage,
    ValidationVerdict,
    WorktreeAssignment,
)
from tiktok2026.controller import ControllerServices, ProductionController
from tiktok2026.evaluation.registry import evaluator_implementation_sha256
from tiktok2026.graph.build import build_production_graph
from tiktok2026.graph.state import ProductionState
from tiktok2026.persistence.checkpointer import SqliteCheckpointer
from tiktok2026.persistence.migrations import MigrationRunner
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.persistence.resources import ResourceLedger
from tiktok2026.recovery import (
    RecoveryCandidate,
    reconcile_recovery,
    validate_pre_registration_assignment,
)
from tiktok2026.use_cases import (
    ServiceTransitions,
    closure_updates_without_agents,
    make_service_transitions,
)


@dataclass(frozen=True)
class RuntimeServices:
    repository_root: Path
    paths: RuntimePaths
    repository: ApplicationRepository


class _CriterionAwareRepositoryTransitionStore(RepositoryTransitionStore):
    """Transition-store seam with the criterion history port required by policy."""

    def get_criterion_repeat_count(
        self, experiment_id: str, criterion_id: ImplementationCriterionId | str
    ) -> int:
        return self._repo.get_criterion_repeat_count(experiment_id, str(criterion_id))


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


def _role_prompt(role: AgentRole) -> str:
    prompt_path = Path(__file__).parent / "agents" / role.value / "prompt.md"
    return prompt_path.read_text(encoding="utf-8")


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
    resource_ledger: ResourceLedger | None = None


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
    store = RepositoryRunStore(repository)
    closure = store.get_run_closure(run_id)
    if closure is None or closure.champion is None:
        raise ValueError("finalization requires a closure with an eligible champion")
    champion = closure.champion
    observation = store.get_scored_observation(champion.observation_id)
    if observation is None:
        raise ValueError("finalization champion observation is unavailable")
    experiment_id = observation.experiment_id
    evaluation = store.get_evaluation_result(champion.evaluation_id)
    source = store.get_source_registration_by_id(f"source-{champion.source_commit}")
    if source is None or evaluation is None:
        raise ValueError("finalization provenance is unavailable")
    if (
        observation.run_id != run_id
        or observation.evaluation_id != evaluation.evaluation_id
        or observation.checkpoint_id != champion.checkpoint_id
        or observation.source_commit != source.source_commit
        or evaluation.run_id != run_id
        or evaluation.experiment_id != experiment_id
        or evaluation.checkpoint_id != champion.checkpoint_id
        or source.run_id != run_id
        or source.experiment_id != experiment_id
    ):
        raise ValueError("finalization provenance does not match closure champion")
    existing = repository.get_finalization(finalization_id)
    if existing is not None:
        if (
            existing.run_id != run_id
            or existing.experiment_id != experiment_id
            or existing.source_commit != source.source_commit
            or existing.checkpoint_id != evaluation.checkpoint_id
            or existing.evaluation_id != evaluation.evaluation_id
        ):
            raise ValueError("persisted finalization is not bound to the closure champion")
        return existing
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


@contextmanager
def _exclusive_runtime_run(runtime_root: Path, run_id: str) -> Generator[None, None, None]:
    locks = runtime_root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    marker = locks / f"{run_id}.lock"
    with (locks / "controller.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OperationalError("another production run owns the runtime root") from error
        handle.seek(0)
        handle.truncate()
        handle.write(run_id)
        handle.flush()
        marker.write_text(run_id, encoding="utf-8")
        try:
            yield
        finally:
            marker.unlink(missing_ok=True)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_baseline_calibration(runtime_root: Path) -> Generator[None, None, None]:
    locks = runtime_root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / "baseline-calibration.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
        baseline_calibrator: Callable[..., tuple[BaselineCalibrationRecord, bool]] | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.runtime_root = runtime_root.resolve()
        self.profile_path = profile_path
        self.operator_config = operator_config
        self.baseline_calibrator = baseline_calibrator

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

    def calibrate_baseline(self) -> OperationResult:
        settings = self._production_settings()
        if settings.dataset_root is None:
            raise OperationalError("baseline calibration requires a configured dataset_root")
        services = initialize_runtime(self.repository_root, self.runtime_root)
        record, created = self._ensure_current_baseline(
            services.repository,
            settings.dataset_root,
            actor_type="human",
            actor_id="cli-operator",
        )
        metrics = {metric.name: metric.value for metric in record.evaluation.metrics}
        diagnostics = {metric.name: metric.value for metric in record.diagnostic_metrics}
        return _result(
            "calibrate-baseline",
            status="created" if created else "cached",
            values={
                "calibration_id": record.calibration_id,
                "split": record.split,
                "GAUC": metrics["GAUC"],
                "nDCG@5": metrics["nDCG@5"],
                "composite": record.evaluation.validation_score,
                "diagnostic_GAUC": diagnostics["GAUC"],
                "diagnostic_nDCG@5": diagnostics["nDCG@5"],
                "diagnostic_primary": diagnostics["primary"],
                "prediction_artifact_uri": record.prediction_artifact_uri,
            },
        )

    def _ensure_current_baseline(
        self,
        repository: ApplicationRepository,
        dataset_root: Path,
        *,
        actor_type: Literal["agent", "controller", "human"] = "controller",
        actor_id: str = "production-operations",
    ) -> tuple[BaselineCalibrationRecord, bool]:
        """Load or create the current Starter Kit calibration without retraining on cache hits."""
        from tiktok2026.benchmark.kuaireand_pure.calibration import calibrate_baseline

        with _exclusive_baseline_calibration(self.runtime_root):
            verify_manifests(self.repository_root)
            calibrator = self.baseline_calibrator or calibrate_baseline
            existing = tuple(
                record.model_dump_json()
                for record in repository.list_baseline_calibrations()
            )
            record, created = calibrator(
                self.repository_root,
                self.runtime_root,
                dataset_root,
                existing,
            )
            repository.put_baseline_calibration(
                record,
                actor_type=actor_type,
                actor_id=actor_id,
            )
        return record, created

    @staticmethod
    def _binding(run_id: str, calibration: BaselineCalibrationRecord) -> RunBaselineBinding:
        return RunBaselineBinding(
            run_id=run_id,
            calibration_id=calibration.calibration_id,
            baseline_evaluation_id=calibration.evaluation.evaluation_id,
            dataset_manifest_id=calibration.dataset_manifest_id,
            dataset_manifest_sha256=calibration.dataset_manifest_sha256,
            evaluator_id=calibration.evaluator_id,
            evaluator_sha256=calibration.evaluator_sha256,
            split=calibration.split,
            metrics=calibration.evaluation.metrics,
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
        logger.info(
            "Run starting synthetic={} runtime_root={}",
            synthetic,
            self.runtime_root,
        )
        ledger: ResourceLedger | None = None
        baseline: RunBaselineBinding | None = None
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
            ledger = services.resource_ledger
            dataset_root = settings.dataset_root
            if dataset_root is None:
                raise OperationalError("baseline calibration requires a configured dataset_root")
            calibration, _created = self._ensure_current_baseline(repository, dataset_root)
            baseline = self._binding(actual_run_id, calibration)
        run_lock = (
            _exclusive_runtime_run(self.runtime_root, actual_run_id)
            if not synthetic
            else nullcontext()
        )
        with run_lock:
            if ledger is not None:
                ledger.claim_run(actual_run_id)
            try:
                if not synthetic:
                    assert baseline is not None
                    RepositoryRunStore(repository).put_run_baseline(baseline)
                    repository.put_run(
                        RunRecord(run_id=actual_run_id, status="active"),
                        f"{actual_run_id}-active",
                        None,
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
                            **(
                                {"baseline_binding": baseline.model_dump(mode="json")}
                                if baseline is not None
                                else {}
                            ),
                        },
                    )
                )
                state = asyncio.run(
                    graph.ainvoke(
                        _initial_state(actual_run_id),
                        {"configurable": {"thread_id": actual_run_id}},
                    )
                )
            except Exception as error:
                raise OperationalError(str(error)) from error
            finally:
                if ledger is not None:
                    ledger.release_run(actual_run_id)
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
        lock = (
            _exclusive_runtime_run(self.runtime_root, run_id)
            if not synthetic
            else nullcontext()
        )
        with lock:
            return self._resume_locked(run_id, synthetic=synthetic)

    def _resume_locked(self, run_id: str, *, synthetic: bool = False) -> OperationResult:
        repository = initialize_runtime(self.repository_root, self.runtime_root).repository
        if not synthetic:
            repository.adopt_legacy_lifecycle(run_id)
        checkpoint = self._load_checkpoint(run_id)
        if checkpoint is None:
            raise OperationalError(f"no durable checkpoint exists for run {run_id}")
        state = _checkpoint_state(checkpoint)
        phase = str(state.get("phase"))
        store = RepositoryRunStore(repository)
        closure = store.get_run_closure(run_id)
        checkpointer = SqliteCheckpointer(self.runtime_root / "graph.sqlite3")
        closure_updates: dict[str, object] | None = None
        if closure is not None and phase not in {RunPhase.COMPLETE.value, str(RunPhase.COMPLETE)}:
            try:
                closure_updates = closure_updates_without_agents(
                    ServiceTransitions(run_store=cast(RunStore, store)),
                    cast(ProductionState, state),
                    closure,
                )
                durable = asyncio.run(
                    checkpointer.aget_tuple({"configurable": {"thread_id": run_id}})
                )
                if durable is None:
                    raise OperationalError("durable checkpoint disappeared during closure recovery")
                asyncio.run(
                    cast(Any, checkpointer).aupdate_state(
                        cast(dict[str, object], durable.config),
                        closure_updates,
                        as_node="update_frontier",
                    )
                )
                if closure.champion is None:
                    self._resume_audit(
                        repository,
                        run_id,
                        True,
                        "closed without an eligible scored observation",
                    )
                    return _result(
                        "resume",
                        run_id=run_id,
                        phase=RunPhase.COMPLETE,
                        status="resumed",
                        values={
                            "pending_route": "complete",
                            "state_version": state.get("state_version", 0),
                        },
                    )
            except Exception as error:
                self._resume_audit(repository, run_id, False, str(error))
                raise OperationalError(str(error)) from error
            state = {**state, **closure_updates}
            phase = str(state.get("phase"))
        baseline: RunBaselineBinding | None = None
        settings: Any | None = None
        if not synthetic:
            try:
                production_settings = self._production_settings()
                settings = production_settings
                dataset_root = production_settings.dataset_root
                if dataset_root is None:
                    raise OperationalError(
                        "baseline calibration requires a configured dataset_root"
                    )
                calibration, _created = self._ensure_current_baseline(repository, dataset_root)
                baseline = self._binding(run_id, calibration)
                RepositoryRunStore(repository).put_run_baseline(baseline)
            except Exception as error:
                self._resume_audit(repository, run_id, False, str(error))
                raise OperationalError(str(error)) from error
        if phase in {RunPhase.COMPLETE.value, str(RunPhase.COMPLETE)}:
            self._resume_audit(repository, run_id, True, "run is already complete", baseline)
            return _result(
                "resume",
                run_id=run_id,
                phase=RunPhase.COMPLETE,
                status="already_complete",
                values={"resumed": False},
            )
        if not synthetic and not (closure is not None and closure.champion is not None):
            self._reconcile_resume_boundary(repository, run_id, state)
        if synthetic:
            _controller, _store, graph = build_synthetic_controller(
                self.repository_root, self.runtime_root
            )
        else:
            assert settings is not None
            services = build_production_services(settings)
            if closure is None or closure.champion is None:
                self._bind_resumed_implementor(services, repository, state)
            graph = services.graph
        try:
            result = asyncio.run(graph.ainvoke(None, {"configurable": {"thread_id": run_id}}))
        except Exception as error:
            self._resume_audit(repository, run_id, False, str(error), baseline)
            raise OperationalError(str(error)) from error
        self._resume_audit(repository, run_id, True, "checkpoint resumed", baseline)
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

    def recover_source_registration(self, run_id: str) -> OperationResult:
        with _exclusive_runtime_run(self.runtime_root, run_id):
            return self._recover_source_registration(run_id)

    def recover_execution_result(self, run_id: str, execution_id: str) -> OperationResult:
        with _exclusive_runtime_run(self.runtime_root, run_id):
            return self._recover_execution_result(run_id, execution_id)

    def retry_execution(self, run_id: str, failed_execution_id: str) -> OperationResult:
        with _exclusive_runtime_run(self.runtime_root, run_id):
            from tiktok2026.recovery import validate_execution_recovery_state

            repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
            store = RepositoryRunStore(repository)
            failed = store.get_execution_result(failed_execution_id)
            if failed is None or failed.failure_kind is None:
                raise OperationalError("failed execution result is unavailable")
            registration = store.get_source_registration_by_id(failed.source_registration_id)
            if (
                registration is None
                or registration.run_id != run_id
                or registration.experiment_id != failed.experiment_id
                or registration.source_commit != failed.source_commit
            ):
                raise OperationalError("run is not eligible for execution retry")
            checkpointer = SqliteCheckpointer(self.runtime_root / "graph.sqlite3")

            async def _find_checkpoint() -> tuple[Any | None, int]:
                target = None
                maximum_state_version = 0
                async for candidate in checkpointer.alist(
                    {"configurable": {"thread_id": run_id}}, limit=100
                ):
                    state = _checkpoint_state(candidate.checkpoint)
                    state_version = state.get("state_version")
                    if isinstance(state_version, int):
                        maximum_state_version = max(maximum_state_version, state_version)
                    if target is None and validate_execution_recovery_state(
                        state, run_id, failed, registration
                    ).resumable:
                        target = candidate
                return target, maximum_state_version

            target, state_version = asyncio.run(_find_checkpoint())
            if target is None:
                raise OperationalError("eligible pre-execution checkpoint was not found")
            target_state = _checkpoint_state(target.checkpoint)
            experiment_id = failed.experiment_id
            retry_execution_id = f"execution-{run_id}-{experiment_id}-{state_version}"
            if (
                store.get_execution_result(retry_execution_id) is not None
                or store.load_transition(run_id, state_version + 1) is not None
            ):
                raise OperationalError("execution retry identity is not fresh")

            services = build_production_services(self._production_settings())
            config = asyncio.run(
                services.graph.aupdate_state(
                    cast(dict[str, object], target.config),
                    {
                        "phase": RunPhase.EXECUTE,
                        "pending_route": "execute",
                        "state_version": state_version,
                        "latest_execution_result_id": None,
                        "latest_evaluation_result_id": None,
                        "terminal_reason": None,
                    },
                    as_node="preflight",
                )
            )
            configurable = config.get("configurable")
            checkpoint_id = (
                cast(dict[str, object], configurable).get("checkpoint_id")
                if isinstance(configurable, dict)
                else None
            )
            repository.put_audit_event(
                AuditEvent(
                    event_id=f"execution-retry-{run_id}-{uuid.uuid4().hex[:8]}",
                    run_id=run_id,
                    experiment_id=experiment_id,
                    event_type="execution_retry_accepted",
                    actor_type="human",
                    actor_id="cli-operator",
                    payload={
                        "failed_execution_id": failed_execution_id,
                        "retry_execution_id": retry_execution_id,
                        "failure_kind": failed.failure_kind.value,
                        "source_registration_id": registration.registration_id,
                        "target_state_version": target_state.get("state_version"),
                        "retry_state_version": state_version,
                        "checkpoint_id": checkpoint_id,
                    },
                )
            )
            return _result(
                "retry_execution",
                run_id=run_id,
                phase=RunPhase.EXECUTE,
                status="recovered",
                values={
                    "pending_route": "execute",
                    "failed_execution_id": failed_execution_id,
                    "retry_execution_id": retry_execution_id,
                    "target_state_version": target_state.get("state_version"),
                    "retry_state_version": state_version,
                },
            )

    def _recover_execution_result(self, run_id: str, execution_id: str) -> OperationResult:
        from tiktok2026.recovery import validate_execution_recovery_state

        repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        store = RepositoryRunStore(repository)
        execution = store.get_execution_result(execution_id)
        if execution is None:
            raise OperationalError("authoritative execution result is unavailable")
        registration = store.get_source_registration_by_id(execution.source_registration_id)
        if registration is None:
            raise OperationalError("execution source registration is unavailable")
        for artifact_id in execution.artifact_ids:
            artifact = store.get_artifact(artifact_id)
            if (
                artifact is None
                or artifact.run_id != run_id
                or artifact.experiment_id != execution.experiment_id
            ):
                raise OperationalError("execution artifact authority is unavailable")

        checkpointer = SqliteCheckpointer(self.runtime_root / "graph.sqlite3")

        async def _find_checkpoint() -> tuple[Any | None, int]:
            target = None
            maximum_state_version = 0
            async for candidate in checkpointer.alist(
                {"configurable": {"thread_id": run_id}}, limit=100
            ):
                state = _checkpoint_state(candidate.checkpoint)
                state_version = state.get("state_version")
                if isinstance(state_version, int):
                    maximum_state_version = max(maximum_state_version, state_version)
                validation = validate_execution_recovery_state(
                    state, run_id, execution, registration
                )
                if target is None and validation.resumable:
                    target = candidate
            return target, maximum_state_version

        target, maximum_state_version = asyncio.run(_find_checkpoint())
        if target is None:
            raise OperationalError("eligible execution checkpoint was not found")
        target_state = _checkpoint_state(target.checkpoint)
        target_config = cast(dict[str, object], target.config)
        services = build_production_services(self._production_settings())
        ledger = services.resource_ledger
        reservation_id = f"reservation-{execution.execution_id}"
        if ledger is None or not ledger.consume(
            reservation_id,
            gpu_hours=execution.gpu_hours,
            wall_seconds=execution.elapsed_seconds,
            tokens=0,
            disk_bytes=0,
        ):
            raise OperationalError("execution resource settlement is unavailable")
        ledger.reconcile(
            reservation_id,
            gpu_hours=execution.gpu_hours,
            wall_seconds=execution.elapsed_seconds,
            tokens=0,
            disk_bytes=0,
        )
        if execution.failure_kind is None:
            pending_route = "evaluate"
            terminal_reason = None
        else:
            detail = json.dumps(
                {
                    "kind": execution.failure_kind.value,
                    "message": "execution failed",
                    "evidence": [execution.execution_id],
                },
                sort_keys=True,
            )
            pending_route = "persist_failure"
            terminal_reason = f"failure:{detail}"
        config = asyncio.run(
            services.graph.aupdate_state(
                target_config,
                {
                    "phase": RunPhase.EXECUTE,
                    "pending_route": pending_route,
                    "state_version": maximum_state_version,
                    "latest_execution_result_id": execution.execution_id,
                    "latest_evaluation_result_id": None,
                    "terminal_reason": terminal_reason,
                },
                as_node="execute",
            )
        )
        configurable = config.get("configurable")
        checkpoint_id = (
            cast(dict[str, object], configurable).get("checkpoint_id")
            if isinstance(configurable, dict)
            else None
        )
        target_checkpoint_id = cast(dict[str, object], target_config.get("configurable", {})).get(
            "checkpoint_id"
        )
        repository.put_audit_event(
            AuditEvent(
                event_id=f"execution-result-recovery-{run_id}-{uuid.uuid4().hex[:8]}",
                run_id=run_id,
                experiment_id=execution.experiment_id,
                event_type="execution_result_recovery_accepted",
                actor_type="human",
                actor_id="cli-operator",
                payload={
                    "execution_id": execution.execution_id,
                    "source_registration_id": registration.registration_id,
                    "source_commit": registration.source_commit,
                    "target_state_version": target_state.get("state_version"),
                    "recovery_state_version": maximum_state_version,
                    "pending_route": pending_route,
                    "target_checkpoint_id": target_checkpoint_id,
                    "checkpoint_id": checkpoint_id,
                },
            )
        )
        return _result(
            "recover_execution_result",
            run_id=run_id,
            phase=RunPhase.EXECUTE,
            status="recovered",
            values={
                "pending_route": pending_route,
                "execution_id": execution.execution_id,
                "source_registration_id": registration.registration_id,
                "target_state_version": target_state.get("state_version"),
                "recovery_state_version": maximum_state_version,
            },
        )

    def _recover_source_registration(self, run_id: str) -> OperationResult:
        checkpoint = self._load_checkpoint(run_id)
        if checkpoint is None:
            raise OperationalError(f"no durable checkpoint exists for run {run_id}")
        state = _checkpoint_state(checkpoint)
        experiment_id = str(state.get("current_experiment_id") or "")
        repository = ApplicationRepository(self.runtime_root / "application.sqlite3")
        store = RepositoryRunStore(repository)
        assignment = store.get_worktree_assignment(experiment_id) if experiment_id else None
        source = store.get_source_registration(experiment_id) if experiment_id else None
        if assignment is None or source is None:
            raise OperationalError("source-registration recovery authority is unavailable")

        events = repository.list_audit_events(run_id)

        def _approved_registration_transition(event: AuditEvent) -> bool:
            updates = event.payload.get("updates")
            return (
                event.event_type == "controller_transition"
                and event.payload.get("operation") == "implementation_validation"
                and isinstance(updates, dict)
                and cast(dict[str, object], updates).get("pending_route")
                == "register_source"
            )

        failure = next(
            (
                event
                for event in reversed(events)
                if event.event_type == "failure_persisted"
                and event.experiment_id == experiment_id
            ),
            None,
        )
        validation = next(
            (
                event
                for event in reversed(events)
                if _approved_registration_transition(event)
            ),
            None,
        )
        failure_evidence = failure.payload.get("evidence_refs") if failure is not None else None
        if (
            failure is None
            or failure_evidence != ["source worktree is not clean after registration"]
            or validation is None
        ):
            raise OperationalError("run is not eligible for source-registration recovery")

        result = validate_pre_registration_assignment(
            assignment,
            self.runtime_root,
            lambda commit: self._approved_parent_for_resume(commit),
            source.source_commit,
        )
        if not result.resumable:
            raise OperationalError(result.reason)

        services = build_production_services(self._production_settings())
        config = asyncio.run(
            services.graph.aupdate_state(
                {"configurable": {"thread_id": run_id}},
                {
                    "phase": RunPhase.IMPLEMENT,
                    "pending_route": "register_source",
                    "terminal_reason": None,
                },
                as_node="implementation_validation",
            )
        )
        configurable = config.get("configurable")
        checkpoint_id = (
            cast(dict[str, object], configurable).get("checkpoint_id")
            if isinstance(configurable, dict)
            else None
        )
        repository.put_audit_event(
            AuditEvent(
                event_id=f"source-registration-recovery-{run_id}-{uuid.uuid4().hex[:8]}",
                run_id=run_id,
                experiment_id=experiment_id,
                event_type="source_registration_recovery_accepted",
                actor_type="human",
                actor_id="cli-operator",
                payload={
                    "prior_registration_id": source.registration_id,
                    "prior_revision": source.revision,
                    "validation_event_id": validation.event_id,
                    "failure_event_id": failure.event_id,
                    "checkpoint_id": checkpoint_id,
                },
            )
        )
        return _result(
            "recover_source_registration",
            run_id=run_id,
            phase=RunPhase.IMPLEMENT,
            status="recovered",
            values={
                "pending_route": "register_source",
                "prior_registration_id": source.registration_id,
                "prior_revision": source.revision,
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
        self,
        repository: ApplicationRepository,
        run_id: str,
        accepted: bool,
        reason: str,
        baseline: RunBaselineBinding | None = None,
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
                payload={
                    "reason": reason,
                    **(
                        {"baseline_binding": baseline.model_dump(mode="json")}
                        if baseline is not None
                        else {}
                    ),
                },
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
        stale_reservation_id = self._reservation_id(run_id)
        if str(state.get("pending_route")) == "execute":
            state_version = state.get("state_version")
            execution_id = f"execution-{run_id}-{experiment_id}-{state_version}"
            execution = store.get_execution_result(execution_id)
            if (
                stale_reservation_id == f"reservation-{execution_id}"
                and execution is not None
                and execution.source_registration_id == source.registration_id
                and execution.source_commit == source.source_commit
            ):
                stale_reservation_id = None
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
            stale_reservation_id=stale_reservation_id,
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
            "persist_failure",
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
            store = RepositoryRunStore(repository)
            assignment = store.get_worktree_assignment(experiment_id) if experiment_id else None
            source = store.get_source_registration(experiment_id) if experiment_id else None
            if assignment is None:
                reason = "pre-registration worktree assignment is unavailable"
                self._resume_audit(repository, run_id, False, reason)
                raise OperationalError(reason)
            result = validate_pre_registration_assignment(
                assignment,
                self.runtime_root,
                lambda commit: self._approved_parent_for_resume(commit),
                source.source_commit if source is not None else None,
                allow_pending_commit=route == "register_source",
            )
            if not result.resumable:
                self._resume_audit(repository, run_id, False, result.reason)
                raise OperationalError(result.reason)
            return
        self._reconcile_late_resume(repository, run_id, state)

    @staticmethod
    def _bind_resumed_implementor(
        services: ProductionServices,
        repository: ApplicationRepository,
        state: dict[str, object],
    ) -> None:
        route = str(state.get("pending_route") or "")
        resumable_implementation_routes = {
            "implement",
            "diff_policy",
            "implementation_validation",
            "register_source",
        }
        if route not in resumable_implementation_routes:
            return
        experiment_id = str(state.get("current_experiment_id") or "")
        store = RepositoryRunStore(repository)
        assignment = store.get_worktree_assignment(experiment_id) if experiment_id else None
        spec = store.get_experiment(experiment_id) if experiment_id else None
        implementor = services.agent_clients.get(AgentRole.IMPLEMENTOR)
        if assignment is None or spec is None or implementor is None:
            raise OperationalError("resumed implementor authority is unavailable")
        implementor.bind_worktree(assignment.path, spec.implementation_scope)

    def _approved_parent_for_resume(self, commit: str) -> bool:
        try:
            subprocess.run(
                (
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    f"{commit}^{{commit}}",
                    "HEAD",
                ),
                cwd=self.repository_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        return True

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
    baseline_calibrator: Callable[..., tuple[BaselineCalibrationRecord, bool]] | None = None,
) -> ProductionOperations:
    """Build the sole operator-facing composition used by the CLI."""
    return ProductionOperations(
        repository_root,
        runtime_root,
        profile_path,
        operator_config,
        baseline_calibrator,
    )


class _ExecutionArtifactContractError(ValueError):
    def __init__(self, message: str, failure_kind: FailureKind) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


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
            DockerResourceTelemetry,
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
            resource_telemetry=DockerResourceTelemetry(),
        )
        # The dataset provider is the authority for the exact staged view.  Bind
        # that identity before launch so the training process writes it into its
        # envelopes, and reject a result that claims to have executed another
        # view.  ``provenance`` is optional on test doubles and on non-production
        # adapters; the execution result remains authoritative in that case.
        provenance = getattr(self.dataset_provider, "provenance", None)
        if callable(provenance):
            current_view = provenance(request)
            if not isinstance(current_view, DatasetViewProvenance):
                raise _ExecutionArtifactContractError(
                    "dataset provider returned invalid view provenance",
                    FailureKind.SCHEMA_MISMATCH,
                )
            if (
                request.dataset_view_sha256 is not None
                and request.dataset_view_sha256 != current_view.view_sha256
            ):
                raise _ExecutionArtifactContractError(
                    "execution request dataset view does not match the authorized view",
                    FailureKind.SCHEMA_MISMATCH,
                )
            request = request.model_copy(
                update={"dataset_view_sha256": current_view.view_sha256}
            )
        result = await executor.execute(request)
        if (
            request.dataset_view_sha256 is not None
            and result.dataset_view_sha256 != request.dataset_view_sha256
        ):
            error = _ExecutionArtifactContractError(
                "execution result dataset view does not match the request",
                FailureKind.SCHEMA_MISMATCH,
            )
            return result.model_copy(
                update={
                    "exit_code": 1,
                    "failure_kind": error.failure_kind,
                    "failure_message": str(error),
                }
            )
        if result.exit_code != 0:
            return result
        if request.execution_kind == "smoke":
            try:
                self._validate_smoke_artifacts(request, result)
            except _ExecutionArtifactContractError as error:
                return result.model_copy(
                    update={
                        "exit_code": 1,
                        "failure_kind": error.failure_kind,
                        "failure_message": str(error),
                    }
                )
            return result.model_copy(update={"smoke_output_valid": True})
        try:
            return self._register_training_artifacts(request, result)
        except _ExecutionArtifactContractError as error:
            # The process completed, but its required output contract did not.
            # Return a typed failed result so the controller persists and routes
            # it like every other execution failure.
            return result.model_copy(
                update={
                    "exit_code": 1,
                    "failure_kind": error.failure_kind,
                    "failure_message": str(error),
                }
            )

    def _validate_smoke_artifacts(
        self, request: ExecutionRequest, result: ExecutionResult
    ) -> None:
        dataset_identity = RepositoryRunStore(self.repository).get_dataset_manifest_identity()
        if dataset_identity is None or (
            request.dataset_manifest_sha256 != dataset_identity.manifest_sha256
        ):
            raise _ExecutionArtifactContractError(
                "smoke dataset provenance is unavailable", FailureKind.SCHEMA_MISMATCH
            )
        expected_rows = result.dataset_valid_rows
        if not expected_rows or len(expected_rows) > 32:
            raise _ExecutionArtifactContractError(
                "smoke valid-view expectation is absent or unbounded",
                FailureKind.SCHEMA_MISMATCH,
            )
        if (
            result.dataset_manifest_id != dataset_identity.manifest_id
            or result.dataset_manifest_sha256 != dataset_identity.manifest_sha256
            or result.dataset_view_sha256 is None
            or (
                request.dataset_view_sha256 is not None
                and result.dataset_view_sha256 != request.dataset_view_sha256
            )
        ):
            raise _ExecutionArtifactContractError(
                "smoke result dataset provenance is invalid", FailureKind.SCHEMA_MISMATCH
            )
        output = request.output_path.resolve()
        predictions_path = output / "predictions.json"
        checkpoint_path = output / "checkpoint_bundle.json"
        if not predictions_path.is_file() or not checkpoint_path.is_file():
            raise _ExecutionArtifactContractError(
                "smoke execution did not produce required outputs", FailureKind.MISSING_PATH
            )
        try:
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise _ExecutionArtifactContractError(
                "smoke outputs are not valid JSON", FailureKind.SCHEMA_MISMATCH
            ) from error
        if not isinstance(predictions, dict) or not isinstance(checkpoint, dict):
            raise _ExecutionArtifactContractError(
                "smoke outputs must be JSON objects", FailureKind.SCHEMA_MISMATCH
            )
        prediction_payload = cast(dict[str, object], predictions)
        checkpoint_payload = cast(dict[str, object], checkpoint)
        required_prediction = {
            "manifest_id": dataset_identity.manifest_id,
            "manifest_sha256": request.dataset_manifest_sha256,
            "dataset_view_sha256": result.dataset_view_sha256,
            "source_commit": request.source_commit,
            "execution_id": request.execution_id,
            "split": "valid",
        }
        if any(prediction_payload.get(key) != value for key, value in required_prediction.items()):
            raise _ExecutionArtifactContractError(
                "smoke prediction provenance is invalid", FailureKind.SCHEMA_MISMATCH
            )
        rows: object = prediction_payload.get("rows")
        if not isinstance(rows, list) or len(cast(list[object], rows)) != len(expected_rows):
            raise _ExecutionArtifactContractError(
                "smoke predictions do not match the staged valid view",
                FailureKind.SCHEMA_MISMATCH,
            )
        seen_row_ids: set[str] = set()
        for raw, expected in zip(cast(list[object], rows), expected_rows, strict=True):
            if not isinstance(raw, dict):
                raise _ExecutionArtifactContractError(
                    "smoke prediction row is invalid", FailureKind.SCHEMA_MISMATCH
                )
            row = cast(dict[str, object], raw)
            if set(row) != {"row_id", "row_identity", "user_id", "item_id", "score"}:
                raise _ExecutionArtifactContractError(
                    "smoke prediction fields are invalid", FailureKind.SCHEMA_MISMATCH
                )
            score = row.get("score")
            if (
                row.get("row_id") != expected.row_id
                or row.get("row_identity") != list(expected.row_identity)
                or row.get("user_id") != expected.user_id
                or row.get("item_id") != expected.item_id
                or expected.row_id in seen_row_ids
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise _ExecutionArtifactContractError(
                    "smoke predictions do not match the staged valid view",
                    FailureKind.SCHEMA_MISMATCH,
                )
            seen_row_ids.add(expected.row_id)
        prediction_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        required_checkpoint = {
            "data_manifest_id": dataset_identity.manifest_id,
            "source_commit": request.source_commit,
            "execution_id": request.execution_id,
            "fidelity": "smoke",
            "prediction_artifact": predictions_path.name,
            "prediction_sha256": prediction_sha256,
            "dataset_view_sha256": result.dataset_view_sha256,
        }
        if any(checkpoint_payload.get(key) != value for key, value in required_checkpoint.items()):
            raise _ExecutionArtifactContractError(
                "smoke checkpoint provenance is invalid", FailureKind.SCHEMA_MISMATCH
            )
        checkpoint_id = checkpoint_payload.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise _ExecutionArtifactContractError(
                "smoke checkpoint identity is absent", FailureKind.SCHEMA_MISMATCH
            )

    def _register_training_artifacts(
        self, request: ExecutionRequest, result: ExecutionResult
    ) -> ExecutionResult:
        if request.run_id is None:
            raise ValueError("execution run identity is unavailable")
        dataset_identity = RepositoryRunStore(self.repository).get_dataset_manifest_identity()
        if dataset_identity is None:
            raise ValueError("verified dataset manifest identity is unavailable")
        if request.dataset_manifest_sha256 != dataset_identity.manifest_sha256:
            raise ValueError("execution dataset identity does not match registered authority")
        output = request.output_path.resolve()
        predictions_path = output / "predictions.json"
        checkpoint_path = output / "checkpoint_bundle.json"
        if not predictions_path.is_file() or not checkpoint_path.is_file():
            raise _ExecutionArtifactContractError(
                "execution did not produce prediction and checkpoint artifacts",
                FailureKind.MISSING_PATH,
            )
        try:
            predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise _ExecutionArtifactContractError(
                "execution artifacts are not valid JSON", FailureKind.SCHEMA_MISMATCH
            ) from error
        if not isinstance(predictions, dict) or not isinstance(checkpoint, dict):
            raise _ExecutionArtifactContractError(
                "execution artifacts must be JSON objects", FailureKind.SCHEMA_MISMATCH
            )
        prediction_payload = cast(dict[str, object], predictions)
        checkpoint_payload = cast(dict[str, object], checkpoint)
        prediction_bytes = predictions_path.read_bytes()
        prediction_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
        if result.dataset_view_sha256 is None:
            raise _ExecutionArtifactContractError(
                "execution result dataset view provenance is absent",
                FailureKind.SCHEMA_MISMATCH,
            )
        if (
            request.dataset_view_sha256 is not None
            and request.dataset_view_sha256 != result.dataset_view_sha256
        ):
            raise _ExecutionArtifactContractError(
                "execution result dataset view does not match the request",
                FailureKind.SCHEMA_MISMATCH,
            )
        expected = {
            "manifest_id": dataset_identity.manifest_id,
            "manifest_sha256": request.dataset_manifest_sha256,
            "source_commit": request.source_commit,
            "execution_id": request.execution_id,
            "dataset_view_sha256": result.dataset_view_sha256,
        }
        if any(prediction_payload.get(key) != value for key, value in expected.items()):
            raise _ExecutionArtifactContractError(
                "prediction artifact provenance does not match execution",
                FailureKind.SCHEMA_MISMATCH,
            )
        checkpoint_expected = {
            "data_manifest_id": dataset_identity.manifest_id,
            "source_commit": request.source_commit,
            "execution_id": request.execution_id,
            "prediction_artifact": predictions_path.name,
            "prediction_sha256": prediction_sha256,
            "dataset_view_sha256": result.dataset_view_sha256,
        }
        if any(checkpoint_payload.get(key) != value for key, value in checkpoint_expected.items()):
            raise _ExecutionArtifactContractError(
                "checkpoint artifact provenance does not match execution",
                FailureKind.SCHEMA_MISMATCH,
            )
        checkpoint_id = checkpoint_payload.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise _ExecutionArtifactContractError(
                "checkpoint artifact has no checkpoint identity", FailureKind.SCHEMA_MISMATCH
            )
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
            dataset_view_sha256=result.dataset_view_sha256,
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
    from tiktok2026.repository.inspector import RepositoryInspector
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
    run_store = _CriterionAwareRepositoryTransitionStore(repo)
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
        policy=ExecutionPolicy(allowed_image_digests=(app_settings.docker_image,)),
        dataset_provider=dataset_provider,
        evaluator=evaluator,
    )
    prompts = {role: _role_prompt(role) for role in AgentRole}
    capabilities = {
        AgentRole.ORCHESTRATION: ("route", "budget", "frontier"),
        AgentRole.RESEARCH: ("repository_read", "dataset_summary", "memory", "literature"),
        AgentRole.IMPLEMENTOR: ("scoped_read", "scoped_write", "diff", "checks"),
        AgentRole.VALIDATOR: (
            "repository_read",
            "diff",
            "checks",
            "provenance",
            "evaluation_read",
        ),
    }
    research_model = app_settings.models.get(AgentRole.RESEARCH)
    online_research = (
        OpenAIWebSearchProvider(research_model, paths.literature)
        if app_settings.online_research.enabled and research_model is not None
        else None
    )
    agents: dict[AgentRole, RoleSpecificAgentClient] = {
        role: RoleSpecificAgentClient(
            OpenAICompatibleClient(model),
            role,
            prompts[role],
            capabilities[role],
            online_research=online_research if role == AgentRole.RESEARCH else None,
            max_online_searches=app_settings.online_research.max_searches,
            max_online_results=app_settings.online_research.max_results_per_search,
            online_allowed_domains=app_settings.online_research.allowed_domains,
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
        repository_inspector=RepositoryInspector(app_settings.repository_root),
        export_service=RepositoryExportService(repo, paths.root),
        bundle_service=RepositoryFinalizationBundleService(repo, paths.root),
        frontier_service=RepositoryFrontierService(
            repo,
            epsilon=app_settings.plateau_epsilon,
            patience=app_settings.plateau_patience,
        ),
        runtime_root=str(paths.root),
        repository_root=str(app_settings.repository_root),
        parent_commit=_current_commit(app_settings.repository_root),
        dataset_root=str(app_settings.dataset_root) if app_settings.dataset_root else None,
        dataset_view_provenance=(
            dataset_provider.provenance if dataset_provider is not None else None
        ),
        evaluator_id=app_settings.evaluator_id,
        docker_image=app_settings.docker_image,
        default_timeout_seconds=app_settings.execution.timeout_seconds,
        default_memory_bytes=app_settings.execution.memory_bytes,
        default_cpus=app_settings.execution.cpus,
        default_gpu_count=app_settings.execution.gpu_count,
        smoke_timeout_seconds=app_settings.execution.smoke_timeout_seconds,
        smoke_memory_bytes=app_settings.execution.smoke_memory_bytes,
        smoke_disk_bytes=app_settings.execution.smoke_disk_bytes,
        max_repairs=app_settings.budget.max_repairs,
        requires_run_baseline=app_settings.profile == "production",
        plateau_epsilon=app_settings.plateau_epsilon,
        plateau_patience=app_settings.plateau_patience,
    )
    controller = ProductionController(ControllerServices(transitions=transitions, store=run_store))
    graph = build_production_graph(controller, checkpointer=SqliteCheckpointer(paths.graph_db))
    return ProductionServices(
        controller,
        repo,
        graph,
        app_settings,
        worktree_manager,
        executor,
        evaluator,
        agents,
        ledger,
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
        self,
        assignment: WorktreeAssignment,
        allowed_scopes: tuple[str, ...],
        previous: SourceRegistration | None = None,
    ) -> SourceRegistration:
        from tiktok2026.repository.diffs import patch_signature

        revision = previous.revision + 1 if previous is not None else 0
        content = f"synthetic source for {assignment.experiment_id} revision {revision}\n".encode()
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
        source_commit = hashlib.sha1((assignment.worktree_id + digest).encode()).hexdigest()
        return SourceRegistration(
            registration_id=f"source-{source_commit}",
            revision=revision,
            experiment_id=assignment.experiment_id,
            run_id=assignment.run_id,
            parent_commit=assignment.parent_commit,
            source_commit=source_commit,
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
        if request.execution_kind == "smoke":
            smoke_row = DatasetViewRow(
                row_id='["synthetic-row","synthetic-user","synthetic-item"]',
                row_identity=("synthetic-row", "synthetic-user", "synthetic-item"),
                user_id="synthetic-user",
                item_id="synthetic-item",
            )
            return ExecutionResult(
                execution_id=request.execution_id,
                experiment_id=request.experiment_id,
                source_registration_id=request.source_registration_id,
                source_commit=request.source_commit,
                command=request.command,
                exit_code=0,
                elapsed_seconds=0.1,
                gpu_hours=0.0,
                artifact_output_bytes=0,
                execution_kind="smoke",
                dataset_manifest_id=self.manifest.manifest_id,
                dataset_manifest_sha256=self.manifest.manifest_sha256,
                dataset_view_sha256=hashlib.sha256(
                    (self.manifest.manifest_sha256 + "smoke").encode()
                ).hexdigest(),
                dataset_valid_rows=(smoke_row,),
                measured_peak_memory_bytes=1 << 20,
                memory_measurement_status="measured",
                resource_measurement_basis="docker_stats",
                gpu_telemetry_status="not_requested",
                smoke_output_valid=True,
                scientific_evidence=False,
            )
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
            source_registration_id=request.source_registration_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=0,
            elapsed_seconds=0.1,
            gpu_hours=0,
            artifact_ids=(prediction.artifact_id, checkpoint.artifact_id),
            checkpoint_id=f"checkpoint-{digest}",
            execution_kind="full",
            dataset_manifest_id=self.manifest.manifest_id,
            dataset_manifest_sha256=self.manifest.manifest_sha256,
            dataset_view_sha256=hashlib.sha256(
                (self.manifest.manifest_sha256 + "full").encode()
            ).hexdigest(),
            measured_peak_memory_bytes=1 << 20,
            memory_measurement_status="measured",
            resource_measurement_basis="docker_stats",
            gpu_telemetry_status="not_requested",
            smoke_output_valid=False,
            scientific_evidence=True,
        )


class _ScriptedAgent:
    def __init__(self) -> None:
        self._proposal_count = 0
        self.scoped_repository: _SyntheticScopedRepository | None = None

    def bind_worktree(
        self,
        path: Path,
        allowed_scopes: tuple[str, ...],
        read_scopes: tuple[str, ...] | None = None,
    ) -> None:
        del path
        self.scoped_repository = _SyntheticScopedRepository(allowed_scopes, read_scopes)

    async def invoke(self, request: ContractModel) -> ContractModel:
        if isinstance(request, OrchestrationRequest):
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
                    implementation_resource_estimate=ImplementationResourceEstimate(
                        predicted_wall_seconds=1.0,
                        predicted_peak_memory_bytes=1 << 20,
                        predicted_artifact_bytes=1 << 20,
                        dataset_passes=1,
                    ),
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
            criterion_assessments = ()
            if request.stage == ValidationStage.IMPLEMENTATION:
                criterion_assessments = tuple(
                    ImplementationCriterionAssessment(
                        criterion_id=criterion,
                        status=CriterionAssessmentStatus.PASS,
                        evidence_refs=("synthetic-implementation-diff",),
                        details="synthetic fixture satisfies the implementation criterion",
                    )
                    for criterion in DEFAULT_IMPLEMENTATION_CRITERIA
                )
            return ValidationReport(
                report_id=f"report-{request.request_id}",
                experiment_id=request.experiment_id,
                stage=request.stage,
                verdict=ValidationVerdict.APPROVED,
                criterion_assessments=criterion_assessments,
                evidence_refs=("synthetic-validator",),
                leakage_risk="none",
            )
        raise ValueError("unsupported synthetic request")


class _SyntheticScopedRepository:
    def __init__(
        self, allowed_scopes: tuple[str, ...], read_scopes: tuple[str, ...] | None = None
    ) -> None:
        self.allowed_scopes = allowed_scopes
        self.write_scopes = allowed_scopes
        self.read_scopes = allowed_scopes if read_scopes is None else read_scopes

    def changed_files(self) -> tuple[str, ...]:
        return (f"{self.allowed_scopes[0]}/train.py",)

    def read(self, relative_path: str, max_characters: int = 20_000) -> str:
        del relative_path, max_characters
        return "def run_training():\n    pass\n"

    def read_base(self, relative_path: str, max_characters: int = 20_000) -> str:
        del relative_path, max_characters
        return "def run_training():\n    pass\n"

    def diff(self) -> str:
        return "synthetic implementation diff\n"


def build_synthetic_controller(
    repository_root: Path, runtime_root: Path, iterations: int = 2
) -> tuple[ProductionController, object, Any]:
    if iterations < 2:
        raise ValueError("synthetic controller requires at least two iterations")
    runtime = initialize_runtime(repository_root, runtime_root)
    repo = _CriterionAwareRepositoryTransitionStore(runtime.repository)
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
    smoke_view_sha256 = hashlib.sha256(
        (manifest.manifest_sha256 + "smoke").encode()
    ).hexdigest()
    smoke_valid_rows = (
        DatasetViewRow(
            row_id='["synthetic-row","synthetic-user","synthetic-item"]',
            row_identity=("synthetic-row", "synthetic-user", "synthetic-item"),
            user_id="synthetic-user",
            item_id="synthetic-item",
        ),
    )

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
        dataset_view_provenance=lambda request: DatasetViewProvenance(
            manifest_id=manifest.manifest_id,
            manifest_sha256=manifest.manifest_sha256,
            view_sha256=smoke_view_sha256,
            valid_rows=smoke_valid_rows,
        ),
        evaluator_id="synthetic-evaluator",
    )
    controller = ProductionController(ControllerServices(transitions=transitions, store=repo))
    graph = build_production_graph(
        controller, checkpointer=SqliteCheckpointer(runtime.paths.graph_db)
    )
    return controller, repo, graph
