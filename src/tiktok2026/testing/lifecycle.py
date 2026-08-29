from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from tiktok2026.bootstrap import initialize_runtime
from tiktok2026.contracts import (
    ArtifactRetention,
    AuditEvent,
    EvaluationResult,
    EvaluatorIdentity,
    ExperimentSpec,
    Fidelity,
    FinalizationRecord,
    ProvisionalFinalizationRequest,
    RunRecord,
    RuntimePaths,
)
from tiktok2026.observability.exports import export_records
from tiktok2026.persistence.artifacts import ArtifactStore
from tiktok2026.repository.worktrees import GitWorktreeManager
from tiktok2026.testing.synthetic import evaluate_fixture, fixture_rows, score_rows


@dataclass(frozen=True)
class SyntheticExports:
    jsonl: Path
    markdown: Path


@dataclass(frozen=True)
class SyntheticLifecycleResult:
    run_id: str
    experiment_ids: tuple[str, ...]
    scores: tuple[float, ...]
    terminal_reason: str
    finalization: FinalizationRecord
    exports: SyntheticExports
    paths: RuntimePaths


def _spec(iteration: int, parent_experiment_id: str | None) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=f"synthetic-{iteration}",
        hypothesis_id=f"hypothesis-{iteration}",
        hypothesis="Increasing signal scale preserves correct within-user ordering.",
        mechanism="Apply a deterministic scale to the synthetic ranking feature.",
        motivation="Exercise proposal, execution, evaluation, and persistence boundaries cheaply.",
        parent_experiment_id=parent_experiment_id,
        expected_signal="NDCG@10 and Recall@50 remain valid and deterministic.",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Both metrics are finite and at least the previous values.",
        failure_criteria="Execution or schema validation fails.",
        source_provenance=("synthetic-fixture-v1",),
    )


async def run_synthetic_lifecycle(
    iterations: int = 2, runtime_root: Path | None = None
) -> SyntheticLifecycleResult:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    repository_root = Path(__file__).resolve().parents[3]
    selected_runtime = (
        runtime_root or repository_root.parent / f"{repository_root.name}.synthetic-runtime"
    )
    services = initialize_runtime(repository_root, selected_runtime)
    run_id = f"synthetic-run-{uuid.uuid4().hex}"
    repository = services.repository
    repository.put_run(
        RunRecord(run_id=run_id, status="running"),
        transition_id=f"{run_id}-running",
        expected_predecessor=None,
    )
    evaluator_id = "synthetic-evaluator-v1"
    repository.put_evaluator_identity(
        EvaluatorIdentity(
            evaluator_id=evaluator_id,
            evaluator_sha256=hashlib.sha256(evaluator_id.encode()).hexdigest(),
            validity="provisional",
        )
    )
    artifact_store = ArtifactStore(services.paths, repository)
    experiment_ids: list[str] = []
    specifications: list[ExperimentSpec] = []
    scores: list[float] = []
    evaluations: list[EvaluationResult] = []
    for iteration in range(1, iterations + 1):
        spec = _spec(iteration, experiment_ids[-1] if experiment_ids else None)
        predecessor = None
        proposal_transition = f"{run_id}-{spec.experiment_id}-proposed"
        repository.put_experiment(
            spec,
            status="proposed",
            run_id=run_id,
            transition_id=proposal_transition,
            expected_predecessor=predecessor,
        )
        repository.put_audit_event(
            AuditEvent(
                event_id=f"{run_id}-proposal-{iteration}",
                run_id=run_id,
                experiment_id=spec.experiment_id,
                event_type="experiment_proposed",
                actor_type="controller",
                actor_id="synthetic",
                payload={"fidelity": spec.fidelity.value},
            )
        )
        rows = fixture_rows()
        evaluation = evaluate_fixture(spec.experiment_id, rows, score_rows(rows, float(iteration)))
        repository.put_json("evaluation", evaluation.evaluation_id, evaluation.model_dump_json())
        evaluation_transition = f"{run_id}-{spec.experiment_id}-evaluated"
        repository.put_experiment(
            spec,
            status="evaluated",
            run_id=run_id,
            transition_id=evaluation_transition,
            expected_predecessor=proposal_transition,
        )
        repository.put_audit_event(
            AuditEvent(
                event_id=f"{run_id}-evaluation-{iteration}",
                run_id=run_id,
                experiment_id=spec.experiment_id,
                event_type="evaluation_persisted",
                actor_type="controller",
                actor_id="synthetic",
                payload={
                    "evaluation_id": evaluation.evaluation_id,
                    "validity": evaluation.validity,
                    "validation_score": evaluation.validation_score,
                },
            )
        )
        experiment_ids.append(spec.experiment_id)
        specifications.append(spec)
        scores.append(evaluation.validation_score)
        evaluations.append(evaluation)
    champion = evaluations[-1]
    champion_spec = specifications[-1]
    parent_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    worktree_manager = GitWorktreeManager(
        repository_root,
        services.paths.root,
        approved_parent_validator=lambda candidate: candidate == parent_commit,
        artifact_registry=repository,
    )
    assignment = worktree_manager.create(run_id, champion_spec, parent_commit)
    try:
        target = assignment.path / "src/tiktok2026/experiment/synthetic_provenance.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
        source = worktree_manager.register_source(
            assignment, ("src/tiktok2026/experiment",)
        )
        repository.put_source_registration(source)
    finally:
        worktree_manager.remove(assignment)
    repository.put_experiment(
        champion_spec,
        status="converged",
        run_id=run_id,
        transition_id=f"{run_id}-{champion_spec.experiment_id}-converged",
        expected_predecessor=f"{run_id}-{champion_spec.experiment_id}-evaluated",
    )
    repository.put_run(
        RunRecord(run_id=run_id, status="converged"),
        transition_id=f"{run_id}-converged",
        expected_predecessor=f"{run_id}-running",
    )
    bundle = artifact_store.publish_bytes(
        run_id=run_id,
        experiment_id=champion.experiment_id,
        kind="finalization_bundle",
        filename="bundle.json",
        content=json.dumps(champion.model_dump(mode="json"), sort_keys=True).encode(),
        producer="controller",
        retention=ArtifactRetention.PROVENANCE,
    )
    finalization = repository.persist_provisional_finalization(
        ProvisionalFinalizationRequest(
            finalization_id=f"finalization-{run_id}",
            run_id=run_id,
            experiment_id=champion.experiment_id,
            source_commit=source.source_commit,
            checkpoint_id=champion.checkpoint_id,
            evaluation_id=champion.evaluation_id,
            bundle_artifact_id=bundle.artifact_id,
            evaluator_id=evaluator_id,
        )
    )
    events = repository.list_audit_events(run_id)
    jsonl, markdown = export_records(
        run_id,
        tuple(event.model_dump(mode="json") for event in events),
        services.paths.exports / run_id,
    )
    return SyntheticLifecycleResult(
        run_id=run_id,
        experiment_ids=tuple(experiment_ids),
        scores=tuple(scores),
        terminal_reason="synthetic_iteration_limit",
        finalization=finalization,
        exports=SyntheticExports(jsonl=jsonl, markdown=markdown),
        paths=services.paths,
    )
