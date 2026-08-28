from __future__ import annotations

import asyncio

from research_agent.capabilities import ResearchCapabilities
from research_agent.contracts import (
    OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
    EvaluationProtocolStatus,
    EvidenceItem,
    EvidenceKind,
    ResearchContext,
    ResearchRequest,
)

DEFAULT_MAX_EVIDENCE_ITEMS = 24


async def build_research_context(
    request: ResearchRequest,
    capabilities: ResearchCapabilities,
    *,
    max_evidence_items: int = DEFAULT_MAX_EVIDENCE_ITEMS,
) -> ResearchContext:
    """Collect bounded evidence without exposing privileged implementations."""

    repository_evidence, data_evidence, memory, literature_evidence = await asyncio.gather(
        capabilities.repository.read_repository_evidence(request),
        capabilities.data.read_data_evidence(request),
        capabilities.memory.query_research_memory(request),
        capabilities.literature.read_literature_evidence(request),
    )

    benchmark = request.benchmark
    evidence: list[EvidenceItem] = [
        EvidenceItem(
            evidence_id=f"benchmark:{benchmark.benchmark_id}:v{benchmark.schema_version}",
            kind=EvidenceKind.BENCHMARK,
            summary=(
                f"Dataset={benchmark.dataset}; positive_label={benchmark.positive_label}; "
                f"metrics={','.join(metric.value for metric in benchmark.metrics)}; "
                f"validation_ranking={benchmark.validation_ranking}; "
                "development=train+validation; organizer hidden test is not locally available; "
                "authority=user-confirmed resolution of the latest Problem Statement plus the "
                "official Starter Kit and typed BenchmarkContract; stale repository manifests "
                "cannot override this "
                f"contract; baseline_status={request.baseline_status.value}; "
                "evaluation_protocol_status="
                f"{request.evaluation_protocol_status.value}."
            ),
            source_ref="problem-statement-latest+official-starter-kit",
        )
    ]

    if request.evaluation_protocol_status is not EvaluationProtocolStatus.UNCONFIRMED:
        evidence.append(
            EvidenceItem(
                evidence_id=OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
                kind=EvidenceKind.VALIDATION,
                summary=(
                    "Official KuaiRand-Pure protocol: label=long_view; task=within-user "
                    "ranking over logged impressions; train=20220408..20220421; "
                    "validation=20220422..20220428; public_holdout=20220429..20220508; "
                    "the organizer hidden test is not locally available; the public holdout is "
                    "not for iterative development; "
                    "metrics=GAUC,nDCG@5; primary=mean(GAUC,nDCG@5); evaluator rules include "
                    "tie-corrected Mann-Whitney AUC, positive-count-weighted GAUC, and zero "
                    "nDCG for all-negative users; official validation FM baseline "
                    "GAUC=0.6674,nDCG@5=0.5357,primary=0.6016; "
                    "official FM config: features=user_id,video_id,author_id,tab,dur_bucket; "
                    "k=16; lr=0.001; l2=0.000001; batch_size=8192; max_epochs=40; patience=4; "
                    "symbols=baseline.py::FM,evaluate.py::evaluate; forbidden direct calls="
                    "data.py::load,baseline.py::run_fm; "
                    "FM reproduction uses seeds=0..4 and CPU only; "
                    "reproduction/convergence tolerance=0.002; "
                    f"evaluate.py sha256={benchmark.evaluator_sha256}; "
                    f"data.py sha256={benchmark.data_loader_sha256}; "
                    f"baseline_scores.json sha256={benchmark.baseline_scores_sha256}."
                ),
                source_ref=(
                    "starter-kit://data.py+evaluate.py+baseline_scores.json"
                ),
            )
        )

    if request.execution_result is not None:
        execution = request.execution_result
        failure = execution.failure_kind.value if execution.failure_kind is not None else "none"
        evidence.append(
            EvidenceItem(
                evidence_id=f"execution:{execution.execution_id}",
                kind=EvidenceKind.EXECUTION,
                summary=(
                    f"Experiment {execution.experiment_id}: exit_code={execution.exit_code}; "
                    f"elapsed_seconds={execution.elapsed_seconds:.3f}; "
                    f"gpu_hours={execution.gpu_hours:.6f}; failure_kind={failure}."
                ),
                source_ref=execution.source_commit,
            )
        )

    if request.evaluation_result is not None:
        metric_summary = ", ".join(
            f"{metric.name}={metric.value:.6f}"
            for metric in request.evaluation_result.metrics
        )
        evidence.append(
            EvidenceItem(
                evidence_id=f"evaluation:{request.evaluation_result.evaluation_id}",
                kind=EvidenceKind.EVALUATION,
                summary=(
                    f"Experiment {request.evaluation_result.experiment_id}: {metric_summary}; "
                    f"validity={request.evaluation_result.validity}."
                ),
                source_ref=request.evaluation_result.evaluator_artifact_id,
            )
        )

    # Request-specific execution/evaluation facts take priority over optional readers
    # when the bounded context must be truncated.
    evidence.extend(repository_evidence)
    evidence.extend(data_evidence)
    evidence.extend(literature_evidence)

    bounded_evidence = tuple(evidence[:max_evidence_items])
    return ResearchContext(
        request=request,
        evidence=bounded_evidence,
        experiment_history=memory.related_experiments,
        experiment_lineage=memory.experiment_lineage,
        retrieved_lessons=memory.retrieved_lessons,
        max_evidence_items=max_evidence_items,
    )
