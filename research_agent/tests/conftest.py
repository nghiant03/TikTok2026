from __future__ import annotations

import pytest

from research_agent.contracts import (
    OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
    BenchmarkContract,
    EvaluationProtocolStatus,
    MetricName,
    ResearchEvaluationResult,
    ResearchMetricValue,
    ResearchRequest,
    ResearchTaskType,
)
from research_agent.shared_contracts import ExecutionResult, FailureKind, ResourceState


@pytest.fixture
def benchmark() -> BenchmarkContract:
    return BenchmarkContract()


@pytest.fixture
def resource_state() -> ResourceState:
    return ResourceState(
        remaining_gpu_hours=1.0,
        accumulated_gpu_hours=0.0,
        remaining_wall_seconds=3600.0,
        used_tokens=0,
        remaining_tokens=20_000,
        disk_bytes_available=1_000_000_000,
        reserved_final_gpu_hours=0.2,
    )


@pytest.fixture
def proposal_request(benchmark: BenchmarkContract, resource_state: ResourceState) -> ResearchRequest:
    return ResearchRequest(
        request_id="request-1",
        task_type=ResearchTaskType.PROPOSE_EXPERIMENT,
        objective="Propose the next informative ranking experiment.",
        benchmark=benchmark,
        evaluation_protocol_status=EvaluationProtocolStatus.CONFIRMED,
        evaluation_protocol_evidence_refs=(
            OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
        ),
        resource_state=resource_state,
        allowed_implementation_scope=("experiment/features", "experiment/models"),
    )


@pytest.fixture
def evaluation_result() -> ResearchEvaluationResult:
    return ResearchEvaluationResult(
        evaluation_id="evaluation-1",
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        metrics=(
            ResearchMetricValue(name=MetricName.GAUC, value=0.6674),
            ResearchMetricValue(name=MetricName.NDCG_5, value=0.5357),
        ),
        evaluator_artifact_id="evaluator-1",
        evaluator_sha256="a" * 64,
        prediction_sha256="b" * 64,
        validity="provisional",
    )


@pytest.fixture
def interpretation_request(
    benchmark: BenchmarkContract,
    resource_state: ResourceState,
    evaluation_result: ResearchEvaluationResult,
) -> ResearchRequest:
    return ResearchRequest(
        request_id="request-interpret-1",
        task_type=ResearchTaskType.INTERPRET_RESULT,
        objective="Interpret the latest ranking result.",
        benchmark=benchmark,
        current_experiment_id="experiment-1",
        evaluation_result=evaluation_result,
        resource_state=resource_state,
        allowed_implementation_scope=("experiment/features", "experiment/models"),
    )


@pytest.fixture
def failed_execution_result() -> ExecutionResult:
    return ExecutionResult(
        execution_id="execution-failed-1",
        experiment_id="experiment-1",
        source_commit="commit-1",
        command=("python", "train.py"),
        exit_code=1,
        elapsed_seconds=12.5,
        gpu_hours=0.0,
        failure_kind=FailureKind.SYNTAX_IMPORT,
    )


@pytest.fixture
def failed_execution_request(
    benchmark: BenchmarkContract,
    resource_state: ResourceState,
    failed_execution_result: ExecutionResult,
) -> ResearchRequest:
    return ResearchRequest(
        request_id="request-failed-execution-1",
        task_type=ResearchTaskType.INTERPRET_RESULT,
        objective="Interpret the failed execution.",
        benchmark=benchmark,
        current_experiment_id="experiment-1",
        execution_result=failed_execution_result,
        resource_state=resource_state,
        allowed_implementation_scope=("experiment/features", "experiment/models"),
    )
