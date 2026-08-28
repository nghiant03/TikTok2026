from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import PurePosixPath

from research_agent.contracts import (
    BaselineStatus,
    EvaluationProtocolStatus,
    EvidenceKind,
    EvidenceRequestCategory,
    ExperimentProposal,
    ProposalPurpose,
    ResearchContext,
    ResearchDecisionKind,
    ResearchResponse,
    ResearchTaskType,
)
from research_agent.shared_contracts import ExperimentSpec


class ResearchPolicyError(ValueError):
    pass


_UNSUPPORTED_NUMERIC_FORECAST = re.compile(
    r"(?<![\w@])(?:[+-]?\d+\.\d+|\d+(?:\.\d+)?%)(?!\w)"
)


def experiment_signature(spec: ExperimentSpec) -> str:
    normalized = {
        "hypothesis": " ".join(spec.hypothesis.lower().split()),
        "mechanism": " ".join(spec.mechanism.lower().split()),
        "scope": sorted(_normalize_path(path) for path in spec.implementation_scope),
        "parent": spec.parent_experiment_id,
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_research_response(
    response: ResearchResponse,
    context: ResearchContext,
) -> ResearchResponse:
    request = context.request
    if response.request_id != request.request_id:
        raise ResearchPolicyError("response request_id does not match the active request")

    allowed_kinds = {
        ResearchTaskType.PROPOSE_EXPERIMENT: {
            ResearchDecisionKind.EXPERIMENT_PROPOSAL,
            ResearchDecisionKind.EVIDENCE_REQUEST,
        },
        ResearchTaskType.INTERPRET_RESULT: {
            ResearchDecisionKind.RESULT_INTERPRETATION,
            ResearchDecisionKind.EVIDENCE_REQUEST,
        },
    }[request.task_type]
    if response.kind not in allowed_kinds:
        raise ResearchPolicyError(
            f"{request.task_type} cannot return {response.kind}"
        )

    if (
        request.task_type is ResearchTaskType.PROPOSE_EXPERIMENT
        and request.evaluation_protocol_status is EvaluationProtocolStatus.UNCONFIRMED
    ):
        if response.kind is not ResearchDecisionKind.EVIDENCE_REQUEST:
            raise ResearchPolicyError(
                "evaluation protocol is unconfirmed; return an EvidenceRequest for the "
                "formal data split and GAUC/nDCG@5 evaluator before proposing an experiment"
            )
        evidence_request = response.evidence_request
        if evidence_request is None:
            raise ResearchPolicyError(
                "unconfirmed evaluation protocol requires an EvidenceRequest"
            )
        required_categories = {
            EvidenceRequestCategory.DATA_SPLIT,
            EvidenceRequestCategory.EVALUATION_PROTOCOL,
        }
        missing_categories = sorted(
            category.value
            for category in required_categories - set(evidence_request.categories)
        )
        if missing_categories:
            raise ResearchPolicyError(
                "evaluation-protocol EvidenceRequest is missing required categories: "
                f"{missing_categories}"
            )

    if response.experiment_proposal is not None:
        _validate_proposal(response.experiment_proposal, context)
    if response.result_interpretation is not None:
        interpretation = response.result_interpretation
        if interpretation.experiment_id != request.current_experiment_id:
            raise ResearchPolicyError("interpretation targets the wrong experiment")
        _validate_evidence_refs(interpretation.evidence_refs, context)
        if request.execution_result is not None:
            execution = request.execution_result
            required_ref = f"execution:{execution.execution_id}"
            if required_ref not in interpretation.evidence_refs:
                raise ResearchPolicyError("interpretation must cite its execution result")
            if interpretation.execution_failure_kind != execution.failure_kind:
                raise ResearchPolicyError("interpretation failure kind does not match execution")
        elif interpretation.execution_failure_kind is not None:
            raise ResearchPolicyError("interpretation claims an unavailable execution failure")
        if request.evaluation_result is not None:
            required_ref = f"evaluation:{request.evaluation_result.evaluation_id}"
            if required_ref not in interpretation.evidence_refs:
                raise ResearchPolicyError("interpretation must cite its evaluation result")

    return response


def _validate_proposal(proposal: ExperimentProposal, context: ResearchContext) -> None:
    spec = proposal.spec
    if not spec.leakage_risks:
        raise ResearchPolicyError("proposal leakage_risks must not be empty")
    if not spec.evidence_refs:
        raise ResearchPolicyError("proposal must cite at least one evidence item")
    _validate_evidence_refs(spec.evidence_refs, context)
    _validate_source_provenance(spec.evidence_refs, spec.source_provenance, context)
    missing_protocol_refs = sorted(
        set(context.request.evaluation_protocol_evidence_refs) - set(spec.evidence_refs)
    )
    if missing_protocol_refs:
        raise ResearchPolicyError(
            f"proposal must cite evaluation protocol evidence: {missing_protocol_refs}"
        )
    if (
        context.request.baseline_status is BaselineStatus.MISSING
        and proposal.purpose is not ProposalPurpose.BASELINE_REPRODUCTION
    ):
        raise ResearchPolicyError(
            "optimization is forbidden until a compliant baseline has been reproduced"
        )
    if proposal.purpose is ProposalPurpose.BASELINE_REPRODUCTION:
        if spec.predicted_gpu_hours != 0.0:
            raise ResearchPolicyError(
                "official FM baseline reproduction is CPU-only and must predict zero GPU hours"
            )
        if not any(
            _normalize_path(path) == "experiment/models"
            or _normalize_path(path).startswith("experiment/models/")
            for path in spec.implementation_scope
        ):
            raise ResearchPolicyError(
                "baseline reproduction must implement its safe wrapper under experiment/models"
            )
    if proposal.purpose is ProposalPurpose.OPTIMIZATION:
        missing_baseline_refs = sorted(
            set(context.request.baseline_evidence_refs) - set(spec.evidence_refs)
        )
        if missing_baseline_refs:
            raise ResearchPolicyError(
                f"optimization must cite baseline evidence: {missing_baseline_refs}"
            )
    if spec.parent_experiment_id != context.request.parent_experiment_id:
        raise ResearchPolicyError("proposal parent does not match the requested parent")
    for path in spec.implementation_scope:
        if not _path_is_allowed(path, context.request.allowed_implementation_scope):
            raise ResearchPolicyError(f"implementation path is outside allowed scope: {path}")
    signature = experiment_signature(spec)
    if signature in {item.normalized_signature for item in context.experiment_history}:
        raise ResearchPolicyError("proposal exactly duplicates a historical experiment")
    _validate_numeric_criteria(spec, context)


def _validate_evidence_refs(refs: tuple[str, ...], context: ResearchContext) -> None:
    missing = sorted(set(refs) - context.evidence_ids)
    if missing:
        raise ResearchPolicyError(f"unknown evidence references: {missing}")


def _validate_source_provenance(
    evidence_refs: tuple[str, ...],
    source_provenance: tuple[str, ...],
    context: ResearchContext,
) -> None:
    if not source_provenance:
        raise ResearchPolicyError("proposal source_provenance must not be empty")
    source_by_evidence_id = {item.evidence_id: item.source_ref for item in context.evidence}
    required_sources = {source_by_evidence_id[ref] for ref in evidence_refs}
    missing_sources = sorted(required_sources - set(source_provenance))
    if missing_sources:
        raise ResearchPolicyError(
            f"source_provenance is missing cited evidence sources: {missing_sources}"
        )


def _validate_numeric_criteria(spec: ExperimentSpec, context: ResearchContext) -> None:
    criteria = f"{spec.expected_signal} {spec.success_criteria} {spec.failure_criteria}"
    claimed_numbers = _normalized_numeric_tokens(criteria)
    if not claimed_numbers:
        return
    cited_score_evidence = [
        item
        for item in context.evidence
        if item.evidence_id in spec.evidence_refs
        and item.kind in {EvidenceKind.EVALUATION, EvidenceKind.VALIDATION}
    ]
    if not cited_score_evidence:
        raise ResearchPolicyError(
            "numeric score forecasts or thresholds require cited evaluation evidence"
        )
    supported_numbers = set().union(
        *(_normalized_numeric_tokens(item.summary) for item in cited_score_evidence)
    )
    unsupported = sorted(claimed_numbers - supported_numbers)
    if unsupported:
        raise ResearchPolicyError(
            "numeric score forecasts or thresholds are absent from cited evaluation "
            f"evidence: {[value for value, _ in unsupported]}"
        )


def _normalized_numeric_tokens(text: str) -> set[tuple[str, bool]]:
    normalized: set[tuple[str, bool]] = set()
    for match in _UNSUPPORTED_NUMERIC_FORECAST.finditer(text):
        token = match.group(0)
        is_percent = token.endswith("%")
        raw = token.removesuffix("%").lstrip("+")
        normalized.add((str(Decimal(raw).normalize()), is_percent))
    return normalized


def _path_is_allowed(path: str, allowed_scopes: tuple[str, ...]) -> bool:
    normalized = _normalize_path(path)
    return any(
        normalized == scope or normalized.startswith(scope + "/")
        for scope in (_normalize_path(item) for item in allowed_scopes)
    )


def _normalize_path(path: str) -> str:
    raw = path.replace("\\", "/")
    parts = PurePosixPath(raw).parts
    if not parts or raw.startswith("/") or any(part == ".." for part in parts):
        raise ResearchPolicyError(f"unsafe implementation path: {path}")
    candidate = "/".join(part for part in parts if part not in {"", "."})
    if not candidate:
        raise ResearchPolicyError(f"unsafe implementation path: {path}")
    return candidate.rstrip("/")
