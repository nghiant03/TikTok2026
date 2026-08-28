from __future__ import annotations

from research_agent.contracts import (
    OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
    BaselineReproductionControl,
    EvidenceItem,
    EvidenceKind,
    ExperimentProposal,
    OfficialFMConfig,
    ProposalPurpose,
    ResearchContext,
    ResearchDecisionKind,
    ResearchRequest,
    ResearchResponse,
)
from research_agent.shared_contracts import ExperimentSpec, Fidelity
from research_agent.testing import fake_capabilities


def make_context(request: ResearchRequest) -> ResearchContext:
    return ResearchContext(
        request=request,
        evidence=(
            EvidenceItem(
                evidence_id="benchmark:kuairand-pure:v1",
                kind=EvidenceKind.BENCHMARK,
                summary="KuaiRand-Pure uses long_view and GAUC/nDCG@5.",
                source_ref="problem-statement-latest+official-starter-kit",
            ),
            EvidenceItem(
                evidence_id=OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
                kind=EvidenceKind.VALIDATION,
                summary=(
                    "Official fixed date split and Starter Kit evaluator; validation FM "
                    "GAUC=0.6674,nDCG@5=0.5357,primary=0.6016."
                ),
                source_ref="starter-kit://data.py+evaluate.py+baseline_scores.json",
            ),
            EvidenceItem(
                evidence_id="repo-1",
                kind=EvidenceKind.REPOSITORY,
                summary="The current editable pipeline does not use recent clicks.",
                source_ref="repository-summary-v1",
            ),
            EvidenceItem(
                evidence_id="data-1",
                kind=EvidenceKind.DATA,
                summary="Training users have timestamped click histories.",
                source_ref="data-summary-v1",
            ),
        ),
    )


def make_capabilities():
    return fake_capabilities(
        repository_evidence=(
            EvidenceItem(
                evidence_id="repo-1",
                kind=EvidenceKind.REPOSITORY,
                summary="The current editable pipeline does not use recent clicks.",
                source_ref="repository-summary-v1",
            ),
        ),
        data_evidence=(
            EvidenceItem(
                evidence_id="data-1",
                kind=EvidenceKind.DATA,
                summary="Training users have timestamped click histories.",
                source_ref="data-summary-v1",
            ),
        ),
    )


def make_proposal_response(
    request: ResearchRequest,
    *,
    response_id: str = "response-1",
    experiment_id: str = "experiment-1",
    evidence_refs: tuple[str, ...] = (
        "repo-1",
        "data-1",
        OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
    ),
    implementation_scope: tuple[str, ...] = ("experiment/models/baseline.py",),
    purpose: ProposalPurpose = ProposalPurpose.BASELINE_REPRODUCTION,
    source_provenance: tuple[str, ...] = (
        "repository-summary-v1",
        "data-summary-v1",
        "starter-kit://data.py+evaluate.py+baseline_scores.json",
    ),
    expected_signal: str = "Produce deterministic official-protocol GAUC and nDCG@5 metrics.",
    success_criteria: str = "The compliant baseline runs and records both official metrics.",
    failure_criteria: str = "The run, prediction validation, or metric recording fails.",
    predicted_gpu_hours: float = 0.0,
    leakage_risks: tuple[str, ...] = (
        "Keep long_view out of input features and isolate the public holdout.",
    ),
) -> ResearchResponse:
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id=f"hypothesis-{experiment_id}",
        hypothesis="A compliant long_view baseline can be reproduced deterministically.",
        mechanism="Fit a leakage-safe baseline using only pre-impression fields.",
        motivation="No compliant baseline result is present in the authorized context.",
        evidence_refs=evidence_refs,
        parent_experiment_id=request.parent_experiment_id,
        expected_signal=expected_signal,
        implementation_scope=implementation_scope,
        fidelity=Fidelity.SMOKE,
        predicted_gpu_hours=predicted_gpu_hours,
        success_criteria=success_criteria,
        failure_criteria=failure_criteria,
        leakage_risks=leakage_risks,
        source_provenance=source_provenance,
    )
    return ResearchResponse(
        response_id=response_id,
        request_id=request.request_id,
        kind=ResearchDecisionKind.EXPERIMENT_PROPOSAL,
        experiment_proposal=ExperimentProposal(
            spec=spec,
            purpose=purpose,
            uses_target_derived_features=False,
            baseline_reproduction_control=(
                BaselineReproductionControl(official_fm_config=OfficialFMConfig())
                if purpose is ProposalPurpose.BASELINE_REPRODUCTION
                else None
            ),
        ),
    )
