from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from research_agent.context import build_research_context
from research_agent.contracts import (
    OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
    BenchmarkContract,
    EvaluationProtocolStatus,
    ResearchRequest,
    ResearchTaskType,
)
from research_agent.graph import run_research_graph
from research_agent.protocol import verify_official_starter_kit
from research_agent.runtime import (
    Phase2Settings,
    build_phase2_capabilities,
    build_phase2_model_client,
)
from research_agent.shared_contracts import ResourceState


def _request(objective: str, starter_kit_root: Path) -> ResearchRequest:
    benchmark = BenchmarkContract()
    verify_official_starter_kit(starter_kit_root, benchmark)
    return ResearchRequest(
        request_id="phase2-live-smoke",
        task_type=ResearchTaskType.PROPOSE_EXPERIMENT,
        objective=objective,
        benchmark=benchmark,
        evaluation_protocol_status=EvaluationProtocolStatus.CONFIRMED,
        evaluation_protocol_evidence_refs=(
            OFFICIAL_EVALUATION_PROTOCOL_EVIDENCE_ID,
        ),
        resource_state=ResourceState(
            remaining_gpu_hours=1.0,
            accumulated_gpu_hours=0.0,
            remaining_wall_seconds=3600.0,
            used_tokens=0,
            remaining_tokens=1_000_000,
            disk_bytes_available=1_000_000_000,
            reserved_final_gpu_hours=0.2,
        ),
        allowed_implementation_scope=("experiment/features", "experiment/models"),
    )


async def _run(args: argparse.Namespace) -> int:
    settings = Phase2Settings.from_env()
    if args.offline_literature:
        settings = replace(settings, online_literature_enabled=False)
    capabilities = build_phase2_capabilities(settings)
    request = _request(args.objective, settings.starter_kit_root)

    if not args.call_model:
        context = await build_research_context(request, capabilities)
        payload = {
            "request_id": request.request_id,
            "evidence_count": len(context.evidence),
            "history_count": len(context.experiment_history),
            "lineage_count": len(context.experiment_lineage),
            "lesson_count": len(context.retrieved_lessons),
            "experiment_lineage": [
                item.model_dump(mode="json") for item in context.experiment_lineage
            ],
            "retrieved_lessons": [
                item.model_dump(mode="json") for item in context.retrieved_lessons
            ],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "kind": item.kind.value,
                    "source_ref": item.source_ref,
                    "summary": item.summary,
                }
                for item in context.evidence
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    model_client = build_phase2_model_client(settings)
    state = await run_research_graph(request, model_client, capabilities)
    if "response" in state:
        print(state["response"].model_dump_json(indent=2))
        if model_client.usage:
            print(
                json.dumps(
                    {"model_usage": [record.__dict__ for record in model_client.usage]},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return 0
    failure = state.get("failure")
    if failure is None:
        raise RuntimeError("research graph ended without a response or typed failure")
    print(failure.model_dump_json(indent=2))
    return 1


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Phase 2 read-only Research Agent smoke test")
    parser.add_argument(
        "--objective",
        default="Propose the next evidence-backed KuaiRand-Pure ranking experiment.",
    )
    parser.add_argument(
        "--offline-literature",
        action="store_true",
        help="Use local PDFs only and skip Crossref retrieval.",
    )
    parser.add_argument(
        "--call-model",
        action="store_true",
        help="Perform a paid DeepSeek call; omitted by default.",
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
