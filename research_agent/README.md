# Research Agent

This directory contains a standalone Research Agent implementation for the TikTok2026
three-agent system. It reuses the bundled read-only contract snapshot from
`external/TikTok2026-main/src/tiktok2026/contracts/models.py` without modifying it.

## Purpose

The Research Agent forms evidence-backed hypotheses and structured experiment specifications
from authorized repository observations, safe data summaries, research memory, evaluation
results, resource limits, and literature records.

It returns exactly one structured result:

1. an `ExperimentProposal` containing an explicit `Hypothesis` and `ExperimentSpec`;
2. an `EvidenceRequest`; or
3. a `ResearchInterpretation` for an execution or evaluation result.

The Research Agent does not modify experiment source code, execute training, invoke the
protected evaluator, access organizer hidden-test labels, or approve its own proposal.

## Implemented architecture

### Contract layer

- Reuses shared `ExperimentSpec`, `ExecutionResult`, and `ResourceState` contracts.
- Provides Research-specific `ResearchRequest`, `ResearchContext`, `ResearchDecision`,
  `Hypothesis`, `ExperimentProposal`, `EvidenceRequest`, and `ResearchInterpretation` models.
- Keeps `ResearchResponse` as a backward-compatible alias for `ResearchDecision`.
- Uses strict, immutable Pydantic models that reject unknown fields.
- Fixes the currently confirmed local protocol to KuaiRand-Pure, positive label `long_view`,
  metrics `GAUC` and `nDCG@5`, and their mean as the primary score.
- Distinguishes the local public holdout from the organizer-owned hidden test.
- Rejects external training data, public-holdout development access, hidden-test access, and
  all pretrained weights.

### Read-only context layer

- Separates repository, data, research-memory, and literature access behind capability
  interfaces.
- Collects all four capability types concurrently into a bounded `ResearchContext`.
- Requires every `EvidenceItem` to have a unique ID and a traceable source.
- Rejects unauthorized evidence and evidence containing test labels.
- Queries related experiments, follows `parent_experiment_id` links to build lineage, and
  retrieves evidence-backed `ResearchLesson` records.

### Decision and policy layer

- Validates model output with Pydantic and deterministic business rules.
- Validates `ResearchRequest` before any context reader is called.
- Verifies evidence references, source provenance, parent experiment, allowed implementation
  paths, numeric claims, and historical duplicate detection.
- Separates execution failures, evaluation facts, and research interpretation.
- Allows one bounded model-repair attempt, then returns a typed `ResearchAgentFailure`.
- Requires a safe train/validation-only wrapper for official FM baseline reproduction.
- Requires seeds 0 through 4, tolerance `0.002`, zero predicted GPU hours, the pinned official
  FM configuration, and explicit leakage risks.
- Forbids direct calls to Starter Kit entry points that also load the local public holdout.

When `evaluation_protocol_status=unconfirmed`, the Research Agent may only return an
`EvidenceRequest` for the formal split and evaluator. When `baseline_status=missing`, it may
only request evidence or propose `baseline_reproduction`; it cannot propose optimization first.

### LangGraph layer

The Research subgraph uses the following control flow:

```text
validate request -> build context -> call model -> validate response -> success
                                                       |
                                                       +-> repair once
                                                           -> validate again
                                                           -> success or typed failure
```

`run_research_with_repair` provides the same bounded-repair behavior without graph scheduling
for unit tests and gradual system integration.

### Phase 2 runtime capabilities

- Real DeepSeek Chat Completions client with token accounting.
- Read-only evidence from the bundled `external/TikTok2026-main` snapshot.
- Safe training-only summaries from the authorized KuaiRand-Pure copy.
- Local JSONL experiment history, lineage, and research lessons.
- Local PDF extraction and public literature retrieval from Semantic Scholar, arXiv, Crossref,
  and explicitly configured web pages.
- SHA256 verification of protected Starter Kit artifacts.

## Project layout

```text
research_agent/
|-- pyproject.toml
|-- README.md
|-- PHASE2.md
|-- EVALUATION_PROTOCOL_RISK.md
|-- external/
|   |-- TikTok2026-main/      # Read-only repository and shared-contract snapshot
|   |-- kuairand-starter-kit/ # Read-only Starter Kit snapshot
|   |-- KuaiRand-Pure/        # Authorized local dataset copy
|   `-- literature/           # Authorized local PDF collection
|-- history/
|   `-- README.md
|-- src/research_agent/
|   |-- contracts.py          # Research input, output, and failure contracts
|   |-- shared_contracts.py   # Read-only bridge to authoritative shared contracts
|   |-- capabilities.py       # Read-only capability interfaces
|   |-- adapters.py           # Repository, data, memory, PDF, and online readers
|   |-- context.py            # Bounded context construction
|   |-- prompt.py             # Runtime prompt construction
|   |-- prompt.md             # Human-readable role contract
|   |-- model.py              # Scripted and DeepSeek model clients
|   |-- policy.py             # Deterministic research-policy validation
|   |-- agent.py              # Non-graph execution and one repair attempt
|   |-- graph.py              # LangGraph Research subgraph
|   |-- protocol.py           # Starter Kit artifact verification
|   |-- runtime.py            # Phase 2 settings and dependency assembly
|   |-- phase2_smoke.py       # Real-environment smoke-test entry point
|   `-- testing.py            # Deterministic test doubles
`-- tests/                    # Contract, context, agent, graph, adapter, and protocol tests
```

## Local verification

Run from PowerShell:

```powershell
cd research_agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -v
```

Run a real local-context smoke test without paid model calls:

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke --offline-literature
```

Enable online literature retrieval without calling the model:

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke
```

Call the real DeepSeek model:

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke --call-model --offline-literature
```

The final command uses paid API tokens. The program does not print or persist the API key.

## Shared-contract loading

`shared_contracts.py` loads
`external/TikTok2026-main/src/tiktok2026/contracts/models.py` in read-only mode. Set
`TIKTOK2026_CONTRACTS_FILE` only when integration requires an explicitly selected replacement.

## Integration boundary

The Orchestration Agent constructs `ResearchRequest` and calls `run_research_graph`. The
integrated system must provide:

- a `ResearchModelClient` implementation;
- repository, data, memory, and literature capabilities; and
- an importable authoritative shared-contract module.

The deterministic Validator remains outside the three language-model agents. Implement,
execution, and protected evaluation components are responsible for writing code, training, and
producing `ExecutionResult` and `EvaluationResult` records.

See `EVALUATION_PROTOCOL_RISK.md` for the unresolved final-evaluation protocol risk.
