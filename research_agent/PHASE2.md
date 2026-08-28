# Research Agent Phase 2 Runtime Guide

Phase 2 extends the Phase 1 contracts and LangGraph control flow with real but controlled
runtime capabilities:

- a DeepSeek `deepseek-v4-pro` Chat Completions client;
- read-only evidence from `external/TikTok2026-main`;
- safe summaries of the authorized KuaiRand-Pure training copy;
- local JSONL experiment history, lineage, and research lessons;
- local PDF, Semantic Scholar, arXiv, Crossref, and controlled web evidence.

## Data boundary

- The default authorized dataset is `external/KuaiRand-Pure`.
- The default repository snapshot is `external/TikTok2026-main`.
- The default Starter Kit snapshot is `external/kuairand-starter-kit`.
- The default local literature directory is `external/literature`.
- `RESEARCH_DATASET_ROOT` and the other root environment variables may explicitly override
  these defaults when integration requires a different location.
- `RESEARCH_DENIED_DATASET_ROOT` may identify an additional dataset root that must never be
  accessed.
- `long_view` is the current primary target and may only be used as a training or validation
  target, never as an input feature.
- The data evidence reader does not read row-level validation or public-holdout data.
- The organizer hidden test is not locally available.
- Non-test feedback such as click, like, follow, comment, and forward may only be declared as
  auxiliary training targets. It cannot be used as a same-impression inference feature.
- Validation feedback is evaluation-only.
- Random-exposure logs and video statistic features remain quarantined pending a separate
  leakage review.

## Quality gates

- The user-confirmed latest Problem Statement resolution, official Starter Kit, and typed
  `BenchmarkContract` currently fix `long_view + GAUC/nDCG@5` for local development.
- Official splits, evaluator behavior, and baseline scores are registered as protocol evidence
  with protected file hashes. Tests fail when a protected file changes.
- Starter Kit dates 20220429 through 20220508 form the local `public_holdout`. They are not the
  organizer hidden test and are excluded from iterative development.
- When `evaluation_protocol_status=unconfirmed`, experiment proposals are forbidden. The agent
  must request the formal split and evaluator.
- When `baseline_status=missing`, only an evidence request or a `baseline_reproduction` proposal
  is allowed.
- `baseline_reproduction` requires `BaselineReproductionControl`, a train/validation-only safe
  wrapper, seeds 0 through 4, tolerance `0.002`, CPU-only execution, and explicit leakage risks.
- `OfficialFMConfig` fixes the five official features, `k=16`, `lr=0.001`,
  `batch_size=8192`, `max_epochs=40`, `patience=4`, and the exact Starter Kit symbols.
- Optimization proposals must cite a recorded baseline.
- Target-derived features require out-of-fold, leave-one-out, or strictly-prior-event training
  construction. Validation features must be constructed from training data only.
- Every cited `evidence_ref` must have its `source_ref` in `source_provenance`.
- Decimal scores, improvement values, and percentages must appear in cited evaluation or
  protocol evidence.

## Environment variables

Required for real model calls:

```powershell
$env:RESEARCH_MODEL_API_KEY="your DeepSeek API key"
```

Optional overrides:

```text
RESEARCH_MODEL_BASE_URL
RESEARCH_MODEL_NAME
RESEARCH_MODEL_TIMEOUT_SECONDS
RESEARCH_MODEL_MAX_TOKENS
RESEARCH_REPOSITORY_ROOT
RESEARCH_STARTER_KIT_ROOT
RESEARCH_DATASET_ROOT
RESEARCH_DENIED_DATASET_ROOT
RESEARCH_HISTORY_PATH
RESEARCH_LITERATURE_ROOT
RESEARCH_ONLINE_LITERATURE
RESEARCH_WEB_EVIDENCE_URLS
SEMANTIC_SCHOLAR_API_KEY
```

`RESEARCH_WEB_EVIDENCE_URLS` is a comma-separated list of explicit `http` or `https` URLs. No
arbitrary web page is read when it is unset. `SEMANTIC_SCHOLAR_API_KEY` is optional and can be
used when the anonymous API is rate limited.

## Verification commands

Verify real read-only local context without a paid model call:

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke --offline-literature
```

Retrieve online literature without calling the model:

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke
```

Perform one real DeepSeek Research decision:

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke --call-model --offline-literature
```

The final command incurs API cost. The program does not print or store the API key.
