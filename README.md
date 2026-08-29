# TikTok2026 Autonomous Recommender Research

TikTok2026 is a CLI-operated research controller for forming, implementing, validating, executing, evaluating, and learning from recommender-system experiments. LLM agents provide typed judgments; deterministic services retain authority over policy, identity, Git, execution, evaluation, persistence, resources, routing, and finalization.

The canonical architecture is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The files under `baseline/` are protected Starter Kit references, not the editable experiment target.

## Setup

```bash
uv sync --dev
export TIKTOK2026_RUNTIME_ROOT="$(dirname "$PWD")/TikTok2026.runtime"
export TIKTOK2026_KUAIRAND_PURE_DATA=/external/read-only/KuaiRand-Pure
uv run tiktok2026 runtime-init --runtime-root "$TIKTOK2026_RUNTIME_ROOT"
uv run tiktok2026 verify-manifests
```

Runtime state and datasets must be outside the repository. Dataset mounts are read-only and identified by manifests and hashes.

## Operator commands

```text
runtime-init       Create external runtime directories and databases
migrate            Apply checksummed application and graph migrations
verify-manifests   Verify protected benchmark reference hashes
synthetic-run      Run offline deterministic lifecycle verification
run                Start a configured production run
resume             Resume an identity-verified interrupted run
inspect            Read persisted run audit state
finalize           Finalize an eligible converged run
export             Export deterministic Markdown and JSONL records
diagnostics        Verify configured environment boundaries
```

The HTTP, SSE, FastAPI, and Uvicorn control surfaces are intentionally out of scope.

## Models and evaluation

Each of the four roles can use an independent OpenAI-compatible Chat Completions endpoint. Configure `base_url`, `model`, `api_key_env`, `temperature`, `max_tokens`, and `timeout_seconds` per role. Credentials are read from environment variables at call time and are not persisted.

Judging metrics are NDCG@10 and Recall@50. The included evaluator and locally generated final bundles are explicitly `provisional`; they must never be represented as official organizer results. Iterative agents cannot access test labels. A controller-only test claim is single-use after convergence and cannot route later research.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv run tiktok2026 synthetic-run --iterations 2 \
  --runtime-root "$(dirname "$PWD")/TikTok2026.synthetic-runtime"
```

Default verification requires no network, paid model, Docker, GPU, or KuaiRand data. Live Docker, provider, dataset, and MLflow checks are operator-enabled diagnostics only.

Production `run` and `resume` load the committed `config/budgets/judged.toml` profile by default. Supply `--profile-path` for another committed profile and `--operator-config` for an external TOML containing operator-specific settings such as model endpoints, dataset paths, and the immutable Docker image.
