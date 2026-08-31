# TikTok2026 Autonomous Recommender Research

TikTok2026 is a CLI-operated research controller for recommender-system experiments. Agents return typed judgments; deterministic controller code owns identity, policy, repository mutation, execution, evaluation, persistence, resource accounting, routing, and finalization. The protected files under `baseline/` are references, not the editable experiment target. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the current boundaries.

## Setup and operator configuration

Install the project from the repository root:

```bash
uv sync --dev

export REPO_ROOT="$PWD"
export RUNTIME_ROOT="/external/tiktok2026-runtime"
export TIKTOK2026_RUNTIME_ROOT="$RUNTIME_ROOT"
export TIKTOK2026_KUAIRAND_PURE_DATA="/external/read-only/KuaiRand-Pure"

uv run tiktok2026 runtime-init \
  --runtime-root "$RUNTIME_ROOT" --repository-root "$REPO_ROOT"
uv run tiktok2026 migrate \
  --runtime-root "$RUNTIME_ROOT" --repository-root "$REPO_ROOT"
uv run tiktok2026 verify-manifests --repository-root "$REPO_ROOT"
```

`runtime-init` creates the external application and graph SQLite databases and runtime directories, and applies migrations. `migrate` can be run again after migration changes. The runtime root must be outside the repository. Dataset files are external, read-only inputs; the repository manifest identifies their files and hashes. `verify-manifests` verifies the committed benchmark manifest and protected baseline reference hashes; it does not verify the external dataset.

Production uses `config/budgets/judged.toml` by default. That checked-in profile contains zero resource limits and is a template, not a runnable judged configuration. Use `--profile-path` for another profile and `--operator-config` for an external TOML file. The operator file may provide `dataset_root`, an immutable `docker_image` containing `@sha256:...`, budget values, and a `[models.<role>]` table for each of `orchestration`, `research`, `implementor`, and `validator`. Each model table supports `base_url`, `model`, `api_key_env`, `temperature`, `max_tokens`, and `timeout_seconds`. Credentials are supplied through the named environment variables and are not persisted.

For example, `/external/tiktok2026-operator.toml` can contain the following settings (repeat the model table for each role when using different endpoints):

```toml
dataset_root = "/external/read-only/KuaiRand-Pure"
docker_image = "registry.example/tiktok2026@sha256:REPLACE_WITH_IMAGE_DIGEST"

[execution]
timeout_seconds = 900
memory_bytes = 4294967296
cpus = 1.0
gpu_count = 0

[budget]
gpu_hours = 1.0
wall_clock_seconds = 7200
tokens = 200000
disk_bytes = 21474836480
reserved_final_gpu_hours = 0.25
frontier_capacity = 4
max_repairs = 3

[models.orchestration]
base_url = "https://provider.example/v1"
model = "operator-approved-model"
api_key_env = "TIKTOK2026_ORCHESTRATION_API_KEY"
temperature = 0.0
max_tokens = 4096
timeout_seconds = 120.0
```

`execution.gpu_count` defaults to `0`. Set it to a positive count only when the host Docker daemon has the required GPU runtime and the pinned image supports that accelerator.

Set the corresponding credential variables, including the variables named by the other role tables, before invoking `run` or `resume`.

### LiteLLM gateway for all agents

The four runtime agents can share a local LiteLLM OpenAI-compatible gateway. The
checked-in configuration in `config/litellm/config.yaml` maps the
`tiktok2026-chatgpt` alias to LiteLLM's documented `chatgpt/gpt-5.4` provider,
and `config/litellm/operator-models.toml` configures that gateway for
orchestration, research, implementor, and validator. Merge those four model
tables into the external operator TOML used for a production run; do not add
credentials to the repository.

Install and start the gateway in one terminal:

```bash
uv sync --dev --group gateway
export LITELLM_MASTER_KEY="$(openssl rand -hex 32)"
export LITELLM_API_KEY="$LITELLM_MASTER_KEY"
uv run --group gateway litellm \
  --config "$REPO_ROOT/config/litellm/config.yaml" \
  --host 127.0.0.1 --port 4000
```

During proxy startup or first use, LiteLLM prints an OAuth device code and
verification URL. Complete that flow in the gateway terminal. LiteLLM stores
the resulting tokens locally for reuse. The current LiteLLM documentation
specifies this subscription flow for ChatGPT Pro and Max; ChatGPT Plus is not
documented as a supported subscription tier. A Plus subscription therefore
cannot be assumed to authorize this integration. Use an eligible Pro/Max
subscription or configure a separately billed OpenAI API key instead.

With the gateway running, start the controller using an operator TOML containing
the four tables from `config/litellm/operator-models.toml`:

```bash
uv run tiktok2026 run \
  --runtime-root "$RUNTIME_ROOT" --repository-root "$REPO_ROOT" \
  --profile-path "$REPO_ROOT/config/budgets/judged.toml" \
  --operator-config /external/tiktok2026-operator.toml
```

The controller still sends ordinary OpenAI-compatible `/v1/chat/completions`
requests. LiteLLM bridges supported ChatGPT subscription models to the
Responses API, so agent contracts and deterministic controller boundaries are
unchanged.

For a live run, the configured dataset directory must contain `manifest.json` and pass train/valid manifest verification; all four role credentials must be available; the repository must have an approved Git commit; and the Docker image must be pinned by digest. The current bootstrap wires SQLite persistence, the constrained Docker executor, Git worktrees, the verified training dataset view, the provisional evaluator, and four role-specific OpenAI-compatible clients. A configured `mlflow_uri` is accepted by settings, but MLflow telemetry, trace sinks, and a concrete literature reader are not composed into the current production bootstrap.

## CLI

The executable is `uv run tiktok2026`. The commands and their actual options are:

```text
runtime-init   --runtime-root PATH [--repository-root PATH]
migrate        --runtime-root PATH [--repository-root PATH]
verify-manifests [--repository-root PATH]
synthetic-run  [--iterations INTEGER] [--runtime-root PATH]
calibrate-baseline --runtime-root PATH [--repository-root PATH]
                   [--profile-path PATH]
run            --runtime-root PATH [--repository-root PATH]
               [--profile-path PATH] [--operator-config PATH] [--synthetic]
resume         --runtime-root PATH --run-id TEXT [--repository-root PATH]
recover-source-registration --runtime-root PATH --run-id TEXT [--repository-root PATH]
               [--profile-path PATH] [--operator-config PATH] [--synthetic]
inspect        --runtime-root PATH --run-id TEXT
finalize       --runtime-root PATH --run-id TEXT [--repository-root PATH] [--synthetic]
export         --runtime-root PATH --run-id TEXT [--repository-root PATH] [--synthetic]
diagnostics    [--repository-root PATH]
```

For example, a configured production run and its later operations are:

```bash
uv run tiktok2026 calibrate-baseline \
  --runtime-root "$RUNTIME_ROOT" --repository-root "$REPO_ROOT" \
  --profile-path "$REPO_ROOT/config/budgets/judged.toml"

uv run tiktok2026 run \
  --runtime-root "$RUNTIME_ROOT" --repository-root "$REPO_ROOT" \
  --profile-path "$REPO_ROOT/config/budgets/judged.toml" \
  --operator-config /external/tiktok2026-operator.toml

uv run tiktok2026 resume --runtime-root "$RUNTIME_ROOT" --run-id RUN_ID \
  --repository-root "$REPO_ROOT" \
  --operator-config /external/tiktok2026-operator.toml
uv run tiktok2026 inspect --runtime-root "$RUNTIME_ROOT" --run-id RUN_ID
uv run tiktok2026 finalize --runtime-root "$RUNTIME_ROOT" --run-id RUN_ID
uv run tiktok2026 export --runtime-root "$RUNTIME_ROOT" --run-id RUN_ID
```

`run --synthetic` and `resume --synthetic` select the deterministic fixture composition. The `--synthetic` options on `finalize` and `export` are accepted by the CLI but are not used by those operations. `diagnostics` currently verifies the repository benchmark manifest and reports the evaluator status; it is not a live provider, Docker, data, or MLflow smoke test.

## Synthetic and offline operation

```bash
uv run tiktok2026 synthetic-run --iterations 2 \
  --runtime-root "$RUNTIME_ROOT/synthetic"
```

Synthetic lifecycle tests use scripted agents, deterministic fixture rows, a fake executor/evaluator, the same persistence, artifact, resource, graph, and export boundaries, and no network, paid model, Docker, GPU, or KuaiRand data. They are lifecycle checks only and do not produce official benchmark results. The default `synthetic-run` runtime is an external sibling of the repository when `--runtime-root` is omitted.

## Metrics and finalization

The judging contract is GAUC and nDCG@5, with the local validation ranking defined as their mean. The repository evaluator and synthetic evaluator return `validity="provisional"`; the protected Starter Kit evaluator is used for diagnostic parity checks only. No local metric, run, submission, or final bundle is an official organizer result.

`calibrate-baseline` is standalone operator tooling, not a graph node. It verifies the protected Starter Kit and external train/valid manifest, runs the unchanged FM validation pipeline once, evaluates its predictions under the current GAUC/nDCG@5 contract, and records the protected Starter Kit values as diagnostics in an immutable identity-keyed record under the sibling runtime root. Repeating the command with the same dataset manifest, evaluator, Starter Kit source, and FM configuration returns the cached record without retraining. Runtime evaluation logs use the matching calibration for baseline deltas.

Iterative execution receives only the verified train/valid view. The contracts and persistence layer contain one controller-authorized test-access boundary, but the current CLI does not expose an organizer evaluator or a separate final-test command. Until that evaluator is supplied and configured, finalization is provisional only.

`finalize` requires persisted converged run/experiment state, a registered source, an evaluation, and a matching bundle artifact. It writes a provenance-bearing bundle under the external runtime artifacts tree. `export` requires that persisted finalization and writes deterministic audit records to:

```text
$RUNTIME_ROOT/exports/<run-id>/iterations.jsonl
$RUNTIME_ROOT/exports/<run-id>/iterations.md
```

The export contains the persisted audit-event records, sorted deterministically by event ID. Application and graph databases, artifacts, worktrees, logs/traces, exports, and temporary files remain outside Git.

## Recovery

`resume` requires a durable graph checkpoint for the supplied run ID. For production runs, recovery checks the pending worktree/source/artifact identities, assigned worktree path and commit, cleanliness, lock ownership, and stale reservation before resuming. Matching stale reservations are released and stale locks are removed; an identity mismatch is rejected and recorded as a resume audit event rather than guessed through. Synthetic resume uses the fixture composition and does not perform the production boundary reconciliation.
