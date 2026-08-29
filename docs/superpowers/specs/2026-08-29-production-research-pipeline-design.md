# Production Research Pipeline Design

## 1. Purpose

Complete the canonical TikTok2026 autonomous recommender-system research controller by selectively integrating the useful work from `origin/research-agent` and `origin/orchestration-agent`, then implementing the remaining production pipeline without importing prohibited datasets, generated outputs, duplicate repository snapshots, or unsafe evaluation behavior.

The completed system supports four runtime roles: Orchestration, Research, Implementor, and Validator. Agents provide typed judgments. Deterministic code retains authority over identity, policy, repository mutation, source registration, execution, evaluation, persistence, resource accounting, routing, and finalization.

### Implementation status at Phase 3 HEAD `9122a9f`

- [x] Canonical contracts, policies, persistence, repository/worktree, execution, evaluation, agent, search/observability, controller/graph, bootstrap, CLI, and recovery implementation paths are present.
- [ ] Targeted tests, full verification, and live-environment checks are not claimed here; validation remains owned by the parent lane.
- [ ] Phase 4 acceptance, including any contributor-authorship history operation, remains unresolved.

## 2. Scope

The implementation includes:

- clean integration of the two contributor branches;
- authoritative typed contracts and capability protocols;
- checksummed SQLite migrations and repositories;
- audit events, artifact registration, and resource accounting;
- protected benchmark and evaluator boundaries;
- editable experiment code derived from baseline concepts without modifying protected files;
- sibling Git worktree management and pre-execution source commits;
- constrained Docker execution with opt-in live integration tests;
- all four typed OpenAI-compatible agents;
- deterministic proposal, patch, validation, failure, budget, convergence, and finalization policies;
- compact LangGraph state and deterministic routing;
- research memory, literature provenance, duplicate detection, and bounded frontier search;
- structured logs, restricted traces, MLflow telemetry, and deterministic Markdown/JSONL exports;
- Pydantic settings and a composition root;
- Typer CLI commands for initialization, execution, inspection, finalization, and export;
- network-free unit, integration, architecture, and synthetic lifecycle tests;
- optional live checks for Docker, KuaiRand-Pure, and configured model providers.

The implementation excludes:

- FastAPI, SSE, Uvicorn, and every HTTP control-plane surface;
- frontend code;
- external training data and pretrained weights;
- automatic use of organizer hidden-test labels during research;
- full MCTS, distributed scheduling, and simultaneous GPU runs;
- an official result claim before an organizer evaluator is configured.

## 3. Integration Strategy and Contributor Attribution

### 3.1 Source branches

The source implementations are:

- `origin/research-agent`, authored by Lumos088;
- `origin/orchestration-agent`, authored by AndyYeom.

The source commits remain referenced in integration provenance. Directly merging either branch is prohibited because it would permanently introduce large copied datasets, PDFs, nested repository snapshots, generated submissions, run logs, duplicate contracts, protected-code copies, and behavior that conflicts with the canonical judging contract.

### 3.2 Clean reconstructed branches

Create two branches from the current canonical `main`:

1. `integration/research-agent`
2. `integration/orchestration-agent`

The research branch ports reusable contract, context-building, bounded repair, provider-client, provenance, and policy concepts into `src/tiktok2026/agents/research`, `src/tiktok2026/literature`, and `src/tiktok2026/memory`. Canonical `tiktok2026.contracts` replace the standalone copied contract loader. NDCG@10 and Recall@50 replace GAUC and nDCG@5 wherever judging behavior is represented.

The orchestration branch ports useful graph-loop, recovery, sandbox, run-log, and finalization concepts into canonical graph, policy, repository, execution, and observability boundaries. Fixed recommender recipes, test-guided iteration, direct subprocess authority, mutable in-repository runtime directories, generated submissions, and free-form dictionaries are excluded.

Each reconstructed branch commit preserves the corresponding original source author identity. Commit messages describe the compliant behavior being integrated and include the original source commit hash in the body. The integration branch merges both reconstructed branches so Git records both lines of contribution before the remaining pipeline is implemented.

### 3.3 Integration branch

Create `integration/production-pipeline` from `main`, merge the two reconstructed branches, and implement the remaining production system there. Merge into `main` only after targeted tests, the complete test suite, Ruff, and Pyright pass and protected baseline files are unchanged.

## 4. Architectural Boundaries

### 4.1 Roles

Exactly four runtime roles exist:

- Orchestration selects among allowed typed actions but cannot execute them.
- Research proposes hypotheses, experiment specifications, evidence requests, and interpretations.
- Implementor edits only an assigned worktree within an approved scope.
- Validator performs read-only adversarial proposal, implementation, and result review.

No additional permanent persona may be introduced. Deterministic classifiers, evaluators, routers, and policy gates are services, not agents.

### 4.2 Dependency direction

- `contracts` imports only standard-library types and Pydantic.
- pure `policies` import contracts and perform no I/O;
- agents import contracts and injected capability protocols;
- graph nodes call controller use cases, never SQL, shell, Git, Docker, evaluator, or MLflow implementations directly;
- evaluation, persistence, memory, and benchmark modules do not import LangGraph or agents;
- privileged adapters depend inward and are connected only by `bootstrap.py`.

Architecture tests enforce forbidden imports and role count.

### 4.3 Runtime storage

Mutable state is stored under `TIKTOK2026_RUNTIME_ROOT`, which must resolve outside the repository and all experiment worktrees. The runtime tree contains application and graph SQLite files, worktrees, artifacts, traces, literature cache, MLflow files, locks, exports, and temporary paths.

Datasets are external read-only inputs. Dataset identity is established by manifest metadata and hashes. Runtime outputs, derived data, checkpoints, full papers, traces, submissions, and histories are never committed.

## 5. Contracts and Configuration

### 5.1 Contracts

Extend the current contracts with versioned representations for:

- run and graph references;
- benchmark and dataset manifests;
- hypotheses and research decisions;
- role requests, responses, and typed agent failures;
- repository observations, worktree assignments, patches, and source registrations;
- execution and evaluator requests;
- artifact records and checkpoint bundles;
- resource reservations and usage;
- experiment, validation, failure, lesson, frontier, convergence, and finalization records;
- trace, audit, and export events.

Contracts remain immutable, reject unknown fields, and carry schema versions. Authoritative cross-role data never uses free-form dictionaries.

### 5.2 Settings

Pydantic settings load in this order:

1. committed TOML profile;
2. optional operator TOML outside Git;
3. environment variables;
4. CLI overrides.

Unknown keys and invalid judged profiles are rejected. Settings cover runtime/data paths, SQLite files, Docker identity, fidelity commands and limits, evaluator identity, MLflow URI, logging, literature sources, and per-role model configuration.

Each role has independently configurable:

- `base_url`;
- `model`;
- `api_key_env`;
- `temperature`;
- `max_tokens`;
- `timeout_seconds`.

The shared client uses the standard OpenAI Chat Completions interface, allowing common OpenAI endpoint models and compatible providers. Credentials are read only from the configured environment variable or mounted secret file, and are never logged or persisted.

## 6. Deterministic Core

### 6.1 Persistence

A migration runner applies numbered SQL files transactionally, records migration checksums, and rejects an applied migration whose checksum changed. Repositories provide typed operations for runs, hypotheses, specs, decisions, validation reports, source registrations, executions, evaluations, failures, lessons, artifacts, resource ledger entries, frontier candidates, finalizations, and audit events.

State-changing operations create audit events in the same transaction where practical. Idempotency keys prevent duplicate records during graph replay. Application SQLite is canonical scientific state; LangGraph SQLite is recoverable workflow state.

### 6.2 Artifacts and resources

Artifacts are written to temporary sibling paths, flushed, hashed, atomically renamed, and then registered. Records contain media type, byte size, SHA-256, producer, source commit, run/experiment IDs, and retention class.

The resource ledger atomically reserves, consumes, releases, and reconciles GPU time, wall time, token use, and disk. Final-evaluation reserves cannot be consumed by iterative experiments. Startup recovery reconciles stale reservations and locks.

### 6.3 Policies

Pure policies enforce:

- protected paths and protected hashes;
- approved implementation scope;
- exact duplicate blocking and semantic-similarity warnings;
- two repairs per experiment specification;
- legal fidelity transitions;
- budget reservation and exhaustion;
- valid failure classification and retry behavior;
- convergence at improvement no greater than `0.002` for three consecutive experiments;
- single final test access after convergence;
- prohibition of external training data and pretrained weights;
- prohibition of test evidence in agent context or later routing;
- provisional result labeling when the official evaluator is unavailable.

## 7. Repository, Execution, and Evaluation

### 7.1 Repository services

`RepositoryInspector` provides bounded maps, searches, symbol observations, and excerpts. `WorktreeManager` creates sibling worktrees under the runtime root from a validated parent commit, records assignments, enforces one experiment per worktree, and supports recovery and cleanup.

Implementor writes are confined to the assigned worktree and approved paths. A deterministic patch policy normalizes diffs, rejects protected or out-of-scope changes, verifies parent identity, computes exact signatures, and captures a patch artifact. After Validator approval, the controller creates the canonical experiment commit and registers its identity before preflight or execution.

### 7.2 Experiment target

Editable model, feature, training, prediction, and checkpoint code lives in `src/tiktok2026/experiment`. It reproduces necessary baseline concepts without importing the protected baseline as the editable target. Training and prediction entry points accept explicit paths and configuration, use deterministic seeds, produce versioned bundles, and never evaluate test labels.

### 7.3 Docker execution

The Docker adapter receives a typed execution request. It uses an approved image identity, no network, read-only source and dataset mounts, a writable artifact mount, explicit CPU/GPU/memory/time limits, a fixed command allowlist, cancellation, and process cleanup. It captures bounded stdout/stderr artifacts and telemetry, then returns a typed `ExecutionResult`.

Default tests use a fake executor. Marked integration tests may use live Docker only when the operator enables them and supplies the required image and external dataset.

### 7.4 Evaluation

The benchmark adapter verifies dataset and protected Starter Kit manifests, validates prediction identity and shape, and writes row-preserving submissions. The evaluator registry resolves immutable evaluator identities.

NDCG@10 and Recall@50 are authoritative. The repository includes a versioned provisional within-user binary evaluator. Every provisional metric and bundle is labeled `provisional`, never `official`. Starter Kit GAUC/nDCG@5 remains diagnostic only and cannot rank champions or finalize judged results.

After convergence, the controller may perform exactly one test evaluation. While the organizer evaluator is unavailable, this produces a clearly marked provisional final bundle containing the selected source, checkpoint, predictions, submission, manifests, evaluator identity, hashes, resource totals, interventions, and logs. No later research decision may consume the final result.

## 8. Agents

### 8.1 Shared client behavior

A generic asynchronous OpenAI-compatible client sends role prompts and requests structured JSON output. Pydantic validates responses. One schema-repair request is allowed; a second invalid response becomes a typed agent failure. Token usage and restricted trace artifacts are recorded without secrets.

Prompts are versioned beside their owning packages and hashed into each call record.

### 8.2 Research integration

Port the source branch's useful bounded context, concurrent capability gathering, evidence provenance, history/lineage retrieval, literature retrieval, leakage checks, and one-repair behavior. Replace standalone snapshots and duplicated contracts with injected canonical capabilities.

Research can return one of:

- an evidence request;
- a hypothesis-backed experiment specification;
- a typed interpretation of an execution or evaluation result.

It cannot write source, run commands, access test labels, invoke an evaluator, or authorize external assets.

### 8.3 Orchestration integration

Port the source branch's useful iteration, error recovery, convergence, and finalization concepts, but not its fixed BPR queue or direct execution. Orchestration receives bounded frontier, lesson, validation, failure, convergence, and resource summaries and returns one allowed typed action.

A deterministic route policy validates the proposed action and maps it to a known graph route.

### 8.4 Implementor and Validator

Implementor receives an approved spec, parent commit, worktree assignment, editable scope, repository excerpts, and allowed checks. It returns changed paths/symbols, checks, assumptions, and unresolved issues.

Validator has stage-specific requests for proposals, implementations, and results. It remains read-only and returns blockers, warnings, evidence references, leakage/fidelity fields, and a verdict. Deterministic policy evidence cannot be overridden by Validator judgment.

## 9. Graph and Controller Data Flow

The production lifecycle is:

```text
bootstrap
  -> inspect
  -> orchestrate
  -> research
  -> proposal policy
  -> proposal validation
  -> create worktree
  -> implement
  -> diff policy
  -> implementation validation
  -> controller source commit
  -> deterministic preflight
  -> Docker execution
  -> failure classification
  -> protected evaluation
  -> result validation
  -> interpretation and persistence
  -> frontier/resource update
  -> orchestrate again
  -> convergence or budget stop
  -> one controller-only final evaluation
  -> provisional final bundle and exports
```

Repairable implementation or execution faults return to Implementor with the same experiment and hypothesis IDs, up to two attempts. Scientific redesign creates a new specification. Invalid runs are persisted as failures but never become scientific evidence. Valid non-improvement is persisted as scientific evidence.

Graph state stores only compact recovery references: run ID, phase, current experiment and hypothesis IDs, worktree ID, latest report/result IDs, decision ID, repair count, fidelity, pending route, terminal reason, and state version.

## 10. Memory, Literature, Search, and Observability

### 10.1 Memory and literature

Memory retrieval returns bounded typed experiment facts, lineage neighbors, failures, and evidence-backed lessons. Raw histories and traces remain in persistence/artifacts.

Literature adapters support configured local documents and public metadata/source retrieval. Cached full text requires recorded open-license evidence. Web pages are read only from explicitly configured URLs. Literature may motivate mechanisms but cannot establish local benchmark performance.

### 10.2 Frontier

The initial frontier is bounded to the champion, two diverse alternatives, and one diagnostic or repair slot. Exact normalized signatures are blocked. Semantic similarity is advisory evidence. Ranking uses persisted validation metrics and resource-aware deterministic scoring, not agent prose.

### 10.3 Observability and exports

Structured logs carry run, experiment, causation, correlation, actor, and source identities. Restricted trace artifacts retain prompts, outputs, tool calls, usage, and hashes. MLflow records telemetry and artifact references but is not canonical scientific storage.

Deterministic Markdown and JSONL exports include each hypothesis, source diff reference, validation metrics, evaluator validity, errors and recovery, intervention count, token usage, GPU hours, lineage, final selection, and artifact hashes.

## 11. CLI

Typer exposes production commands for:

- initializing and verifying runtime storage;
- applying and checking migrations;
- verifying dataset and protected manifests;
- running a synthetic lifecycle;
- starting or resuming a production research run;
- inspecting run, experiment, resource, and failure state;
- finalizing an eligible run;
- exporting Markdown/JSONL audit bundles;
- running opt-in environment diagnostics for Docker, data, evaluator, MLflow, and model endpoints.

Commands return nonzero exit status for policy, configuration, migration, or runtime failures. Human interventions are explicit CLI operations that create audit events.

## 12. Error Handling and Recovery

Boundary exceptions are converted to typed failure records. Failure classification distinguishes syntax/import, dependency/environment, missing path, CUDA/CPU OOM, NaN divergence, schema mismatch, evaluator output, timeout, disk, corrupted checkpoint, unstable validation, and scientific non-improvement.

Repository, artifact, and persistence operations use locks and idempotency identifiers. Startup recovery detects incomplete writes, abandoned worktrees, stale reservations, and interrupted graph states. It resumes only when source, artifact, and database identities agree; otherwise it records a blocking intervention requirement.

## 13. Testing and Verification

Default verification requires no network, paid model, Docker, GPU, or KuaiRand data.

Required tests include:

1. contract serialization, strictness, invariants, and schema versions;
2. migration checksums, transactions, replay, idempotency, and lineage;
3. artifact hashing, atomic publication, and resource accounting;
4. protected-path, duplicate, budget, retry, convergence, test-isolation, and finalization policies;
5. repository worktree lifecycle and source-registration behavior using temporary Git repositories;
6. evaluator metric fixtures, invalid predictions, provisional labeling, and single final access;
7. agent contexts, capability restrictions, provider request construction, response repair, and typed failures using scripted clients;
8. every legal graph route and failure path;
9. CLI configuration, commands, exit codes, and exports;
10. forbidden imports, four-role invariant, runtime-root exclusion, and protected baseline hashes;
11. two complete synthetic autonomous cycles including persistence, artifacts, validation, frontier updates, convergence/finalization, and exports;
12. failure injection for malformed agent output, invalid patches, failed preflight, timeout, OOM evidence, missing artifacts, and evaluator rejection.

Opt-in integration tests cover:

- a constrained Docker execution;
- an external KuaiRand-Pure smoke/proxy run;
- configured OpenAI-compatible endpoint calls for each role;
- local MLflow artifact registration;
- provisional final bundle generation from a real configured environment.

Completion requires targeted tests and the full suite to pass, followed by `uv run ruff check .` and `uv run pyright`. Protected baseline files must match their pre-integration hashes, and `git status` must not contain runtime outputs.

## 14. Documentation and Status

Update `README.md`, `docs/ARCHITECTURE.md`, CLI help, configuration examples, and module-level scoped instructions where boundaries changed. Remove FastAPI and SSE from the recommended repository tree, dependency list, responsibility table, configuration, and implementation status. Document that HTTP control surfaces are intentionally out of scope.

The final architecture status must distinguish implemented production code from opt-in environment verification and must state that generated final bundles remain provisional until an organizer evaluator is configured.

## 15. Acceptance Criteria

The work is accepted when:

- both contributor implementations are represented by clean reconstructed branch commits with original authorship and source-commit provenance;
- no source-branch dataset, generated submission, copied repository, copied Starter Kit, PDF, runtime history, or trace enters the integrated tree;
- exactly four typed agents run behind least-privilege capabilities;
- deterministic policy owns every privileged transition;
- all evaluated source states are validated commits in sibling runtime worktrees;
- canonical metrics are NDCG@10 and Recall@50 and all non-organizer results are labeled provisional;
- test labels cannot affect iterative research and final evaluation is single-use;
- runtime state and datasets remain external to Git;
- Typer CLI can initialize, run/resume, inspect, finalize provisionally, and export;
- common OpenAI endpoint models and compatible providers can be configured independently per role;
- two network-free autonomous synthetic cycles complete with a reconstructable audit trail;
- all required tests, Ruff, and Pyright pass;
- protected baseline references are unchanged.
