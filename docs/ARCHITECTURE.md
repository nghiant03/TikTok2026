# Autonomous Recommender Research Architecture

## System boundary

TikTok2026 is a CLI-operated research controller. Exactly four runtime roles provide typed judgments:

1. **Orchestration** selects one policy-allowed next action.
2. **Research** forms evidence-backed hypotheses and experiment specifications.
3. **Implementor** edits only the assigned worktree and approved scope.
4. **Validator** performs read-only proposal, implementation, and result review.

Agents do not own authority. Deterministic code owns identity, policy, repository mutation, source registration, execution, evaluation, persistence, resource accounting, routing, and finalization. The current runtime has no web control plane; the supported operator boundary is the Typer CLI in `src/tiktok2026/cli.py`.

## Dependency direction and composition

```text
contracts and pure policies
           ↓
agents and capability protocols
           ↓
controller use cases
           ↓
repository, execution, evaluation, persistence, observability
           ↓
bootstrap composition
```

`src/tiktok2026/contracts` contains versioned Pydantic models and capability protocols. `policies` contains pure decisions. Agent code depends on contracts and injected capabilities, not concrete Git, Docker, SQLite, or evaluator services. Graph nodes call controller operations; they do not issue SQL, shell, Git, Docker, or evaluator calls. `bootstrap.py` is the composition root for the concrete production adapters. The CLI constructs `ProductionOperations`, which delegates service construction to that root.

The production bootstrap currently composes the application and graph SQLite stores, resource ledger, artifact store, Git worktree manager, constrained Docker executor, verified dataset provider, provisional evaluator, and one role-specific OpenAI-compatible client per role. The repository also contains reusable memory, trace, and MLflow protocol/adapter modules, but those are not wired into `build_production_services`; a concrete literature adapter is not present. The dedicated `ResearchAgent` and research context modules are available as typed components, while the current production composition uses `RoleSpecificAgentClient` with bootstrap-supplied prompts and capability names.

## Authority boundaries

| Module | Owns | Must not own |
|---|---|---|
| `contracts` | Versioned models and capability protocols | I/O or framework state |
| `policies` | Protected paths, scope, repair, resource, fidelity, and convergence decisions | Side effects |
| `agents` | Prompted judgment and structured responses | Policy exceptions or privileged adapters |
| `graph` | Compact state and finite routing | SQL, shell, Git, Docker, evaluation |
| `controller` and `use_cases` | Ordered transitions and persisted operation boundaries | Concrete adapter construction |
| `repository` | Inspection, diffs, worktrees, source identity, and patch artifacts | Scientific selection |
| `execution` | Constrained container execution and failure evidence | Interpretation |
| `evaluation` | Prediction validation, metric calculation, and evaluator provenance | Training or routing decisions |
| `persistence` | Migrations, transactions, audit, records, and resource state | Agent judgment |
| `memory` | Bounded experiment-backed retrieval | Canonical history |
| `observability` | Deterministic exports and optional telemetry/trace seams | Canonical scientific truth |
| `bootstrap` | Construction and wiring of concrete services | Domain policy |

The implementor receives a `ScopedWorktreeRepository` only after a worktree is assigned. Its `read_scopes` may include controller-approved contract helpers, but its write scopes are only the approved experiment paths. Validator capabilities are read-only: `read_file`, `diff`, and bounded implementation checks. Research receives evidence/context capabilities rather than source mutation capabilities. Tool arguments and agent payloads are untrusted judgments, not authority; controller-computed identities, policy results, and persisted records are authoritative. The controller, not an agent, creates/registers the source commit and decides whether a transition may proceed.

### Proposal admission

Research proposals must carry a quantitative, technique-neutral
`implementation_resource_estimate`: predicted full-fidelity wall seconds, peak
memory bytes, artifact bytes, and dataset-pass count, with both structural risk
flags clear. The deterministic proposal gate compares wall time, memory, disk,
and dataset passes with the execution envelope and remaining resource state; it
rejects high-cardinality nested scans and duplicate full materialization rather
than accepting an estimate that merely fits today's budget. The current
structural limit is four dataset passes. Proposal validation then checks only
the scientific claims, novelty and authoritative duplicate evidence, bounded
scope, measurable NDCG@10/Recall@50 criteria, leakage, informativeness, and
proportional cost. Source commits, data staging, evaluator arithmetic, artifact
publication, sandboxing, and test access remain controller-owned lifecycle facts.

The experiment registry snapshot is the duplicate authority. A complete
snapshot with no matching evaluated entry is sufficient negative duplicate
evidence; agents must not request an invented second scan. A proposal is
admitted only when its implementation scope is policy-allowed and its mechanism
can be integrated into `src/tiktok2026/experiment/train.py` under the fixed
execution contract. A new specification is required for scientific redesign;
repair feedback does not authorize changing the hypothesis or parent lineage.

### Payload and read/write capability boundaries

The controller sends typed request payloads with bounded subjects and explicit
capabilities. The implementor can read the approved source and controller
contract scopes, write only `allowed_scopes`, inspect its diff, and use the
allowlisted checks. It cannot write contracts, baseline/data/runtime state,
persistence, or unrelated infrastructure. The validator can inspect the bound
worktree and supplied controller check results but has no write capability and
cannot run arbitrary commands. Agents return typed decisions, results, and
reports; they do not create authoritative source, artifact, evaluator, dataset,
or blocker identities. The controller owns validation of payload identity,
policy, hashes, persistence, and routing.

The execution payload is fixed and includes the required controller-injected
`--dataset-manifest-sha256` (and optional dataset-view hash), `--data-root`, and
the source/execution identities. The experiment must score exact valid manifest
rows in order and emit `predictions.json` and `checkpoint_bundle.json`. Generic
artifact envelope, scalar, hash, provenance, and publication checks are
controller/evaluation authority; implementation validation additionally checks
that the experiment-specific mechanism is reconstructed in the entrypoint and
actually determines its permitted outputs. Guarded pre-submit contract checking
is static only and never executes candidate code. Executable smoke is a separate
controller-owned sandbox step, permitted only after implementation validation,
source commit and registration, and sandbox staging; it supplies executable
evidence for the CLI, read-only synthetic inputs, exact artifact set, strict JSON
scalar types, row coverage/order, and provenance.

### Criterion history and escalation

Implementation validation uses the bounded, controller-supplied stable criterion
IDs. A report must assess every supplied criterion exactly once. Criterion
occurrences and evidence-backed resolution claims are append-only and keyed by
report plus criterion, so replaying a report cannot inflate history. Failed or
partial occurrences count as repeats; passing occurrences do not. Blockers keep
stable criterion identity while their prose may vary. A `pass` or `partial`
resolution claim must name existing matching blocker IDs and cite evidence; a
failed criterion cannot claim resolution, and a report cannot resolve a blocker
it introduces. The controller carries unresolved blocker text/evidence into the
next repair and persists each operation idempotently. When resource-feasibility
failure remains unresolved for two criterion occurrences, the controller
escalates to orchestration rather than looping repairs indefinitely; normal
repair limits and a new specification govern other repeated failures.

## Runtime lifecycle

The production graph in `src/tiktok2026/graph/build.py` and `src/tiktok2026/use_cases.py` follows this bounded route:

```text
bootstrap → inspect → orchestrate → research
→ proposal policy → proposal validation → create worktree
→ implement → diff policy → implementation validation
→ source registration → sandbox preflight → executable smoke/constrained execution
→ failure classification → valid-split evaluation → result validation
→ interpretation → persistence → frontier/resource update
→ orchestrate again → convergence or stop
→ provisional finalization → deterministic export → complete
```

Repairable implementation and execution failures retain experiment identity up to two repair attempts. Scientific redesign requires a new `ExperimentSpec`. Invalid runs are persisted as failures and do not become scientific evidence; valid non-improvement remains evidence. The frontier uses the configured plateau epsilon and patience (defaults are `0.002` and `3` in `AppSettings`).

The application database is authoritative for runs, experiments, evaluations, source registrations, artifacts, failures, resources, and audit events. The LangGraph SQLite checkpointer stores only the compact `ProductionState`: run/phase, experiment and hypothesis IDs, worktree and latest result IDs, orchestration decision ID, repair count, fidelity, pending route, terminal reason, and state version. It does not store source files, logs, checkpoints, or full evidence. `resume` loads the latest checkpoint and, for production runs, validates the persisted source/artifact/worktree boundary before continuing.

## Source, data, and runtime isolation

Every production execution is tied to a validated Git commit in a sibling worktree at:

```text
<runtime-root>/worktrees/<run-id>/<experiment-id>
```

The worktree is created from an approved parent commit. The implementor's changed paths are checked against the experiment scope and protected baseline paths. After validation, the controller creates one source commit, records its parent/source identities, normalizes and hashes the patch, publishes the patch artifact, and requires a clean worktree before execution.

`RuntimePaths.create()` rejects a runtime root inside the repository and creates `artifacts`, `worktrees`, `traces`, `exports`, `locks`, `literature`, and `tmp`, plus `application.sqlite3` and `graph.sqlite3`. Dataset files are external read-only inputs. The KuaiRand manifest at `src/tiktok2026/benchmark/kuaireand_pure/manifest.json` identifies files and hashes; production verifies the external manifest and train/valid files before creating the authorized training view. Test data is not placed in that iterative view. Runtime outputs, derived data, checkpoints, predictions, submissions, traces, and databases are not committed.

Recovery refuses to infer missing provenance. It checks the assigned path, source commit, patch artifact hash/URI, worktree cleanliness, and lock identity; it releases a matching stale reservation and removes the stale lock only after those identities agree. Otherwise it persists a rejected-resume audit event and requires intervention.

## Contracts, agents, and models

Contracts in `src/tiktok2026/contracts/models.py` and `ports.py` carry schema versions and reject unknown fields. They cover role requests/responses, evidence, experiment specifications, worktrees, source registrations, execution/evaluation, artifacts, resources, finalization, and audit events. The research context protocol gathers bounded repository, data, memory, and literature evidence concurrently and rejects duplicate, unauthorized, or test-label evidence.

Each role has independent `ModelSettings`: `base_url`, `model`, `api_key_env`, `temperature`, `max_tokens`, and `timeout_seconds`. `OpenAICompatibleClient` sends standard Chat Completions JSON requests and reads the credential at call time. The current production client returns structured role responses; Pydantic validation and one bounded repair are applied by the structured invocation path. Secrets are not serialized into application records.

## Execution, evaluation, and finalization

The Docker executor accepts a typed request, verifies registered source identity, stages only the authorized train/valid dataset view, disables network access, applies read-only and resource limits, and publishes bounded output artifacts. The staged manifest describes only files visible to training and therefore has a different canonical hash from the full authoritative manifest. The executor supplies the authoritative manifest SHA as a separate controller-owned argument; training records that opaque identity in its artifacts, and publication verifies it against the registered dataset authority. Failure classification distinguishes execution failures from scientific non-improvement.

The evaluation registry validates prediction shape, row identity, ordering, hashes, dataset manifest identity, source, checkpoint, and execution provenance. NDCG@10 and Recall@50 are the judging metrics and their mean is the local validation ranking. The repository's evaluator is `provisional`; the protected Starter Kit's incompatible metrics are diagnostic only. Current production composition does not configure an organizer evaluator or expose a CLI final-test operation. Therefore all current evaluation records and finalization records are provisional and must not be described as official results.

Finalization is controller/persistence controlled and guarded. Orchestration may
select `stop` only when the controller marks `finalization_ready`; an agent's
stop rationale never authorizes finalization. The finalization transition
requires converged persisted run and experiment state, an eligible source
registration, a matching evaluation/checkpoint and evaluator identity, and a
materialized bundle. The controller creates the bundle first, then atomically
persists the `ProvisionalFinalizationRequest` and rejects missing or mismatched
provenance. Export is gated on that persisted record. The bundle records source,
checkpoint, evaluation, evaluator, and provenance references. The CLI export
service writes audit-event records, sorted deterministically by event ID, as
`iterations.jsonl` and `iterations.md` under the external run export directory.
The bundle and exports are records of a provisional local run, not organizer
submissions or official scores; any official test access requires a separate
controller-issued authorization claim and is not available to iterative agents.

## Runtime status and verification boundary

The checked-in `judged.toml` profile supplies zero resource limits and must be completed by an operator through an external configuration/profile before a live run. A live production run additionally needs a verified external dataset manifest, all four model credentials, an approved repository commit, and an immutable Docker image digest. `diagnostics` currently checks the committed benchmark/protected-file manifest and reports evaluator status; it does not perform live model, Docker, dataset, or MLflow smoke checks.

Synthetic lifecycle operation uses scripted agents, a deterministic fixture evaluator/executor, a temporary source-manager fixture that creates directories and fabricated source identities without Git operations, the same persistence/artifact/resource boundaries, and external runtime paths. It is suitable for offline lifecycle checks and needs no network, paid model, Docker, GPU, or KuaiRand data. It is not evidence of benchmark performance.
