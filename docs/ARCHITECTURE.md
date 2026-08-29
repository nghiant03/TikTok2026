# Autonomous Recommender Research Architecture

## System boundary

TikTok2026 is a CLI-operated autonomous research controller. Exactly four runtime roles provide typed judgments:

1. Orchestration selects one policy-allowed next action.
2. Research forms evidence-backed hypotheses, experiment specifications, and interpretations.
3. Implementor changes only the assigned worktree and approved scope.
4. Validator performs read-only proposal, implementation, and result review.

Agents do not own authority. Deterministic code owns identity, policy, repository mutation, source registration, execution, evaluation, persistence, resource accounting, routing, and finalization. FastAPI, Uvicorn, REST, SSE, and a browser UI are intentionally excluded.

## Dependency direction

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

Contracts depend only on Pydantic and standard-library types. Agents depend on contracts and injected capabilities, never concrete Git, Docker, SQLite, evaluator, or MLflow implementations. Graph nodes call controller methods only. Evaluation, persistence, memory, and benchmark code do not depend on LangGraph or agent implementations. `bootstrap.py` is the composition root for concrete privileged adapters.

## Runtime lifecycle

```text
bootstrap → inspect → orchestrate → research
→ proposal policy → proposal validation → create worktree
→ implement → diff policy → implementation validation
→ controller source registration → preflight → constrained execution
→ failure classification → protected evaluation → result validation
→ interpretation/persistence → frontier/resource update → orchestrate
→ convergence/budget stop → controller-only finalization → exports
```

Repairable implementation and execution failures retain the experiment and hypothesis identity for at most two repair attempts. Scientific redesign requires a new immutable `ExperimentSpec`. Invalid runs are persisted as failures and never become scientific evidence. Valid non-improvement remains scientific evidence.

LangGraph stores only compact recovery references: run, phase, experiment, hypothesis, worktree, validation/execution/evaluation/decision IDs, repair count, fidelity, pending route, terminal reason, and state version. SQLite and artifacts remain canonical.

## Authority boundaries

| Module | Owns | Must not own |
|---|---|---|
| `contracts` | Versioned models and capability protocols | I/O or framework state |
| `agents` | Prompted judgment and structured validation | Policy exceptions or privileged adapters |
| `graph` | Compact state and finite routing | SQL, shell, Git, Docker, evaluation |
| `controller` | Ordered use cases and persisted transitions | Concrete adapter construction |
| `policies` | Pure path, resource, repair, fidelity, convergence checks | Side effects |
| `repository` | Bounded inspection, diffs, worktrees, source identity | Scientific selection |
| `execution` | Constrained Docker command and failure evidence | Interpretation |
| `evaluation` | Prediction validation and metric provenance | Training or routing |
| `persistence` | Migrations, transactions, audit and record storage | Agent context judgment |
| `memory` | Bounded evidence-backed lesson retrieval | Canonical history |
| `literature` | Configured local licensed-source retrieval | Benchmark performance claims |
| `observability` | Restricted traces, MLflow telemetry, deterministic exports | Canonical scientific truth |
| `bootstrap` | Concrete dependency construction | Domain policy |

## Source, data, and runtime isolation

Every evaluated source state is a validated Git commit in a sibling worktree under an external runtime root. Protected `baseline/` files cannot be modified. Experiment changes are limited to approved scopes.

KuaiRand data is an external, read-only input. Runtime datasets, derived data, checkpoints, predictions, submissions, traces, papers, databases, logs, exports, and worktrees are never committed. The runtime root contains application and graph SQLite files plus artifacts, worktrees, traces, exports, locks, literature cache, and temporary files.

Startup recovery resumes only when persisted source and artifact identities match the worktree and filesystem. Otherwise, stale locks and reservations are preserved for an audited intervention.

## Models, memory, and observability

Each role independently configures a generic OpenAI-compatible Chat Completions client. Role prompts are versioned beside their agent packages. Pydantic validates structured output, with one bounded repair request before a typed failure. Secrets are loaded at call time and redacted from restricted trace artifacts.

Memory retrieval returns bounded experiment-backed lessons, not raw histories. Literature retrieval is limited to configured local sources with explicit license provenance. External training data and pretrained weights remain prohibited.

MLflow records telemetry and artifact IDs only; SQLite and the artifact store remain authoritative. Deterministic Markdown and JSONL exports contain audit events, evaluation validity, lineage references, failures, interventions, and final selection evidence.

## Metrics and finalization

NDCG@10 and Recall@50 are the judging contract. The included within-user binary evaluator is provisional. Starter Kit GAUC and nDCG@5 are diagnostic only and cannot rank champions. Until organizer evaluator code is supplied, all metric records and final bundles are labeled `provisional`.

Iterative agents never receive test labels. Final test access can be claimed exactly once after convergence, is controller-only, and cannot influence later research routing.

## Verification

Default tests are offline and require no paid model, network, Docker, GPU, or KuaiRand data. They cover contracts, policies, migrations, artifacts, resources, benchmark manifests, evaluation, repositories, agents, graph routing, recovery, traces, MLflow references, and two persisted synthetic cycles. Live provider, Docker, dataset, and local MLflow checks are opt-in diagnostics.
