# Repository Instructions

## Purpose

Build an autonomous recommender-system research controller that can form, implement, execute, evaluate, and learn from experiments while preserving scientific validity and a complete audit trail.

## Architectural invariants

- LLM agents make judgments. Deterministic code owns authority, identity, policy, execution, evaluation, persistence, and resource accounting.
- Preserve exactly four runtime roles: Orchestration, Research, Implementor, and Validator. Do not add permanent personas without an architecture decision.
- LangGraph coordinates typed state transitions. It is not the research database or artifact store.
- Runtime agents communicate through contracts in `src/tiktok2026/contracts`; do not use free-form agent-to-agent protocols for authoritative data.
- `baseline/README.md`, `baseline/data.py`, `baseline/evaluate.py`, `baseline/submit.py`, and `baseline/baseline_scores.json` are protected reference files. Never modify them during experiments.
- The judging metric contract is GAUC and nDCG@5, with their arithmetic mean as the primary score. The repository evaluator remains provisional; provisional results must never be labeled official.
- Test data is unavailable to agents. A controller-only test evaluation may occur once after convergence and must not route later research decisions.
- Do not encode recommender recipes, fixed model sequences, or technique-specific experiment tools in infrastructure.
- Agents receive least-privilege tool sets in code, not merely prompt instructions.
- Every evaluated source state is a validated Git commit in an isolated sibling worktree.
- Dataset files are external, read-only inputs identified by manifests and hashes. Do not commit runtime datasets or derived data.
- Mutable runtime state belongs in the configured sibling runtime root, never inside the repository or experiment worktrees.
- External training data and pretrained weights are prohibited. Literature, documentation, and public source references are allowed with provenance.
- Keep source files, logs, checkpoints, full papers, and full histories out of LangGraph state. Store IDs and bounded summaries.
- Do not make unrelated refactors while implementing an experiment or infrastructure ticket.

## Dependency direction

`contracts` and pure `policies` are the innermost modules. Agents and graph nodes depend on contracts and capability protocols. Privileged implementations in repository, execution, evaluation, persistence, and observability depend inward and are composed only at application bootstrap. Evaluation, persistence, memory, and benchmark code must not depend on LangGraph or agent implementations.

Agents must not import concrete Docker, Git, SQLite, MLflow, or evaluator implementations. Graph nodes must not issue SQL, shell commands, Git operations, or evaluator calls directly.

## Development rules

- Python 3.11, Ruff, Pyright, and pytest are the standard toolchain.
- Prefer small typed functions and protocols over service-class hierarchies.
- Add contract tests for schemas, routing tests for graph changes, and policy/failure tests for privileged boundaries.
- Tests must not require network access, paid LLM calls, KuaiRand data, Docker, or GPUs unless explicitly marked integration-only.
- Never expose secrets in config, prompts, logs, traces, fixtures, or commits.
- Runtime prompts live with their owning agent package and are versioned and hashable.
- SQL schema changes use numbered, checksummed migrations. Never mutate an applied migration.
- Public APIs return typed, versioned representations and all state-changing operations create audit events.

## Definition of done

A change is done when contracts and migrations are coherent, protected boundaries remain enforced, targeted tests pass, Ruff and Pyright pass for maintained code, runtime outputs remain outside Git, and user-facing behavior or architecture documentation is updated where needed.
