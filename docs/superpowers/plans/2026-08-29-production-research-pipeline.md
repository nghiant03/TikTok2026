# Production Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the useful Research and Orchestration branch work into the canonical package and deliver a complete, auditable CLI-driven autonomous research pipeline.

**Architecture:** Typed contracts and pure policies form the core. Concrete SQLite, artifact, Git, Docker, evaluator, model, memory, literature, and observability adapters are composed only in `bootstrap.py`; compact LangGraph nodes invoke controller use cases rather than privileged implementations. Runtime state and datasets remain outside Git, and all non-organizer metrics and final bundles remain provisional.

**Tech Stack:** Python 3.11, Pydantic 2, Pydantic Settings, LangGraph, OpenAI-compatible Chat Completions, sqlite3, Typer, MLflow, pytest, Ruff, Pyright.

**Spec:** `docs/superpowers/specs/2026-08-29-production-research-pipeline-design.md`

**Progress at Phase 3 HEAD `9122a9f`:** The checked implementation steps below mark source paths that are present in the canonical tree. Unchecked test-writing, test-execution, and verification steps are not claims that those checks passed. Phase 4 Task 12 remains open, including its final verification and any authorship-history operation.

## Global Constraints

- Preserve exactly four runtime roles: Orchestration, Research, Implementor, and Validator.
- Never modify `baseline/README.md`, `baseline/data.py`, `baseline/evaluate.py`, `baseline/submit.py`, or `baseline/baseline_scores.json`.
- NDCG@10 and Recall@50 are authoritative; local evaluator results are `provisional`.
- Test labels never enter agent context or iterative routing; final test evaluation is single-use.
- Runtime state, worktrees, datasets, artifacts, traces, histories, and submissions remain outside Git.
- Agents never import concrete Git, Docker, SQLite, MLflow, or evaluator implementations.
- Default tests require no network, paid model, Docker, GPU, or KuaiRand data.
- FastAPI, SSE, Uvicorn, and HTTP control-plane code are out of scope.
- Use TDD for every task and run the targeted test immediately after each implementation change.
- Do not create commits unless the user explicitly authorizes commits; reconstructed attribution branches and merges remain blocked until then.

---

### Task 1: Canonical Contracts and Capability Protocols

**Files:**
- Modify: `src/tiktok2026/contracts/models.py`
- Modify: `src/tiktok2026/contracts/__init__.py`
- Create: `src/tiktok2026/contracts/ports.py`
- Modify: `tests/contracts/test_models.py`
- Create: `tests/contracts/test_ports.py`

**Interfaces:**
- Consumes: existing `ExperimentSpec`, `ExecutionResult`, `EvaluationResult`, `ResourceState`.
- Produces: strict models for agent requests/responses, artifacts, manifests, source registration, graph references, finalization, and `Protocol` seams used by every later task.

- [ ] **Step 1: Write failing contract tests**

```python
from pydantic import ValidationError
import pytest
from tiktok2026.contracts import AgentRole, ArtifactRecord, FinalizationRecord


def test_runtime_roles_are_exactly_the_four_authorized_roles() -> None:
    assert {role.value for role in AgentRole} == {
        "orchestration",
        "research",
        "implementor",
        "validator",
    }


def test_registered_artifact_requires_sha256() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord(
            artifact_id="artifact-1",
            run_id="run-1",
            kind="predictions",
            uri="file:///tmp/predictions.csv",
            sha256="bad",
            size_bytes=1,
            producer="controller",
            retention="run",
        )


def test_provisional_finalization_cannot_claim_official() -> None:
    record = FinalizationRecord(
        finalization_id="final-1",
        run_id="run-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        checkpoint_id="checkpoint-1",
        evaluation_id="evaluation-1",
        validity="provisional",
        bundle_artifact_id="bundle-1",
        consumed_test_access=True,
    )
    assert record.validity == "provisional"
```

- [ ] **Step 2: Run tests and verify missing symbols fail**

Run: `uv run pytest tests/contracts/test_models.py tests/contracts/test_ports.py -v`
Expected: FAIL because the new contracts and ports do not exist.

- [x] **Step 3: Implement strict versioned models and protocols**

Implement enums `AgentRole`, `RunPhase`, `ArtifactRetention`, and typed models including `Hypothesis`, `EvidenceItem`, `ResearchRequest`, `ResearchDecision`, `AgentFailure`, `WorktreeAssignment`, `SourceRegistration`, `ExecutionRequest`, `EvaluationRequest`, `ArtifactRecord`, `ResourceReservation`, `LessonRecord`, `FrontierCandidate`, `GraphStateReference`, `FinalizationRecord`, and `ModelUsage`. Add protocols with exact async or sync signatures:

```python
class AgentClient(Protocol):
    async def invoke(self, request: ContractModel) -> ContractModel: ...


class WorktreeManager(Protocol):
    def create(
        self, run_id: str, spec: ExperimentSpec, parent_commit: str
    ) -> WorktreeAssignment: ...
    def register_source(self, assignment: WorktreeAssignment) -> SourceRegistration: ...
    def remove(self, assignment: WorktreeAssignment) -> None: ...


class Executor(Protocol):
    async def execute(self, request: ExecutionRequest) -> ExecutionResult: ...


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...
```

Re-export all public models and protocols from `contracts/__init__.py`.

- [ ] **Step 4: Run contract tests**

Run: `uv run pytest tests/contracts -v`
Expected: PASS.

### Task 2: Settings, Runtime Layout, Migrations, and Persistence

**Files:**
- Create: `src/tiktok2026/config.py`
- Create: `src/tiktok2026/persistence/migrations.py`
- Create: `src/tiktok2026/persistence/repositories.py`
- Create: `src/tiktok2026/persistence/__init__.py`
- Create: `migrations/application/002_pipeline.sql`
- Create: `tests/persistence/test_migrations.py`
- Create: `tests/persistence/test_repositories.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: contract JSON representations and committed budget profiles.
- Produces: `AppSettings`, `RuntimePaths`, `MigrationRunner`, and `ApplicationRepository` with idempotent typed persistence.

- [ ] **Step 1: Write failing settings and migration tests**

```python
def test_runtime_root_must_be_outside_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        RuntimePaths.create(tmp_path, tmp_path / ".runtime")


def test_changed_applied_migration_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "app.sqlite3"
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    migration = migrations / "001_initial.sql"
    migration.write_text("CREATE TABLE sample (id TEXT PRIMARY KEY);", encoding="utf-8")
    MigrationRunner(database, migrations).apply()
    migration.write_text("CREATE TABLE changed (id TEXT PRIMARY KEY);", encoding="utf-8")
    with pytest.raises(MigrationChecksumError):
        MigrationRunner(database, migrations).apply()
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/test_config.py tests/persistence -v`
Expected: FAIL because settings and persistence modules are absent.

- [x] **Step 3: Implement configuration and transactional repositories**

`RuntimePaths.create(repository_root, runtime_root)` resolves paths, rejects containment in the repository, and creates `artifacts`, `worktrees`, `traces`, `exports`, `locks`, `literature`, and `tmp`. `AppSettings` loads profile TOML, optional operator TOML, environment values, and CLI overrides, with per-role `ModelSettings(base_url, model, api_key_env, temperature, max_tokens, timeout_seconds)`.

`MigrationRunner.apply()` hashes each numbered SQL migration, runs unapplied files in order, and rejects checksum changes. `ApplicationRepository` exposes `put_experiment`, `put_audit_event`, `put_artifact`, `put_execution`, `put_evaluation`, `put_failure`, `put_lesson`, `put_finalization`, `get_run_summary`, and `claim_final_test_access`, all using explicit transactions and JSON serialization through contracts.

- [ ] **Step 4: Run persistence tests**

Run: `uv run pytest tests/test_config.py tests/persistence -v`
Expected: PASS.

### Task 3: Artifact, Resource, and Pure Policy Services

**Files:**
- Create: `src/tiktok2026/persistence/artifacts.py`
- Create: `src/tiktok2026/persistence/resources.py`
- Create: `src/tiktok2026/policies/paths.py`
- Create: `src/tiktok2026/policies/resources.py`
- Create: `src/tiktok2026/policies/lifecycle.py`
- Create: `src/tiktok2026/policies/__init__.py`
- Create: `tests/persistence/test_artifacts.py`
- Create: `tests/persistence/test_resources.py`
- Create: `tests/policies/test_paths.py`
- Create: `tests/policies/test_lifecycle.py`

**Interfaces:**
- Consumes: runtime paths, repository transactions, specs, diffs, resources, and metric history.
- Produces: atomic `ArtifactStore`, `ResourceLedger`, protected-path decisions, retry/fidelity/convergence/finalization decisions.

- [ ] **Step 1: Write failing policy tests**

```python
def test_protected_baseline_change_is_rejected() -> None:
    decision = check_changed_paths(("baseline/evaluate.py",), ("src/tiktok2026/experiment",))
    assert not decision.allowed
    assert decision.reason == "protected_path"


def test_converges_after_three_insignificant_results() -> None:
    assert convergence_reason([0.50, 0.501, 0.5015, 0.5018], epsilon=0.002) == "plateau"


def test_final_reserve_cannot_fund_iteration() -> None:
    state = resource_state(remaining_gpu_hours=1.0, reserved_final_gpu_hours=0.75)
    assert not can_reserve_iteration(state, requested_gpu_hours=0.5).allowed
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/policies tests/persistence/test_artifacts.py tests/persistence/test_resources.py -v`
Expected: FAIL because policy and stores are absent.

- [x] **Step 3: Implement stores and pure decisions**

`ArtifactStore.publish_bytes()` writes to runtime `tmp`, flushes, hashes, atomically renames under `artifacts/<run>/<experiment>`, and registers only after publication. `ResourceLedger` uses `BEGIN IMMEDIATE` to reserve, consume, release, and reconcile GPU hours, wall seconds, tokens, and disk.

Implement immutable `PolicyDecision(allowed: bool, reason: str)` and pure checks for protected paths/hashes, approved scope, two-repair limit, fidelity transitions, resource reserve, convergence, single final access, and evaluator validity. Exact values use the spec's `0.002` epsilon and three consecutive experiments.

- [ ] **Step 4: Run policy/store tests**

Run: `uv run pytest tests/policies tests/persistence -v`
Expected: PASS.

### Task 4: Benchmark Manifest, Provisional Evaluator, and Experiment Target

**Files:**
- Create: `src/tiktok2026/benchmark/kuaireand_pure/manifest.py`
- Create: `src/tiktok2026/benchmark/kuaireand_pure/adapter.py`
- Create: `src/tiktok2026/evaluation/metrics.py`
- Create: `src/tiktok2026/evaluation/registry.py`
- Create: `src/tiktok2026/experiment/__init__.py`
- Create: `src/tiktok2026/experiment/config.py`
- Create: `src/tiktok2026/experiment/train.py`
- Create: `tests/benchmark/test_manifest.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/experiment/test_training_contract.py`

**Interfaces:**
- Consumes: external dataset path, benchmark manifest, explicit predictions, and checkpoint paths.
- Produces: verified dataset identity, provisional NDCG@10/Recall@50 results, row-preserving submissions, deterministic training artifacts.

- [ ] **Step 1: Write failing evaluator tests**

```python
def test_provisional_metrics_match_fixture() -> None:
    users = [1, 1, 1, 2, 2]
    labels = [1, 0, 1, 0, 1]
    scores = [0.9, 0.1, 0.8, 0.2, 0.7]
    result = evaluate_rankings(users, labels, scores, k_ndcg=10, k_recall=50)
    assert result["NDCG@10"] == pytest.approx(1.0)
    assert result["Recall@50"] == pytest.approx(1.0)


def test_invalid_predictions_are_rejected() -> None:
    with pytest.raises(PredictionValidationError):
        evaluate_rankings([1], [1], [float("nan")])
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/benchmark tests/evaluation tests/experiment -v`
Expected: FAIL because benchmark, evaluator, and experiment modules are absent.

- [ ] **Step 3: Implement manifest, evaluator, and explicit experiment entry point**

Implement standard per-user binary Recall@50 and DCG/IDCG NDCG@10, validating lengths, finite scores, binary labels, and nonempty groups. `ProvisionalEvaluator.evaluate(EvaluationRequest)` returns both `MetricValue`s and `validity="provisional"`; `EvaluatorRegistry` resolves immutable IDs and rejects official mode without configured organizer code.

Implement a small deterministic CPU experiment target whose CLI accepts `--data-manifest`, `--output-dir`, `--seed`, and fidelity settings, writes predictions and a versioned checkpoint bundle, and never imports `baseline.evaluate` or reads a test split during iterative execution.

- [ ] **Step 4: Run benchmark/evaluator/experiment tests**

Run: `uv run pytest tests/benchmark tests/evaluation tests/experiment -v`
Expected: PASS.

### Task 5: Git Worktrees, Diff Policy, and Source Registration

**Files:**
- Create: `src/tiktok2026/repository/inspector.py`
- Create: `src/tiktok2026/repository/worktrees.py`
- Create: `src/tiktok2026/repository/diffs.py`
- Create: `src/tiktok2026/repository/__init__.py`
- Create: `tests/repository/test_inspector.py`
- Create: `tests/repository/test_worktrees.py`
- Create: `tests/repository/test_diffs.py`

**Interfaces:**
- Consumes: canonical repository, external runtime root, parent commit, approved spec, changed files.
- Produces: bounded observations, sibling `WorktreeAssignment`, normalized patch artifact, validated `SourceRegistration`.

- [ ] **Step 1: Write failing temporary-repository tests**

```python
def test_worktree_is_created_under_runtime_root(git_repo: Path, tmp_path: Path) -> None:
    manager = GitWorktreeManager(git_repo, tmp_path / "runtime")
    assignment = manager.create("run-1", spec("exp-1"), git_head(git_repo))
    assert assignment.path.startswith(str(tmp_path / "runtime" / "worktrees"))


def test_diff_rejects_out_of_scope_file() -> None:
    result = validate_diff(("README.md",), ("src/tiktok2026/experiment",))
    assert not result.allowed
    assert result.reason == "outside_implementation_scope"
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/repository -v`
Expected: FAIL because repository services are absent.

- [x] **Step 3: Implement narrow Git services**

Use non-interactive `git` subprocess calls with argument arrays and captured output. Worktrees are siblings under runtime root, branch names derive from validated IDs, and assignments record parent commit. Normalize `git diff --binary --no-ext-diff`, list changed paths, enforce scope/protected policy, hash the patch, and detect duplicate signatures. `register_source()` verifies clean staged content and controller-created commit identity; no agent receives this method.

- [ ] **Step 4: Run repository tests**

Run: `uv run pytest tests/repository -v`
Expected: PASS.

### Task 6: Docker Execution, Failure Classification, and Opt-In Integration Test

**Files:**
- Create: `src/tiktok2026/execution/docker.py`
- Create: `src/tiktok2026/execution/failures.py`
- Create: `src/tiktok2026/execution/__init__.py`
- Create: `tests/execution/test_docker_request.py`
- Create: `tests/execution/test_failures.py`
- Create: `tests/integration/test_docker_execution.py`

**Interfaces:**
- Consumes: typed execution request, approved image, read-only source/data mounts, writable artifact mount, limits.
- Produces: typed `ExecutionResult`, bounded log artifacts, deterministic `FailureKind`.

- [ ] **Step 1: Write failing command and classifier tests**

```python
def test_docker_command_disables_network_and_mounts_data_read_only(tmp_path: Path) -> None:
    command = build_docker_command(request(tmp_path))
    assert "--network=none" in command
    assert any(value.endswith(":ro") and "dataset" in value for value in command)


def test_cuda_oom_evidence_is_classified() -> None:
    assert classify_failure(137, "CUDA out of memory", timed_out=False) == FailureKind.CUDA_OOM
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/execution -v`
Expected: FAIL because executor modules are absent.

- [x] **Step 3: Implement constrained executor**

Build a Docker CLI argument list containing `--network=none`, `--read-only`, explicit mounts, memory/CPU/GPU limits, working directory, environment allowlist, and image identity. Run asynchronously with timeout, process-group termination, bounded output publication, and resource usage. Classify known evidence without converting valid non-improvement into an execution failure.

Mark the live test `@pytest.mark.integration`; skip unless `TIKTOK2026_RUN_DOCKER_TESTS=1` and an image is configured.

- [ ] **Step 4: Run execution tests**

Run: `uv run pytest tests/execution -v`
Expected: PASS; live integration test remains skipped by default.

### Task 7: Research Agent Selective Port

**Files:**
- Create: `src/tiktok2026/agents/common/client.py`
- Create: `src/tiktok2026/agents/common/structured.py`
- Create: `src/tiktok2026/agents/research/context.py`
- Create: `src/tiktok2026/agents/research/agent.py`
- Modify: `src/tiktok2026/agents/research/prompt.md`
- Create: `src/tiktok2026/memory/retrieval.py`
- Create: `src/tiktok2026/literature/retrieval.py`
- Create: `tests/agents/test_model_client.py`
- Create: `tests/agents/research/test_context.py`
- Create: `tests/agents/research/test_agent.py`

**Interfaces:**
- Consumes: canonical `ResearchRequest`, injected read-only repository/data/memory/literature capabilities, per-role model settings.
- Produces: `ResearchDecision | AgentFailure` with one bounded repair and source-branch provenance in module documentation.

- [ ] **Step 1: Write failing provider and research tests**

```python
async def test_openai_compatible_client_uses_configured_endpoint(fake_transport) -> None:
    client = OpenAICompatibleClient(settings("https://example.test/v1", "gpt-4.1"), fake_transport)
    await client.complete("system", "user", response_schema={"type": "object"})
    assert fake_transport.last_url == "https://example.test/v1/chat/completions"
    assert fake_transport.last_json["model"] == "gpt-4.1"


async def test_research_repairs_invalid_response_once(scripted_model, capabilities) -> None:
    scripted_model.outputs = ["{}", valid_research_decision_json()]
    result = await ResearchAgent(scripted_model, capabilities).invoke(request())
    assert isinstance(result, ResearchDecision)
    assert scripted_model.calls == 2
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/agents/test_model_client.py tests/agents/research -v`
Expected: FAIL because the integrated client and Research Agent are absent.

- [ ] **Step 3: Port compliant Research behavior**

Port bounded concurrent context assembly, evidence IDs/provenance, lineage, lessons, literature metadata, leakage checks, response parsing, and one repair from source commit `8c776fd11c612211375d0712490a01642abb5187`. Remove standalone contract loading, copied roots, fixed DeepSeek defaults, GAUC/nDCG@5, and direct network assumptions. Context rejects test labels, full datasets, unlicensed full text, external assets, and unsupported numerical claims.

The common client reads the configured API key environment variable at call time, never serializes it, posts standard Chat Completions requests, captures usage, and validates JSON through the requested Pydantic model.

- [ ] **Step 4: Run Research and client tests**

Run: `uv run pytest tests/agents/test_model_client.py tests/agents/research -v`
Expected: PASS.

### Task 8: Orchestration, Implementor, and Validator Agents

**Files:**
- Create: `src/tiktok2026/agents/orchestration/agent.py`
- Create: `src/tiktok2026/agents/implementor/agent.py`
- Create: `src/tiktok2026/agents/validator/agent.py`
- Modify: all three owning `prompt.md` files
- Create: `tests/agents/orchestration/test_agent.py`
- Create: `tests/agents/implementor/test_agent.py`
- Create: `tests/agents/validator/test_agent.py`
- Create: `tests/architecture/test_agent_capabilities.py`

**Interfaces:**
- Consumes: common structured client, canonical role requests, role-specific least-privilege capabilities.
- Produces: `OrchestrationDecision`, `ImplementationResult`, `ValidationReport`, or typed failure.

- [ ] **Step 1: Write failing capability and repair tests**

```python
def test_only_four_agent_packages_define_runtime_agents() -> None:
    assert discover_agent_roles() == {
        AgentRole.ORCHESTRATION,
        AgentRole.RESEARCH,
        AgentRole.IMPLEMENTOR,
        AgentRole.VALIDATOR,
    }


async def test_implementor_cannot_write_protected_path(scripted_model, scoped_repo) -> None:
    scripted_model.outputs = [implementation_for("baseline/data.py")]
    result = await ImplementorAgent(scripted_model, scoped_repo).invoke(implement_request())
    assert isinstance(result, AgentFailure)
    assert not scoped_repo.writes
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/agents/orchestration tests/agents/implementor tests/agents/validator tests/architecture/test_agent_capabilities.py -v`
Expected: FAIL because agents are absent.

- [x] **Step 3: Implement role-specific typed agents**

Port orchestration iteration/recovery/finalization ideas from source commit `448f7e39e70d5745a784a72f7305bd5ad8df357c`, excluding its recipe queue, direct subprocesses, test scoring, and in-repository sandboxes. Orchestration returns only allowed `DecisionAction` values and deterministic policy validates them.

Implementor exposes assigned read/search/write/diff/check methods only and rejects paths before writing. Validator exposes read-only proposal/diff/result evidence and cannot repair, execute, or evaluate. All use the shared one-repair structured response behavior.

- [ ] **Step 4: Run all agent tests**

Run: `uv run pytest tests/agents tests/architecture/test_agent_capabilities.py -v`
Expected: PASS.

### Task 9: Frontier, Duplicate Detection, Lessons, and Observability

**Files:**
- Create: `src/tiktok2026/search/signatures.py`
- Create: `src/tiktok2026/search/frontier.py`
- Create: `src/tiktok2026/memory/lessons.py`
- Create: `src/tiktok2026/observability/traces.py`
- Create: `src/tiktok2026/observability/mlflow.py`
- Create: `src/tiktok2026/observability/exports.py`
- Create: `tests/search/test_frontier.py`
- Create: `tests/memory/test_lessons.py`
- Create: `tests/observability/test_exports.py`

**Interfaces:**
- Consumes: persisted specs/results/failures, artifact store, bounded context, model usage.
- Produces: normalized signatures, four-slot frontier, evidence-backed lessons, restricted traces, MLflow references, deterministic Markdown/JSONL exports.

- [ ] **Step 1: Write failing frontier/export tests**

```python
def test_frontier_keeps_champion_two_diverse_and_diagnostic() -> None:
    selected = select_frontier(candidates(), limit=4)
    assert len(selected) == 4
    assert [item.slot for item in selected] == [
        "champion",
        "alternative",
        "alternative",
        "diagnostic",
    ]


def test_export_is_deterministic(repository, tmp_path: Path) -> None:
    first = export_run("run-1", repository, tmp_path / "first")
    second = export_run("run-1", repository, tmp_path / "second")
    assert first.jsonl.read_bytes() == second.jsonl.read_bytes()
    assert first.markdown.read_bytes() == second.markdown.read_bytes()
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/search tests/memory tests/observability -v`
Expected: FAIL because services are absent.

- [ ] **Step 3: Implement bounded scientific memory and exports**

Normalize specs and patches with stable JSON and SHA-256; block exact duplicates. Rank candidates by persisted validation score, diversity tags, fidelity, and resource cost. Lessons require supporting experiment IDs and evidence strength. Trace files omit secret values and are restricted runtime artifacts. MLflow adapter records telemetry/references only. Exports sort by canonical timestamps and IDs and include hypotheses, source/patch refs, metrics/validity, failures/recovery, interventions, tokens, GPU hours, lineage, final selection, and hashes.

- [ ] **Step 4: Run search/memory/observability tests**

Run: `uv run pytest tests/search tests/memory tests/observability -v`
Expected: PASS.

### Task 10: Controller Use Cases and Production LangGraph

**Files:**
- Create: `src/tiktok2026/controller.py`
- Create: `src/tiktok2026/graph/state.py`
- Create: `src/tiktok2026/graph/routes.py`
- Create: `src/tiktok2026/graph/nodes.py`
- Create: `src/tiktok2026/graph/build.py`
- Create: `tests/graph/test_routes.py`
- Create: `tests/graph/test_pipeline.py`
- Create: `tests/architecture/test_dependency_direction.py`

**Interfaces:**
- Consumes: injected protocols and services from Tasks 1-9.
- Produces: controller operations and compact production graph covering proposal, implementation, execution, repair, persistence, convergence, and provisional finalization.

- [ ] **Step 1: Write failing routing and state tests**

```python
def test_graph_state_contains_references_not_artifacts() -> None:
    assert set(ProductionState.__annotations__) == {
        "run_id",
        "phase",
        "current_experiment_id",
        "current_hypothesis_id",
        "active_worktree_id",
        "latest_validation_report_id",
        "latest_execution_result_id",
        "latest_evaluation_result_id",
        "orchestration_decision_id",
        "repair_attempts",
        "fidelity",
        "pending_route",
        "terminal_reason",
        "state_version",
    }


def test_repairable_failure_routes_to_repair_until_limit() -> None:
    assert route_after_failure(state(repair_attempts=1), repairable_failure()) == "repair"
    assert route_after_failure(state(repair_attempts=2), repairable_failure()) == "persist_failure"
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/graph tests/architecture/test_dependency_direction.py -v`
Expected: FAIL because production graph is absent.

- [ ] **Step 3: Implement controller-owned transitions and graph**

Implement use cases for inspect, decide, research, proposal gate, create worktree, implement, diff gate, validate, source registration, preflight, execute, classify, evaluate, interpret, persist, update frontier/resources, repair, stop, final test, and export. Graph nodes call only those use cases. Route functions accept typed enums and records and return a finite set of node names. Persist every transition before returning its compact reference.

- [ ] **Step 4: Run graph and architecture tests**

Run: `uv run pytest tests/graph tests/architecture/test_dependency_direction.py -v`
Expected: PASS.

### Task 11: Bootstrap, CLI, Recovery, and End-to-End Synthetic Pipeline

**Files:**
- Create: `src/tiktok2026/bootstrap.py`
- Modify: `src/tiktok2026/cli.py`
- Modify: `src/tiktok2026/testing/lifecycle.py`
- Modify: `src/tiktok2026/testing/synthetic.py`
- Create: `tests/cli/test_commands.py`
- Modify: `tests/integration/test_synthetic_lifecycle.py`
- Create: `tests/integration/test_recovery.py`

**Interfaces:**
- Consumes: settings and all concrete adapters.
- Produces: composition root, production/synthetic runners, CLI initialize/run/resume/inspect/finalize/export/diagnostics commands.

- [ ] **Step 1: Write failing CLI and lifecycle tests**

```python
def test_runtime_init_creates_external_layout(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(app, ["runtime-init", "--runtime-root", str(tmp_path / "runtime")])
    assert result.exit_code == 0
    assert (tmp_path / "runtime" / "application.sqlite3").exists()


async def test_two_cycles_persist_audit_and_provisional_bundle(tmp_path: Path) -> None:
    result = await run_synthetic_lifecycle(iterations=2, runtime_root=tmp_path / "runtime")
    assert len(result.experiment_ids) == 2
    assert result.finalization.validity == "provisional"
    assert result.exports.jsonl.exists()
```

- [ ] **Step 2: Verify failures**

Run: `uv run pytest tests/cli tests/integration/test_synthetic_lifecycle.py tests/integration/test_recovery.py -v`
Expected: FAIL because CLI and full synthetic composition are incomplete.

- [ ] **Step 3: Implement composition and operational commands**

`bootstrap.py` is the sole place that constructs SQLite, artifact, resource, Git, Docker, evaluator, model, memory, literature, trace, MLflow, controller, and graph implementations. Synthetic mode substitutes scripted agents, temporary source manager, fake executor, and fixture evaluator while retaining policies, persistence, resources, artifacts, graph, and exports.

CLI commands are `runtime-init`, `migrate`, `verify-manifests`, `synthetic-run`, `run`, `resume`, `inspect`, `finalize`, `export`, and `diagnostics`. They return nonzero status on typed failures and create audit events for human interventions. Recovery reconciles stale locks/reservations and resumes only when source/artifact identities agree.

- [ ] **Step 4: Run CLI and integration tests**

Run: `uv run pytest tests/cli tests/integration -v`
Expected: PASS with live environment tests skipped.

### Task 12: Branch Provenance, Documentation, Dependencies, and Full Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Create: `docs/INTEGRATION_PROVENANCE.md`
- Modify: `.gitignore`
- Modify: `tests/architecture/test_protected_baseline.py`
- Create: `tests/architecture/test_runtime_outputs.py`

**Interfaces:**
- Consumes: implemented production behavior and source branch metadata.
- Produces: accurate operator documentation, dependency set without HTTP stack, integration provenance, and final verification evidence.

- [ ] **Step 1: Write failing architecture assertions**

```python
def test_http_stack_is_not_a_runtime_dependency() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {requirement.split("[")[0].split(">=")[0] for requirement in project["dependencies"]}
    assert not {"fastapi", "uvicorn"} & names


def test_source_branch_payloads_are_not_integrated() -> None:
    forbidden = ("research_agent", "prototype", "submission_final.csv", "run_log.jsonl")
    tracked = subprocess.run(["git", "ls-files"], text=True, capture_output=True, check=True).stdout
    assert all(name not in tracked for name in forbidden)
```

- [ ] **Step 2: Verify documentation/dependency tests initially fail**

Run: `uv run pytest tests/architecture -v`
Expected: FAIL until dependencies and architecture status are updated.

- [ ] **Step 3: Update project metadata and documentation**

Remove FastAPI and Uvicorn dependencies and every API/SSE claim. Document CLI commands, external runtime/data setup, per-role OpenAI-compatible model settings, provisional evaluator semantics, opt-in integration checks, recovery, and final bundle contents. `docs/INTEGRATION_PROVENANCE.md` records source commits `8c776fd11c612211375d0712490a01642abb5187` and `448f7e39e70d5745a784a72f7305bd5ad8df357c`, original authors, selectively ported behavior, and excluded payloads. Expand ignores for runtime/export/submission outputs without ignoring source fixtures.

If commit authorization is later provided, reconstruct and merge the contributor branches exactly as specified in the design; otherwise leave all implementation changes uncommitted on the working branch and report that attribution merges are blocked by the no-commit rule.

- [ ] **Step 4: Run targeted and full verification**

Run:

```bash
uv sync --dev
uv run pytest tests/contracts tests/policies tests/persistence tests/benchmark tests/evaluation tests/repository tests/execution tests/agents tests/search tests/memory tests/observability tests/graph tests/cli tests/integration tests/architecture -v
uv run pytest
uv run ruff check .
uv run pyright
git diff --check
git status --short
```

Expected: all tests pass, Ruff and Pyright report no errors, protected baseline hashes are unchanged, and status contains no runtime output or dataset files added by this work.
