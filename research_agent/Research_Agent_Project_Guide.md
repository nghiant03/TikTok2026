# Research Agent Project Guide


This guide explains what the Research Agent currently implements, which code files are responsible for each layer, and how to verify the system step by step with pytest, smoke tests, and a real DeepSeek call.

**Intended audience:** People who can use agents but are new to agent engineering  
**Last updated:** August 28, 2026

## 1. The Project in One Sentence

The Research Agent is the system's **researcher**. It reads authorized project facts, safe dataset summaries, experiment history, and academic papers. It then proposes the next experiment, requests missing evidence, or interprets an existing experiment result. It does not write training code or run model training itself.

Within the three-agent system:

- **Orchestration Agent:** Decides whether the current iteration should research, implement, repair, or stop, and creates a `ResearchRequest`.
- **Research Agent:** Produces an evidence-based `Hypothesis` and `ExperimentSpec`, or returns an `EvidenceRequest`.
- **Implement Agent:** Writes or modifies experiment code according to an approved `ExperimentSpec`.
- **Validator:** A deterministic checking node outside the three agents. It does not rely on a language model's free-form judgment.

The main internal flow of the Research Agent is:

```text
ResearchRequest -> ResearchContext -> Prompt -> DeepSeek
       -> Pydantic contract validation -> Policy business gates
       -> ResearchDecision (success) or failure after one repair attempt
```

## 2. Main Implemented Capabilities

- **Read real project code:** Extracts traceable, length-limited repository evidence from authorized files in `TikTok2026-main`.
- **Read safe summaries of real data:** Reads only authorized training data and reports summaries such as row count, user count, video count, date range, and label ratio. It does not send the full dataset to the model.
- **Query research memory:** Reads related historical experiments from local JSONL, follows experiment lineage through `parent_experiment_id`, and retrieves reusable `ResearchLesson` records.
- **Retrieve research literature:** Reads local PDFs and can retrieve public literature evidence from Semantic Scholar, arXiv, Crossref, and approved web pages.
- **Call a real research model:** Uses the DeepSeek client to generate a structured `ResearchDecision` and records token usage.
- **Apply quality gates:** Rejects incorrect labels, incorrect metrics, unauthorized external training data, prohibited pretrained weights, hidden-test access, leakage features, unauthorized paths, and unsupported numerical claims.
- **Allow one bounded repair:** If the first model response is invalid, the original response and validation errors are sent back to the model. If the second response is still invalid, the agent returns `ResearchAgentFailure`.
- **Verify the Starter Kit:** Uses SHA-256 hashes to confirm that `data.py`, `evaluate.py`, and `baseline_scores.json` match the pinned evaluation protocol.

> **Boundary:** The current Research Agent can propose how an experiment should be trained, but it does not execute FM training or generate an `EvaluationResult` by itself. Training, container execution, and protected evaluation belong to the merged system's controller, execution node, and evaluator.

## 3. Contracts: Standard Forms Shared Between Agents

A **contract** is not a legal document. It is a standard data format that agents must follow when exchanging information.

A simple analogy:

- DeepSeek is the person filling in a form.
- A Pydantic contract is the form template.
- Pydantic validation checks whether fields are missing, have the wrong type, or contain forbidden extras.
- `policy.py` then checks whether the content violates project rules.

Research-specific contracts are mainly defined in `src/research_agent/contracts.py`. Shared experiment contracts are loaded read-only through `shared_contracts.py` from `external/TikTok2026-main/src/tiktok2026/contracts/models.py`.

### 3.1 Important Input and Context Contracts

#### `ResearchContractModel` - Research Contract Base Model

**Plain meaning:** The common foundation for all Research Agent contracts. It rejects unknown fields and prevents contract objects from being modified casually.

**Code:** `src/research_agent/contracts.py`  
**Analogy:** The paper format and common completion rules used by every form.

#### `BenchmarkContract` - Benchmark Rules Contract

**Plain meaning:** Fixes the dataset, positive label, metrics, date split, baseline scores, and prohibited actions. The model cannot rewrite these rules.

**Code:** `src/research_agent/contracts.py`  
**Current example:** KuaiRand-Pure, `long_view`, GAUC, and nDCG@5.

#### `ResearchRequest` - Research Task Request

**Plain meaning:** The task form sent by the Orchestration Agent. It states the objective, task type, parent experiment, resource state, baseline status, and permitted implementation scope.

**Code:** `src/research_agent/contracts.py`  
**Example:** "Use the available evidence to propose the next experiment, within these resource limits."

#### `EvidenceItem` - One Evidence Record

**Plain meaning:** A small, traceable fact. Each item has an evidence ID, evidence type, summary, and source. It also records whether it contains test labels and whether access is authorized.

**Code:** `src/research_agent/contracts.py`  
**Examples:** A code summary identified by `repository://...`, or a training-data statistic identified by `dataset://...`.

#### `ResearchContext` - Research Context Package

**Plain meaning:** The controlled information package the model is allowed to see during the current run. It contains the `ResearchRequest`, `EvidenceItem` records, historical experiments, experiment lineage, and research lessons, with a maximum evidence limit.

**Code:** The contract is in `src/research_agent/contracts.py`; construction logic is in `src/research_agent/context.py`.  
**Analogy:** A controlled briefing pack given to a researcher before a meeting.

#### `ExperimentHistoryItem` - Historical Experiment Record

**Plain meaning:** Records what was tried before, the direction of the result, the parent experiment, and evidence references. This helps prevent repeated proposals.

**Code:** Defined in `src/research_agent/contracts.py`; read by `src/research_agent/adapters.py`.  
**Example:** `exp-002` modified `exp-001`, but its result was worse.

#### `ResearchLesson` - Reusable Research Finding

**Plain meaning:** A reusable conclusion distilled from one or more experiments. It includes evidence strength, scope, and supporting experiments.

**Code:** Defined in `src/research_agent/contracts.py`; read by `src/research_agent/adapters.py`.  
**Example:** "This type of target encoding must use OOF processing; otherwise, it leaks the label."

#### `ResearchMemoryQueryResult` - Research Memory Query Result

**Plain meaning:** The complete result of one history query: related experiments, full lineage, and relevant lessons.

**Code:** `src/research_agent/contracts.py`  
**Example:** The structured source behind `history_count`, `lineage_count`, and `lesson_count`.

### 3.2 Important Output Contracts

#### `Hypothesis` - Research Hypothesis

**Plain meaning:** States what should be tested, the possible mechanism, why the test is worthwhile, and which evidence supports it.

**Code:** `src/research_agent/contracts.py`  
**Example:** "The official FM baseline can be reproduced on the fixed split within the protocol tolerance."

#### `ExperimentSpec` - Experiment Specification

**Plain meaning:** A precise technical task that the Implement Agent can build. It contains the experiment ID, hypothesis, mechanism, implementation scope, success and failure criteria, resource estimate, and leakage risks.

**Code:** `external/TikTok2026-main/src/tiktok2026/contracts/models.py`, loaded by `shared_contracts.py`.  
**Important:** It is not a training result. It precisely describes how the next experiment should be performed.

#### `ExperimentProposal` - Experiment Proposal

**Plain meaning:** Packages an `ExperimentSpec`, a `Hypothesis`, and compliance declarations. These declarations cover test labels, external data, pretrained weights, target-derived features, and similar risks.

**Code:** `src/research_agent/contracts.py`  
**Example:** If the baseline is missing, the agent may only propose `baseline_reproduction` or request evidence.

#### `EvidenceRequest` - Request for Additional Evidence

**Plain meaning:** Clearly states what evidence is missing, why it is needed, and whether the missing evidence blocks further work.

**Code:** `src/research_agent/contracts.py`  
**Example:** If the evaluation protocol is unconfirmed, request the official data split and evaluator.

#### `ResearchInterpretation` - Interpretation of Results

**Plain meaning:** After receiving an `ExecutionResult` or `EvaluationResult`, it separates objective facts from research judgment and recommends the next step.

**Code:** `src/research_agent/contracts.py`  
**Example:** A metric drop may mean the idea is ineffective, or it may only mean execution failed. These cases must be separated.

#### `ResearchDecision` - Research Decision

**Plain meaning:** The outermost response from the Research Agent. Each response must contain exactly one of the following: an experiment proposal, an evidence request, or a result interpretation.

**Code:** `src/research_agent/contracts.py`  
**Example:** When `kind=experiment_proposal`, the other two payloads must be empty.

#### `ResearchResponse` - Compatibility Name for Research Response

**Plain meaning:** A compatibility alias for `ResearchDecision`, allowing older graphs and integration drafts to continue using the previous name.

**Code:** `src/research_agent/contracts.py`  
**Definition:** `ResearchResponse = ResearchDecision`.

#### `ResearchAgentFailure` - Typed Research Agent Failure

**Plain meaning:** Converts a model, schema, or policy error into structured data instead of allowing the program to crash without explanation.

**Code:** `src/research_agent/contracts.py`  
**Example:** `kind=schema` and `repair_attempts=1`.

### 3.3 Shared Runtime and Evaluation Contracts

#### `ResourceState` - Available Resource State

**Plain meaning:** Tells the Research Agent how much GPU time, wall-clock time, token budget, and disk space remain.

**Code:** `external/TikTok2026-main/src/tiktok2026/contracts/models.py`, loaded by `shared_contracts.py`.  
**Purpose:** A proposal must respect the remaining budget.

#### `ExecutionResult` - Program Execution Result

**Plain meaning:** Records whether the training program exited successfully, how long it ran, GPU usage, and the failure type. It is not the model-quality score.

**Code:** `external/TikTok2026-main/src/tiktok2026/contracts/models.py`, loaded by `shared_contracts.py`.  
**Example:** If `exit_code` is not zero, the system should interpret an execution failure.

#### `ResearchEvaluationResult` - Research Evaluation Result

**Plain meaning:** Records experiment metrics, evaluator information, prediction-file hashes, and result validity so that the Research Agent can interpret the outcome.

**Code:** `src/research_agent/contracts.py`  
**Example:** GAUC and nDCG@5 form an ordered metric pair.

## 4. LangGraph: Connecting Steps into a Controlled Process

LangGraph can be understood as a **flowchart executor**. It does not provide research knowledge and does not store history. It decides which node runs next and carries `ResearchGraphState` through the workflow.

```text
START
  -> validate_request
  -> build_context
  -> call_model
  -> validate_response
       -> END (accepted)
       -> repair -> validate_response -> END (accepted or failed)
```

| Node | Plain meaning | Responsibility | Code |
|---|---|---|---|
| `validate_request` | Validate the research request | Validates the input as a `ResearchRequest`; invalid input returns a typed failure immediately. | `graph.py` |
| `build_context` | Build the research context | Reads repository, data, memory, and literature evidence in parallel and builds the prompt. | `graph.py`, `context.py` |
| `call_model` | Call the model | Sends the prompt to a scripted test client or the real DeepSeek client. | `graph.py`, `model.py` |
| `validate_response` | Validate the model response | Runs Pydantic structural validation, followed by the business gates in `policy.py`. | `graph.py`, `agent.py`, `policy.py` |
| `repair` | Repair once | Sends the original response and validation errors back to the model. | `graph.py` |
| `terminal_failure` | End with failure | Returns `ResearchAgentFailure` if the repaired response is still invalid. | `graph.py` |

Only one repair is allowed to prevent infinite retries and keep cost, time, and behavior predictable.

## 5. Code Files by Architectural Layer

| File | Layer | Main responsibility |
|---|---|---|
| `contracts.py` | Contract layer | Defines Research Agent inputs, context, decisions, hypotheses, proposals, interpretations, and failure formats. |
| `shared_contracts.py` | Shared-contract bridge | Loads `ExperimentSpec`, `ExecutionResult`, `ResourceState`, and other contracts read-only from `TikTok2026-main`. |
| `capabilities.py` | Capability interface layer | Defines the methods required from the four read-only capabilities: repository, data, memory, and literature. |
| `adapters.py` | Real readers | Implements filesystem, KuaiRand summary, JSONL, PDF, Semantic Scholar, arXiv, Crossref, and web readers. |
| `context.py` | Context layer | Calls the four capability groups in parallel and assembles a size-limited `ResearchContext`. |
| `prompt.md` | Human-readable role instructions | Describes the Research Agent's responsibilities and boundaries. |
| `prompt.py` | Prompt construction | Combines system rules, `ResearchContext`, and the `ResearchDecision` JSON Schema into model input. |
| `model.py` | Model client | Defines the common model interface, real DeepSeek client, token recording, and scripted test client. |
| `policy.py` | Deterministic quality gates | Checks evidence references, baseline ordering, implementation scope, historical duplication, leakage risk, and numerical support. |
| `agent.py` | Non-graph execution entry | Parses and validates model responses and provides a one-repair flow that does not require LangGraph. |
| `graph.py` | LangGraph subgraph | Defines nodes, routes, the single repair attempt, and final success or failure states. |
| `protocol.py` | Evaluation protocol verification | Uses SHA-256 to check whether protected Starter Kit files have changed. |
| `runtime.py` | Runtime assembly | Reads paths and model settings from environment variables and assembles real readers and the DeepSeek client. |
| `phase2_smoke.py` | Real-environment smoke entry point | Runs real read-only context checks and optionally calls DeepSeek with `--call-model`. |
| `testing.py` | Simulated capabilities | Provides predictable repository, data, history, and literature evidence for pytest. |
| `tests/` | Automated tests | Covers adapters, the agent, context, contracts, LangGraph, the DeepSeek client, and the evaluation protocol. |

## 6. How the Four Read-Only Capabilities Work

`ResearchCapabilities` is a toolbox. `context.py` opens four drawers at the same time:

| Interface | Plain meaning | Current implementation | Output |
|---|---|---|---|
| `RepositoryEvidenceReader` | Repository evidence reader | `FileSystemRepositoryEvidenceReader` | Summaries of real code files |
| `DataEvidenceReader` | Dataset evidence reader | `KuaiRandPureDataEvidenceReader` | Safe training-data statistics and data boundaries |
| `ResearchMemoryReader` | Research memory reader | `JsonlExperimentHistoryReader` | Historical experiments, lineage, and `ResearchLesson` records |
| `LiteratureEvidenceReader` | Literature evidence reader | Local PDF, Semantic Scholar, arXiv, Crossref, and web readers | Paper titles, abstracts, and sources |

The reader outputs are normalized into `EvidenceItem` records and placed in `ResearchContext`. Every item must contain a `source_ref`, so later users can trace where each statement came from.

## 7. Three Verification Layers and What Each Command Proves

Run the commands from the project directory in PowerShell:

```powershell
cd "...\research_agent"
```

### 7.1 pytest: Verify Code Logic

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

This command uses simulated evidence, small temporary CSV files, simulated history, simulated `ExecutionResult` and `EvaluationResult` objects, and prewritten model responses. It quickly checks 74 predefined scenarios.

- `test_contracts.py`: Contract fields, label and metric rules, prohibited pretrained weights, and test-set access.
- `test_context.py`: Parallel evidence collection, evidence limits, test-label exclusion, and protocol evidence completeness.
- `test_agent.py`: Valid proposals, one repair attempt, scope rejection, baseline ordering, and evidence gates.
- `test_graph.py`: LangGraph routing, two-round research flow, result interpretation, and typed failures.
- `test_adapters.py`: Logic of the real readers; network responses are mainly simulated.
- `test_model_deepseek.py`: DeepSeek request format, API-key protection, and error handling without making a real paid call.
- `test_official_protocol.py`: Starter Kit hashes and GAUC/nDCG@5 evaluation rules.

If the output says `74 passed`, the code behaves correctly in the predefined scenarios. This result alone does not prove that real paths, real PDFs, live network retrieval, and the real DeepSeek service are available.

### 7.2 Offline Smoke Test: Verify Real Local Resources

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke --offline-literature
```

This command does not call the paid model or access online literature services. It verifies:

- `repository://...`: Real project code was read successfully.
- `dataset://...`: A safe summary of the real KuaiRand training data was produced.
- `literature-local://...`: Local PDFs were found and text was extracted.
- `evaluation-protocol:...`: Starter Kit hashes and the evaluation protocol passed verification.
- `history_count`, `lineage_count`, and `lesson_count`: The history query ran. A value of zero means no records currently exist; it does not mean the program failed.

### 7.3 Online Smoke Test: Verify Public Literature Retrieval

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke
```

In addition to the offline checks, this command uses Semantic Scholar, arXiv, Crossref, and any configured approved web source.

- `literature-arxiv:...`: arXiv returned a real paper title and abstract.
- `literature-online:...` with a DOI `source_ref`: Crossref returned publication metadata.
- `literature-semantic-scholar-error:...`: A request was sent but failed at the HTTP level. This proves graceful degradation worked, but it does not prove successful retrieval.

If anonymous Semantic Scholar access is restricted, configure `SEMANTIC_SCHOLAR_API_KEY`.

### 7.4 Real DeepSeek Call: Verify Research Proposal Generation

```powershell
.\.venv\Scripts\python.exe -m research_agent.phase2_smoke --call-model --offline-literature
```

This command requires `RESEARCH_MODEL_API_KEY`. It may incur API charges, but it does not train the FM model or modify project source code.

- `kind=experiment_proposal`, `evidence_request`, or `result_interpretation`: Shows which research decision the model selected.
- `experiment_proposal.spec`: The experiment specification generated by the model.
- `hypothesis`: The structured research hypothesis generated by the model.
- No `failure` and a normal command exit: The response passed both Pydantic and policy gates.
- `model_usage`: The real API's `prompt_tokens`, `completion_tokens`, and `total_tokens`.

> **Important distinction:** The model generates an experiment plan, not a training result. A real `EvaluationResult` exists only after an execution node runs the training and an evaluator computes the metrics.

## 8. Current Verification Status

| Check | Status | Evidence or explanation |
|---|---|---|
| pytest | Passed | `74 passed`; contracts, graph routing, policy gates, and adapter logic are working. |
| Offline smoke test | Passed | Collected 16 evidence items; read real code, a summary of 1,141,112 training rows, and three local PDFs. |
| arXiv | Passed | A live request returned paper titles and abstracts. |
| Crossref | Passed | A live request returned DOI and publication metadata. |
| Semantic Scholar | Partially passed | The observed request returned an `HTTPError`; the error was recorded and the main flow did not crash. |
| DeepSeek | Passed | Generated a `baseline_reproduction` proposal and recorded 15,550 tokens. |
| Real training and evaluation | Not run in this project | Requires the Implement Agent, execution node, and protected evaluator. |

These figures are a snapshot from the recorded test run. Counts may change as literature results, test cases, and history records change.

## 9. Integration with the Other Agents

1. The Orchestration Agent creates a `ResearchRequest` and injects resource state, baseline status, evaluation protocol status, and permitted implementation scope.
2. The Orchestration Agent calls `run_research_graph(request, model_client, capabilities)`.
3. The Research Agent returns a `ResearchDecision`: an experiment proposal, evidence request, or result interpretation.
4. The deterministic Validator checks the proposal. If approved, the Implement Agent implements the `ExperimentSpec`.
5. The controller calls the execution node to run training and then calls the protected evaluator to produce an `EvaluationResult`.
6. The new `ExecutionResult` and `EvaluationResult` return to the Research Agent for interpretation and the next research iteration.

**Interface boundary:** The Research Agent's LangGraph is one subgraph of the complete system LangGraph. It manages research decisions, not the entire training lifecycle.

## 10. Commonly Confused Concepts

### What Is Context?

The information package the model is allowed to see during the current run. It is not the complete project or the complete dataset.

### What Is Evidence?

A factual summary with a unique ID and source. The model must cite evidence when making important claims.

### What Is an Experiment?

An executable and comparable model plan. The Research Agent designs the experiment; other system components execute it.

### What Is the Difference Between `ExecutionResult` and `EvaluationResult`?

`ExecutionResult` answers, "Did the program run successfully?"  
`EvaluationResult` answers, "What model-quality metrics did the run produce?"

### Why Is It Called a Smoke Test?

The term comes from checking whether equipment emits smoke when first powered on. A smoke test quickly checks whether real connections work end to end; it does not attempt to cover every detail.

### Why Does pytest Use Prewritten Model Responses?

Tests must run repeatedly in a fast, stable, and free way. A real API can cost money, time out, or return different content on different runs.

### Does the Research Agent Train Models?

No. It outputs training plans. The merged Implement, execution, and evaluation components perform training and scoring.

## 11. Current Evaluation Protocol Risk

The following risk must remain visible:

- The current Research Agent contracts and local baseline reproduction use `long_view`, GAUC, and nDCG@5.
- Other project materials still contain conflicting information suggesting that the final evaluation may use `is_click`, NDCG@10, and Recall@50.

This guide describes the current code implementation. It does not mean that the final evaluation-protocol risk has been resolved.

The risk is recorded in `EVALUATION_PROTOCOL_RISK.md`. Before final integration or submission, the final confirmed Problem Statement and the organizer's evaluator must be treated as the highest authority.

## Appendix: Quick Verification Checklist

- [ ] pytest reports that all tests passed.
- [ ] The offline smoke output contains `repository://`, `dataset://`, and `literature-local://`.
- [ ] The online smoke output contains `literature-arxiv:` or a DOI source. External-service failures are recorded instead of silently ignored.
- [ ] A real model call returns a `ResearchDecision` and `model_usage`.
- [ ] A proposal does not request test labels, the public holdout, external training data, or prohibited pretrained weights.
- [ ] If the baseline is missing, the agent does not jump directly to `optimization`.
- [ ] An `experiment_proposal` is not mistaken for an `EvaluationResult`.
