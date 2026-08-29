# Integration Provenance

## Selective-port policy

The canonical implementation selectively ports useful behavior from two source branches rather than merging their histories or payloads directly. Canonical contracts, deterministic authority boundaries, protected benchmark semantics, external runtime storage, and NDCG@10/Recall@50 take precedence.

## Research Agent source

- Source branch: `origin/research-agent`
- Source implementation commit: `8c776fd11c612211375d0712490a01642abb5187`
- Source branch tip inspected: `317a64772e73fa165d937bc82ed51288f00f4bec`
- Original author: `Lumos088 <wangzhengyuan55@163.com>`

Ported behavior:

- concurrent bounded gathering from repository, data-summary, memory, and literature capabilities;
- typed evidence provenance and evidence-ID validation;
- test-label and unauthorized-evidence rejection;
- generic OpenAI-compatible structured response handling;
- one bounded schema repair;
- experiment identity, evidence reference, and implementation-scope checks;
- persistence-backed lesson retrieval and configured licensed local literature retrieval.

Excluded payloads and behavior:

- copied KuaiRand datasets and derived data;
- nested repository and Starter Kit snapshots;
- PDFs and cached full papers without canonical license records;
- generated `submission_final.csv` and run logs;
- duplicate standalone contracts and loaders;
- GAUC/nDCG@5 judging authority;
- online retrieval from guessed or unconfigured URLs;
- direct shell, evaluator, test-label, and repository-write authority;
- fixed recommender recipes and external training assets.

## Orchestration Agent source

- Source branch: `origin/orchestration-agent`
- Source commit: `448f7e39e70d5745a784a72f7305bd5ad8df357c`
- Original author: `AndyYeom <130391646+AndyYeom@users.noreply.github.com>`

Ported behavior:

- typed orchestration decisions constrained to deterministic allowed actions;
- iterative graph-loop, bounded repair, convergence, recovery, and finalization concepts;
- compact reference-only graph state and finite routes;
- deterministic persistence, audit, resource, and export boundaries;
- provisional final bundle semantics;
- external sibling-worktree and source-registration concepts.

Excluded payloads and behavior:

- direct branch merge and copied prototype packages;
- fixed BPR or model-technique queues;
- test-metric-guided iteration and later routing from final evaluation;
- direct subprocess, Git, Docker, SQL, evaluator, or persistence authority in agents or graph nodes;
- mutable runtime directories inside the repository;
- generated submissions, histories, traces, and free-form authoritative dictionaries;
- FastAPI, Uvicorn, REST, SSE, and frontend integration.

## Authorship preservation status

This document records the exact source identities and port boundaries. Contributor-authored reconstructed commits and merge history are a separate Git-history operation and must be explicitly authorized before creation. No claim is made that uncommitted working-tree changes currently preserve commit authorship.
