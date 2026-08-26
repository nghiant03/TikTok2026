# TikTok2026 Autonomous Recommender Research

An autonomous research controller for forming, implementing, validating, executing, and learning from recommender-system experiments with deterministic policy and provenance boundaries.

The canonical architecture is documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The files under `baseline/` are protected Starter Kit references and are not the editable experiment target.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run pyright
uv run tiktok2026 synthetic-run --iterations 2
```

The synthetic lifecycle exercises two complete graph cycles without network access, LLM calls, Docker, GPU, or KuaiRand data.

## Runtime boundaries

- Set `TIKTOK2026_RUNTIME_ROOT` to a sibling runtime directory.
- Set `TIKTOK2026_KUAIRAND_PURE_DATA` to an external read-only dataset directory.
- Judging metrics are NDCG@10 and Recall@50. Current local implementations are provisional until organizer evaluator code is supplied.
- Runtime agents cannot access test labels. A controller-only test evaluation may run once after convergence.

## Current status

This repository contains the approved architecture, protected benchmark manifest, budget profiles, initial SQL migrations, Docker image definition, typed core contracts, and a deterministic synthetic graph scaffold. Production agent, repository, execution, evaluation, persistence, API, and memory adapters remain implementation work.
