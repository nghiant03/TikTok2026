# Evaluation Instructions

## Authority

Evaluation is deterministic and independent of LangGraph and runtime agents. It validates row identity, invokes the configured evaluator, parses exact metrics, hashes inputs and outputs, and returns an immutable `EvaluationResult`.

## Rules

- Authoritative challenge metrics are NDCG@10 and Recall@50 with equal-weight validation ranking unless the organizer contract replaces this rule.
- The protected Starter Kit currently computes GAUC and nDCG@5. Keep it available as a diagnostic adapter only; label every result from it provisional.
- Never reinterpret, round before comparison, or silently combine metrics beyond the configured official rule.
- Reject missing, extra, reordered, duplicate-row-ID, non-finite, or schema-invalid predictions.
- Agents cannot invoke evaluators directly or supply evaluator paths, commands, images, or parsing rules.
- Test labels are inaccessible during research. One controller-only final evaluation is allowed after convergence and cannot create another experiment route.
- Record evaluator identity, version/hash, dataset manifest, prediction hash, checkpoint ID, source commit, command, and raw artifact reference.

## Dependencies and tests

Depend on contracts, benchmark manifests, and injected process/artifact capabilities. Never depend on agents or LangGraph. Maintain golden metric fixtures, malformed submission tests, hash/provenance tests, provisional-label tests, and hidden-test access tests.
