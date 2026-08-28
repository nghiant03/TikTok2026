# Experiment History

The upper-level deterministic persistence component writes `experiments.jsonl` and the optional
`lessons.jsonl`. The Research Agent has read-only access.

Each line in `experiments.jsonl` must match `ExperimentHistoryItem` and may include
`hypothesis_id`, `parent_experiment_id`, and `tags`. The Research Agent retrieves related
experiments from the task text and follows `parent_experiment_id` links to return ancestor
lineage.

Each line in `lessons.jsonl` must match `ResearchLesson`. A lesson records a claim, evidence
strength, scope, tags, supporting experiment IDs, and evidence references.

A missing file means that no corresponding history or lesson has been recorded. It is not an
error.
