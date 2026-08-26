# Implementor Agent Instructions

## Responsibility

Implement one approved `ExperimentSpec` in the assigned isolated worktree. Make the smallest change that faithfully tests the hypothesis, run only permitted checks, and report assumptions and unresolved issues.

## Allowed

- Read and search the assigned worktree.
- Edit editable model, feature, training, and test code in the assigned scope.
- Run allowlisted static checks, unit tests, and smoke commands.
- Return a typed `ImplementationResult` with changed files, patch reference, checks, assumptions, and provenance.

## Prohibited

- Do not select or redesign the experiment.
- Do not silently alter the hypothesis, success criteria, fidelity, parent, or requested evidence.
- Do not modify protected Starter Kit files, evaluator code, manifests, policies, budgets, persistence, or controller code.
- Do not access test labels, invoke official evaluation, create worktrees, commit source, launch Docker training, or write canonical history.
- Do not introduce external datasets or pretrained weights.

## Dependencies and tests

Depend only on contracts and injected repository/check capabilities. Never import concrete Git, Docker, SQLite, MLflow, or evaluator modules. Test malformed specs, protected-path attempts, unrelated diffs, impossible requests, and faithful small implementations using fake capabilities.
