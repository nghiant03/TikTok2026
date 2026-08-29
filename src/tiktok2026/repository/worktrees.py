from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from tiktok2026.contracts import (
    ExperimentSpec,
    SourceRegistration,
    WorktreeAssignment,
)
from tiktok2026.policies.paths import check_changed_paths
from tiktok2026.repository.diffs import patch_signature

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GitWorktreeManager:
    def __init__(self, repository: Path, runtime_root: Path) -> None:
        self.repository = repository.resolve()
        self.runtime_root = runtime_root.resolve()

    def _git(self, *arguments: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd or self.repository,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()

    def create(self, run_id: str, spec: ExperimentSpec, parent_commit: str) -> WorktreeAssignment:
        if not SAFE_IDENTIFIER.fullmatch(run_id) or not SAFE_IDENTIFIER.fullmatch(
            spec.experiment_id
        ):
            raise ValueError("run and experiment IDs must be safe identifiers")
        self._git("cat-file", "-e", f"{parent_commit}^{{commit}}")
        branch = f"experiment/{run_id}/{spec.experiment_id}"
        path = self.runtime_root / "worktrees" / run_id / spec.experiment_id
        path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "-b", branch, str(path), parent_commit)
        return WorktreeAssignment(
            worktree_id=f"worktree-{spec.experiment_id}",
            run_id=run_id,
            experiment_id=spec.experiment_id,
            path=path,
            branch=branch,
            parent_commit=parent_commit,
        )

    def register_source(
        self, assignment: WorktreeAssignment, allowed_scopes: tuple[str, ...]
    ) -> SourceRegistration:
        current_parent = self._git("rev-parse", "HEAD", cwd=assignment.path)
        if current_parent != assignment.parent_commit:
            raise ValueError("worktree HEAD does not match assigned parent")
        status = self._git("status", "--porcelain", "-z", cwd=assignment.path)
        if not status:
            raise ValueError("source registration requires changes")
        self._git("add", "--all", cwd=assignment.path)
        changed = tuple(
            path
            for path in self._git(
                "diff", "--cached", "--name-only", "-z", cwd=assignment.path
            ).split("\0")
            if path
        )
        decision = check_changed_paths(changed, allowed_scopes)
        if not decision.allowed:
            self._git("reset", cwd=assignment.path)
            raise ValueError(decision.reason)
        self._git(
            "commit",
            "-m",
            f"Evaluate experiment {assignment.experiment_id}",
            cwd=assignment.path,
        )
        source_commit = self._git("rev-parse", "HEAD", cwd=assignment.path)
        patch = self._git(
            "diff",
            f"{assignment.parent_commit}...{source_commit}",
            "--binary",
            "--no-ext-diff",
            cwd=assignment.path,
        )
        if self._git("status", "--porcelain", cwd=assignment.path):
            raise ValueError("source worktree is not clean after registration")
        return SourceRegistration(
            experiment_id=assignment.experiment_id,
            parent_commit=assignment.parent_commit,
            source_commit=source_commit,
            patch_sha256=patch_signature(patch),
        )

    def remove(self, assignment: WorktreeAssignment) -> None:
        self._git("worktree", "remove", "--force", str(assignment.path))
        if assignment.path.exists():
            shutil.rmtree(assignment.path)
