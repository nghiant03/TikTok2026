from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tiktok2026.contracts import (
    ArtifactRecord,
    ArtifactRegistry,
    ArtifactRetention,
    ExperimentSpec,
    SourceRegistration,
    WorktreeAssignment,
)
from tiktok2026.policies.paths import check_changed_paths
from tiktok2026.repository.diffs import normalize_patch, patch_signature

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GitWorktreeManager:
    def __init__(
        self,
        repository: Path,
        runtime_root: Path,
        approved_parent_validator: Callable[[str], bool],
        artifact_registry: ArtifactRegistry,
    ) -> None:
        self.repository = repository.resolve()
        self.runtime_root = runtime_root.resolve()
        self.approved_parent_validator = approved_parent_validator
        self.artifact_registry = artifact_registry

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
        if not self.approved_parent_validator(parent_commit):
            raise ValueError("parent commit is not approved")
        parent_commit = self._git("rev-parse", f"{parent_commit}^{{commit}}")
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
        self,
        assignment: WorktreeAssignment,
        allowed_scopes: tuple[str, ...],
        previous: SourceRegistration | None = None,
    ) -> SourceRegistration:
        worktree_root = self.runtime_root / "worktrees"
        expected_path = worktree_root / assignment.run_id / assignment.experiment_id
        if assignment.path.resolve() != expected_path.resolve():
            raise ValueError("worktree path does not match its approved assignment")
        current_parent = self._git("rev-parse", "HEAD", cwd=assignment.path)
        parent_commit = self._git("rev-parse", f"{assignment.parent_commit}^{{commit}}")
        if not self.approved_parent_validator(parent_commit):
            raise ValueError("parent commit is not approved")
        if previous is not None and (
            previous.experiment_id != assignment.experiment_id
            or previous.run_id != assignment.run_id
            or previous.parent_commit != parent_commit
        ):
            raise ValueError("previous source registration does not match its assignment")
        expected_commit = previous.source_commit if previous is not None else parent_commit
        if current_parent == expected_commit:
            status = self._git("status", "--porcelain", "-z", cwd=assignment.path)
            if not status:
                if previous is not None:
                    return previous
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
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                f"Evaluate experiment {assignment.experiment_id}",
                cwd=assignment.path,
            )
            source_commit = self._git("rev-parse", "HEAD", cwd=assignment.path)
        else:
            self._git(
                "merge-base", "--is-ancestor", expected_commit, current_parent, cwd=assignment.path
            )
            if (
                self._git(
                    "rev-list",
                    "--count",
                    f"{expected_commit}..{current_parent}",
                    cwd=assignment.path,
                )
                != "1"
            ):
                raise ValueError("worktree HEAD is not the approved registration commit")
            if self._git("status", "--porcelain", cwd=assignment.path):
                raise ValueError("source worktree is not clean after registration")
            changed = tuple(
                path
                for path in self._git(
                    "diff",
                    f"{parent_commit}...{current_parent}",
                    "--name-only",
                    cwd=assignment.path,
                ).splitlines()
                if path
            )
            decision = check_changed_paths(changed, allowed_scopes)
            if not decision.allowed:
                raise ValueError(decision.reason)
            source_commit = current_parent
        patch = self._git(
            "diff",
            f"{parent_commit}...{source_commit}",
            "--binary",
            "--no-ext-diff",
            cwd=assignment.path,
        )
        normalized_patch = normalize_patch(patch)
        patch_sha256 = patch_signature(normalized_patch)
        patch_artifact_id = f"patch-{patch_sha256}"
        patch_artifact = self._publish_patch(
            assignment, patch_artifact_id, normalized_patch
        )
        self.artifact_registry.register(
            ArtifactRecord(
                artifact_id=patch_artifact_id,
                run_id=assignment.run_id,
                experiment_id=assignment.experiment_id,
                kind="source_patch",
                uri=patch_artifact.resolve().as_uri(),
                sha256=patch_sha256,
                size_bytes=patch_artifact.stat().st_size,
                producer="controller",
                retention=ArtifactRetention.PROVENANCE,
                created_at=datetime.fromtimestamp(0, UTC),
            )
        )
        if self._git("status", "--porcelain", cwd=assignment.path):
            raise ValueError("source worktree is not clean after registration")
        return SourceRegistration(
            registration_id=f"source-{source_commit}",
            revision=previous.revision + 1 if previous is not None else 0,
            experiment_id=assignment.experiment_id,
            parent_commit=parent_commit,
            source_commit=source_commit,
            patch_sha256=patch_sha256,
            run_id=assignment.run_id,
            patch_artifact_id=patch_artifact_id,
            patch_artifact_uri=patch_artifact.resolve().as_uri(),
            allowed_scopes=allowed_scopes,
            eligible=True,
        )

    def _publish_patch(
        self, assignment: WorktreeAssignment, artifact_id: str, content: str
    ) -> Path:
        destination = (
            self.runtime_root
            / "artifacts"
            / assignment.run_id
            / assignment.experiment_id
            / f"{artifact_id}.diff"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_text(encoding="utf-8") == content:
            return destination
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def remove(self, assignment: WorktreeAssignment) -> None:
        self._git("worktree", "remove", "--force", str(assignment.path))
        if assignment.path.exists():
            shutil.rmtree(assignment.path)
