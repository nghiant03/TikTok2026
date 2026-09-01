from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path


class RepositoryInspector:
    def __init__(self, repository: Path) -> None:
        self.repository = repository.resolve()

    def _resolve(self, relative_path: str) -> Path:
        path = (self.repository / relative_path).resolve()
        if path != self.repository and self.repository not in path.parents:
            raise ValueError("path escapes repository")
        return path

    def read(self, relative_path: str, max_characters: int = 20_000) -> str:
        return self._resolve(relative_path).read_text(encoding="utf-8")[:max_characters]

    def read_at_commit(self, commit: str, relative_path: str, max_characters: int = 20_000) -> str:
        """Read bounded source from an exact immutable commit."""
        if not commit or not relative_path:
            raise ValueError("commit and repository-relative path are required")
        self._resolve(relative_path)
        result = subprocess.run(
            ("git", "show", f"{commit}:{relative_path}"),
            cwd=self.repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout[:max_characters]

    def sha256_at_commit(self, commit: str, relative_path: str) -> str:
        """Return the content digest for a path at an exact commit."""
        self._resolve(relative_path)
        result = subprocess.run(
            ("git", "show", f"{commit}:{relative_path}"),
            cwd=self.repository,
            check=True,
            capture_output=True,
        )
        return hashlib.sha256(result.stdout).hexdigest()

    def structural_summary_at_commit(
        self, commit: str, relative_path: str, max_items: int = 32
    ) -> tuple[str, ...]:
        """Return top-level Python construct names without exposing full source."""
        source = self.read_at_commit(commit, relative_path)
        tree = ast.parse(source, filename=relative_path)
        summary = tuple(
            f"{node.__class__.__name__.lower()}:{node.name}"
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        )
        return summary[:max_items]

    def search(self, query: str, max_results: int = 20) -> tuple[str, ...]:
        results: list[str] = []
        for path in sorted(self.repository.rglob("*.py")):
            if ".git" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if query in text:
                results.append(path.relative_to(self.repository).as_posix())
                if len(results) >= max_results:
                    break
        return tuple(results)
