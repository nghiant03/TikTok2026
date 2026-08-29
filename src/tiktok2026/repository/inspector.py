from __future__ import annotations

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
