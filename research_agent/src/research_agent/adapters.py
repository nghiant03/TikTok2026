from __future__ import annotations

import asyncio
import csv
import hashlib
import html
import json
import os
import re
import ssl
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from research_agent.contracts import (
    EvidenceItem,
    EvidenceKind,
    EvidenceStrength,
    ExperimentHistoryItem,
    ResearchLesson,
    ResearchMemoryQueryResult,
    ResearchRequest,
)

_SKIPPED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
_REPOSITORY_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
}
_SECRET_FILE_NAMES = {".env", ".env.local", ".env.production", "credentials.json"}
_SECRET_VALUE_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|secret|token)\s*[:=]\s*([^\s,;]+)"
)


class EvidenceAccessError(ValueError):
    """A configured reader attempted to leave its authorized read boundary."""


def _normalized(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _is_within(path: Path, root: Path) -> bool:
    candidate = _normalized(path)
    boundary = _normalized(root)
    try:
        return os.path.commonpath((candidate, boundary)) == boundary
    except ValueError:
        return False


def _require_allowed(path: Path, root: Path, denied_roots: tuple[Path, ...] = ()) -> None:
    if not _is_within(path, root):
        raise EvidenceAccessError(f"path is outside authorized root: {path}")
    for denied in denied_roots:
        if _is_within(path, denied):
            raise EvidenceAccessError(f"path is inside a denied root: {path}")


def _bounded_text(text: str, limit: int) -> str:
    collapsed = " ".join(text.replace("\x00", " ").split())
    redacted = _SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}=<redacted>", collapsed)
    if len(redacted) <= limit:
        return redacted
    return redacted[: max(1, limit - 1)].rstrip() + "…"


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


@dataclass(frozen=True)
class FileSystemRepositoryEvidenceReader:
    root: Path
    max_items: int = 8
    max_file_bytes: int = 256_000
    max_summary_chars: int = 900

    async def read_repository_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        return await asyncio.to_thread(self._read, request)

    def _read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        root = self.root.resolve(strict=True)
        priority_paths = {
            "AGENTS.md": 100,
            "README.md": 95,
            "docs/ARCHITECTURE.md": 110,
            "src/tiktok2026/contracts/models.py": 105,
            "src/tiktok2026/benchmark/kuaireand_pure/manifest.json": 105,
            "src/tiktok2026/agents/research/prompt.md": 100,
        }
        objective_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_@-]{3,}", request.objective)
        }
        candidates: list[tuple[int, str, Path, str]] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            if any(part in _SKIPPED_DIRECTORY_NAMES for part in path.relative_to(root).parts):
                continue
            if path.name.lower() in _SECRET_FILE_NAMES or path.suffix.lower() not in _REPOSITORY_SUFFIXES:
                continue
            _require_allowed(path, root)
            if path.stat().st_size > self.max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            lower = text.lower()
            score = priority_paths.get(relative, 0)
            score += sum(3 for token in objective_tokens if token in relative.lower())
            score += min(15, sum(1 for token in objective_tokens if token in lower))
            if score or relative.endswith(("README.md", "AGENTS.md")):
                candidates.append((score, relative, path, text))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        evidence: list[EvidenceItem] = []
        for _, relative, path, text in candidates[: self.max_items]:
            stat = path.stat()
            source_fingerprint = f"{relative}:{stat.st_size}:{stat.st_mtime_ns}"
            evidence.append(
                EvidenceItem(
                    evidence_id=_stable_id("repository", source_fingerprint),
                    kind=EvidenceKind.REPOSITORY,
                    summary=f"{relative}: {_bounded_text(text, self.max_summary_chars)}",
                    source_ref=f"repository://TikTok2026-main/{relative}",
                )
            )
        return tuple(evidence)


PRIMARY_LABEL = "long_view"
AUXILIARY_TRAINING_TARGETS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
)
FORBIDDEN_INTERACTION_FEATURES = (
    PRIMARY_LABEL,
    *AUXILIARY_TRAINING_TARGETS,
    "play_time_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
)
SAFE_IMPRESSION_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "duration_ms",
    "is_rand",
    "tab",
)


@dataclass
class KuaiRandPureDataEvidenceReader:
    dataset_root: Path
    denied_roots: tuple[Path, ...] = ()
    _cache: tuple[EvidenceItem, ...] | None = field(default=None, init=False, repr=False)

    async def read_data_evidence(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        del request
        if self._cache is None:
            self._cache = await asyncio.to_thread(self._read)
        return self._cache

    def _read(self) -> tuple[EvidenceItem, ...]:
        root = self.dataset_root.resolve(strict=True)
        _require_allowed(root, root, self.denied_roots)
        data_root = root / "data"
        _require_allowed(data_root, root, self.denied_roots)
        required = (
            "log_random_4_22_to_5_08_pure.csv",
            "log_standard_4_08_to_4_21_pure.csv",
            "log_standard_4_22_to_5_08_pure.csv",
            "user_features_pure.csv",
            "video_features_basic_pure.csv",
            "video_features_statistic_pure.csv",
        )
        inventory: list[str] = []
        headers: dict[str, tuple[str, ...]] = {}
        for name in required:
            path = data_root / name
            _require_allowed(path, root, self.denied_roots)
            if not path.is_file() or path.is_symlink():
                raise EvidenceAccessError(f"required KuaiRand-Pure file is missing: {name}")
            with path.open("r", encoding="utf-8", newline="") as stream:
                header = next(csv.reader(stream), None)
            if not header:
                raise EvidenceAccessError(f"CSV header is missing: {name}")
            headers[name] = tuple(header)
            inventory.append(f"{name}={path.stat().st_size} bytes")

        log_header = headers["log_standard_4_08_to_4_21_pure.csv"]
        missing_columns = sorted(
            set(SAFE_IMPRESSION_COLUMNS + FORBIDDEN_INTERACTION_FEATURES) - set(log_header)
        )
        if missing_columns:
            raise EvidenceAccessError(f"training log is missing expected columns: {missing_columns}")

        train_path = data_root / "log_standard_4_08_to_4_21_pure.csv"
        rows = positives = invalid_labels = 0
        users: set[str] = set()
        videos: set[str] = set()
        date_min: int | None = None
        date_max: int | None = None
        with train_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                label = row[PRIMARY_LABEL]
                if label not in {"0", "1"}:
                    invalid_labels += 1
                    continue
                rows += 1
                positives += int(label)
                users.add(row["user_id"])
                videos.add(row["video_id"])
                date = int(row["date"])
                date_min = date if date_min is None else min(date_min, date)
                date_max = date if date_max is None else max(date_max, date)

        inventory_fingerprint = "|".join(inventory)
        training_fingerprint = (
            f"{rows}:{len(users)}:{len(videos)}:{date_min}:{date_max}:{positives}:"
            f"{invalid_labels}"
        )
        return (
            EvidenceItem(
                evidence_id=_stable_id("data-schema", inventory_fingerprint),
                kind=EvidenceKind.DATA,
                summary=(
                    "KuaiRand-Pure file inventory verified. "
                    + "; ".join(inventory)
                    + ". Log schema contains long_view and auxiliary interaction outcomes."
                ),
                source_ref="dataset://KuaiRand-Pure/data#headers-and-sizes",
            ),
            EvidenceItem(
                evidence_id=_stable_id("data-train", training_fingerprint),
                kind=EvidenceKind.DATA,
                summary=(
                    f"Training-only safe summary: rows={rows}; users={len(users)}; "
                    f"videos={len(videos)}; observed_dates={date_min}..{date_max}; "
                    f"long_view_positive_rate={(positives / rows if rows else 0.0):.6f}; "
                    f"invalid_long_view_labels={invalid_labels}. No validation/public-holdout "
                    "rows were read."
                ),
                source_ref="dataset://KuaiRand-Pure/log_standard_4_08_to_4_21_pure.csv",
            ),
            EvidenceItem(
                evidence_id="data-access-policy:kuairand-pure:v1",
                kind=EvidenceKind.DATA,
                summary=(
                    "Research data boundary: long_view is the primary target and never an input "
                    "feature; validation outcomes are evaluation-only; the local public holdout "
                    "is quarantined from iterative development; the separate organizer hidden "
                    "test is not locally available; training-split "
                    "click/like/follow/comment/forward/hate "
                    "columns may be declared auxiliary training targets but never same-row "
                    "inference features; "
                    f"safe impression columns={','.join(SAFE_IMPRESSION_COLUMNS)}; blocked "
                    f"same-row outcome features={','.join(FORBIDDEN_INTERACTION_FEATURES)}. "
                    "Random-exposure logs and video_features_statistic are quarantined pending "
                    "leakage review."
                ),
                source_ref="problem-statement+verified-kuairand-schema",
            ),
        )


@dataclass(frozen=True)
class JsonlExperimentHistoryReader:
    path: Path
    max_items: int = 100
    max_related_items: int = 12
    max_lesson_items: int = 8
    lessons_path: Path | None = None

    async def query_research_memory(
        self, request: ResearchRequest
    ) -> ResearchMemoryQueryResult:
        return await asyncio.to_thread(self._query, request)

    async def read_experiment_history(
        self, request: ResearchRequest
    ) -> tuple[ExperimentHistoryItem, ...]:
        """Compatibility view for callers that have not migrated to memory queries."""

        result = await self.query_research_memory(request)
        return result.related_experiments

    def _read_history(self) -> tuple[ExperimentHistoryItem, ...]:
        if not self.path.exists():
            return ()
        if not self.path.is_file() or self.path.is_symlink():
            raise EvidenceAccessError(f"experiment history is not a regular file: {self.path}")
        items: list[ExperimentHistoryItem] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    continue
                try:
                    item = ExperimentHistoryItem.model_validate_json(raw)
                except (ValidationError, json.JSONDecodeError) as exc:
                    raise EvidenceAccessError(
                        f"invalid experiment history JSONL at line {line_number}: {exc}"
                    ) from exc
                items.append(item)
        return tuple(items[-self.max_items :])

    def _read_lessons(self) -> tuple[ResearchLesson, ...]:
        path = self.lessons_path or self.path.with_name("lessons.jsonl")
        if not path.exists():
            return ()
        if not path.is_file() or path.is_symlink():
            raise EvidenceAccessError(f"research lessons are not a regular file: {path}")
        lessons: list[ResearchLesson] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, start=1):
                if not raw.strip():
                    continue
                try:
                    lessons.append(ResearchLesson.model_validate_json(raw))
                except (ValidationError, json.JSONDecodeError) as exc:
                    raise EvidenceAccessError(
                        f"invalid research lesson JSONL at line {line_number}: {exc}"
                    ) from exc
        return tuple(lessons[-self.max_items :])

    def _query(self, request: ResearchRequest) -> ResearchMemoryQueryResult:
        history = self._read_history()
        lessons = self._read_lessons()
        tokens = _query_tokens(request.objective)
        anchors = {
            value
            for value in (request.current_experiment_id, request.parent_experiment_id)
            if value is not None
        }

        def history_score(item: ExperimentHistoryItem) -> tuple[int, str]:
            searchable = " ".join((item.experiment_id, item.summary, *item.tags)).lower()
            score = sum(1 for token in tokens if token in searchable)
            if item.experiment_id in anchors or item.parent_experiment_id in anchors:
                score += 100
            return score, item.experiment_id

        related = tuple(
            item
            for item in sorted(history, key=lambda item: (-history_score(item)[0], history_score(item)[1]))
            if history_score(item)[0] > 0
        )[: self.max_related_items]
        if not related:
            related = history[-self.max_related_items :]

        index = {item.experiment_id: item for item in history}
        lineage: list[ExperimentHistoryItem] = []
        cursor = request.current_experiment_id or request.parent_experiment_id
        visited: set[str] = set()
        while cursor is not None and cursor not in visited and cursor in index:
            visited.add(cursor)
            item = index[cursor]
            lineage.append(item)
            cursor = item.parent_experiment_id
        lineage.reverse()

        related_ids = {item.experiment_id for item in (*related, *lineage)}

        def lesson_score(lesson: ResearchLesson) -> tuple[int, str]:
            searchable = " ".join(
                (lesson.claim, lesson.scope, *lesson.affected_modules, *lesson.tags)
            ).lower()
            score = sum(1 for token in tokens if token in searchable)
            score += 20 * len(set(lesson.supporting_experiment_ids) & related_ids)
            strength_score = {
                EvidenceStrength.WEAK: 0,
                EvidenceStrength.MODERATE: 1,
                EvidenceStrength.STRONG: 2,
            }[lesson.evidence_strength]
            return score + strength_score, lesson.lesson_id

        retrieved_lessons = tuple(
            lesson
            for lesson in sorted(
                lessons,
                key=lambda lesson: (-lesson_score(lesson)[0], lesson_score(lesson)[1]),
            )
            if lesson_score(lesson)[0] > 0
        )[: self.max_lesson_items]
        return ResearchMemoryQueryResult(
            query=request.objective,
            related_experiments=related,
            experiment_lineage=tuple(lineage),
            retrieved_lessons=retrieved_lessons,
        )


def _query_tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_@-]{3,}", value)}


PdfTextExtractor = Callable[[Path, int], str]


def _extract_pdf_text(path: Path, max_pages: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise EvidenceAccessError("pypdf is required for local literature reading") from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages[:max_pages])


@dataclass(frozen=True)
class LocalPdfLiteratureEvidenceReader:
    root: Path
    max_items: int = 6
    max_pages_per_pdf: int = 3
    max_summary_chars: int = 1400
    extractor: PdfTextExtractor = field(default=_extract_pdf_text, repr=False)

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        del request
        return await asyncio.to_thread(self._read)

    def _read(self) -> tuple[EvidenceItem, ...]:
        if not self.root.exists():
            return ()
        root = self.root.resolve(strict=True)
        evidence: list[EvidenceItem] = []
        for path in sorted(root.glob("*.pdf"), key=lambda item: item.name.lower()):
            if len(evidence) >= self.max_items:
                break
            if not path.is_file() or path.is_symlink():
                continue
            _require_allowed(path, root)
            stat = path.stat()
            fingerprint = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}"
            extracted = self.extractor(path, self.max_pages_per_pdf)
            summary = _bounded_text(extracted, self.max_summary_chars)
            if not summary:
                summary = "Text extraction returned no text; use the cited local PDF for review."
            evidence.append(
                EvidenceItem(
                    evidence_id=_stable_id("literature-local", fingerprint),
                    kind=EvidenceKind.LITERATURE,
                    summary=f"Local paper {path.name}: {summary}",
                    source_ref=f"literature-local://{path.name}",
                )
            )
        return tuple(evidence)


LiteratureJsonTransport = Callable[[Request, float | None], dict[str, Any]]


def _literature_json_transport(request: Request, timeout: float | None) -> dict[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceAccessError("literature API response must be a JSON object")
    return payload


@dataclass(frozen=True)
class CrossrefLiteratureEvidenceReader:
    max_items: int = 4
    timeout_seconds: float | None = 15.0
    transport: LiteratureJsonTransport = field(default=_literature_json_transport, repr=False)

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        try:
            return await asyncio.to_thread(self._read, request)
        except Exception as exc:
            return (
                EvidenceItem(
                    evidence_id=_stable_id("literature-online-error", type(exc).__name__),
                    kind=EvidenceKind.LITERATURE,
                    summary=f"Online literature retrieval was unavailable: {type(exc).__name__}.",
                    source_ref="https://api.crossref.org/works",
                ),
            )

    def _read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        query = f"recommender systems {request.objective}"
        url = "https://api.crossref.org/works?" + urlencode(
            {"query.bibliographic": query, "rows": self.max_items, "select": "DOI,title,URL,abstract"}
        )
        api_request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "TikTok2026-ResearchAgent/0.2"},
        )
        payload = self.transport(api_request, self.timeout_seconds)
        items = payload.get("message", {}).get("items", [])
        if not isinstance(items, list):
            raise EvidenceAccessError("Crossref response is missing message.items")
        evidence: list[EvidenceItem] = []
        for item in items[: self.max_items]:
            if not isinstance(item, dict):
                continue
            raw_title = item.get("title") or []
            title = raw_title[0] if isinstance(raw_title, list) and raw_title else "Untitled paper"
            doi = str(item.get("DOI") or "").strip()
            source = f"https://doi.org/{doi}" if doi else str(item.get("URL") or url)
            abstract = re.sub(r"<[^>]+>", " ", html.unescape(str(item.get("abstract") or "")))
            summary = _bounded_text(f"{title}. {abstract}", 1200)
            evidence.append(
                EvidenceItem(
                    evidence_id=_stable_id("literature-online", doi or source or title),
                    kind=EvidenceKind.LITERATURE,
                    summary=summary,
                    source_ref=source,
                )
            )
        return tuple(evidence)


@dataclass(frozen=True)
class SemanticScholarLiteratureEvidenceReader:
    max_items: int = 4
    timeout_seconds: float | None = 15.0
    api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    transport: LiteratureJsonTransport = field(default=_literature_json_transport, repr=False)

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        try:
            return await asyncio.to_thread(self._read, request)
        except Exception as exc:
            return (
                EvidenceItem(
                    evidence_id=_stable_id("literature-semantic-scholar-error", type(exc).__name__),
                    kind=EvidenceKind.LITERATURE,
                    summary=f"Semantic Scholar retrieval was unavailable: {type(exc).__name__}.",
                    source_ref="https://api.semanticscholar.org/graph/v1/paper/search",
                ),
            )

    def _read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urlencode(
            {
                "query": f"recommender systems {request.objective}",
                "limit": self.max_items,
                "fields": "paperId,title,abstract,url,year,externalIds",
            }
        )
        headers = {"Accept": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["x-api-key"] = api_key
        payload = self.transport(Request(url, headers=headers), self.timeout_seconds)
        raw_items = payload.get("data", [])
        if not isinstance(raw_items, list):
            raise EvidenceAccessError("Semantic Scholar response is missing data")
        evidence: list[EvidenceItem] = []
        for item in raw_items[: self.max_items]:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paperId") or "").strip()
            title = str(item.get("title") or "Untitled paper")
            abstract = str(item.get("abstract") or "")
            source = str(item.get("url") or "").strip()
            if not source and paper_id:
                source = f"https://www.semanticscholar.org/paper/{paper_id}"
            evidence.append(
                EvidenceItem(
                    evidence_id=_stable_id("literature-semantic-scholar", paper_id or source or title),
                    kind=EvidenceKind.LITERATURE,
                    summary=_bounded_text(f"{title}. {abstract}", 1200),
                    source_ref=source or url,
                )
            )
        return tuple(evidence)


LiteratureTextTransport = Callable[[Request, float | None], str]


def _literature_text_transport(request: Request, timeout: float | None) -> str:
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi is a declared runtime dependency
        context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ArxivLiteratureEvidenceReader:
    max_items: int = 4
    timeout_seconds: float | None = 15.0
    transport: LiteratureTextTransport = field(default=_literature_text_transport, repr=False)

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        try:
            return await asyncio.to_thread(self._read, request)
        except Exception as exc:
            return (
                EvidenceItem(
                    evidence_id=_stable_id("literature-arxiv-error", type(exc).__name__),
                    kind=EvidenceKind.LITERATURE,
                    summary=f"arXiv retrieval was unavailable: {type(exc).__name__}.",
                    source_ref="https://export.arxiv.org/api/query",
                ),
            )

    def _read(self, request: ResearchRequest) -> tuple[EvidenceItem, ...]:
        objective_terms = [
            token
            for token in re.findall(r"[A-Za-z0-9_-]{3,}", request.objective.lower())
            if token
            not in {
                "backed",
                "evidence",
                "evidence-backed",
                "experiment",
                "informative",
                "kuairand",
                "kuairand-pure",
                "next",
                "propose",
                "recommender",
                "recommenders",
                "system",
                "systems",
                "the",
            }
        ]
        search_terms = list(dict.fromkeys(("recommender", "system", *objective_terms[:3])))
        url = "https://export.arxiv.org/api/query?" + urlencode(
            {
                "search_query": " AND ".join(f"all:{term}" for term in search_terms),
                "start": 0,
                "max_results": self.max_items,
            }
        )
        root = ET.fromstring(self.transport(Request(url), self.timeout_seconds))
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        evidence: list[EvidenceItem] = []
        for entry in root.findall("atom:entry", namespace)[: self.max_items]:
            source = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
            title = " ".join(
                (entry.findtext("atom:title", default="Untitled paper", namespaces=namespace) or "").split()
            )
            summary = " ".join(
                (entry.findtext("atom:summary", default="", namespaces=namespace) or "").split()
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=_stable_id("literature-arxiv", source or title),
                    kind=EvidenceKind.LITERATURE,
                    summary=_bounded_text(f"{title}. {summary}", 1200),
                    source_ref=source or url,
                )
            )
        return tuple(evidence)


@dataclass(frozen=True)
class WebPageLiteratureEvidenceReader:
    urls: tuple[str, ...]
    max_items: int = 4
    timeout_seconds: float | None = 15.0
    transport: LiteratureTextTransport = field(default=_literature_text_transport, repr=False)

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        del request
        groups = await asyncio.gather(
            *(asyncio.to_thread(self._read_one, url) for url in self.urls[: self.max_items]),
            return_exceptions=True,
        )
        evidence: list[EvidenceItem] = []
        for url, result in zip(self.urls, groups, strict=False):
            if isinstance(result, BaseException):
                evidence.append(
                    EvidenceItem(
                        evidence_id=_stable_id("literature-web-error", url),
                        kind=EvidenceKind.LITERATURE,
                        summary=f"Web retrieval was unavailable: {type(result).__name__}.",
                        source_ref=url,
                    )
                )
            else:
                evidence.append(result)
        return tuple(evidence)

    def _read_one(self, url: str) -> EvidenceItem:
        if not url.lower().startswith(("https://", "http://")):
            raise EvidenceAccessError(f"unsupported web evidence URL: {url}")
        raw = self.transport(
            Request(url, headers={"User-Agent": "TikTok2026-ResearchAgent/0.3"}),
            self.timeout_seconds,
        )
        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        return EvidenceItem(
            evidence_id=_stable_id("literature-web", url),
            kind=EvidenceKind.LITERATURE,
            summary=_bounded_text(text, 1400) or "Web page contained no extractable text.",
            source_ref=url,
        )


@dataclass(frozen=True)
class CompositeLiteratureEvidenceReader:
    readers: tuple[Any, ...]
    max_items: int = 8

    async def read_literature_evidence(
        self, request: ResearchRequest
    ) -> tuple[EvidenceItem, ...]:
        groups = await asyncio.gather(
            *(reader.read_literature_evidence(request) for reader in self.readers)
        )
        combined: list[EvidenceItem] = []
        for index in range(max((len(group) for group in groups), default=0)):
            for group in groups:
                if index < len(group):
                    combined.append(group[index])
                if len(combined) >= self.max_items:
                    return tuple(combined)
        return tuple(combined[: self.max_items])
