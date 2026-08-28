from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from research_agent.adapters import (
    ArxivLiteratureEvidenceReader,
    CompositeLiteratureEvidenceReader,
    CrossrefLiteratureEvidenceReader,
    FileSystemRepositoryEvidenceReader,
    JsonlExperimentHistoryReader,
    KuaiRandPureDataEvidenceReader,
    LocalPdfLiteratureEvidenceReader,
    SemanticScholarLiteratureEvidenceReader,
    WebPageLiteratureEvidenceReader,
)
from research_agent.capabilities import ResearchCapabilities
from research_agent.model import DeepSeekModelConfig, DeepSeekResearchModelClient


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Phase2Settings:
    repository_root: Path
    starter_kit_root: Path
    dataset_root: Path
    denied_dataset_root: Path
    history_path: Path
    literature_root: Path
    model_base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-pro"
    model_api_key_env: str = "RESEARCH_MODEL_API_KEY"
    model_timeout_seconds: float | None = None
    model_max_tokens: int | None = None
    online_literature_enabled: bool = True
    web_evidence_urls: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> Phase2Settings:
        project = _project_root()
        external = project / "external"
        timeout_raw = os.environ.get("RESEARCH_MODEL_TIMEOUT_SECONDS")
        max_tokens_raw = os.environ.get("RESEARCH_MODEL_MAX_TOKENS")
        online_raw = os.environ.get("RESEARCH_ONLINE_LITERATURE", "1").strip().lower()
        web_urls_raw = os.environ.get("RESEARCH_WEB_EVIDENCE_URLS", "")
        return cls(
            repository_root=Path(
                os.environ.get(
                    "RESEARCH_REPOSITORY_ROOT",
                    external / "TikTok2026-main",
                )
            ),
            starter_kit_root=Path(
                os.environ.get(
                    "RESEARCH_STARTER_KIT_ROOT",
                    external / "kuairand-starter-kit",
                )
            ),
            dataset_root=Path(
                os.environ.get(
                    "RESEARCH_DATASET_ROOT",
                    external / "KuaiRand-Pure",
                )
            ),
            denied_dataset_root=Path(
                os.environ.get(
                    "RESEARCH_DENIED_DATASET_ROOT",
                    external / ".denied-dataset-root",
                )
            ),
            history_path=Path(
                os.environ.get(
                    "RESEARCH_HISTORY_PATH",
                    project / "history" / "experiments.jsonl",
                )
            ),
            literature_root=Path(
                os.environ.get(
                    "RESEARCH_LITERATURE_ROOT",
                    external / "literature",
                )
            ),
            model_base_url=os.environ.get(
                "RESEARCH_MODEL_BASE_URL", "https://api.deepseek.com"
            ),
            model_name=os.environ.get("RESEARCH_MODEL_NAME", "deepseek-v4-pro"),
            model_api_key_env=os.environ.get(
                "RESEARCH_MODEL_API_KEY_ENV", "RESEARCH_MODEL_API_KEY"
            ),
            model_timeout_seconds=float(timeout_raw) if timeout_raw else None,
            model_max_tokens=int(max_tokens_raw) if max_tokens_raw else None,
            online_literature_enabled=online_raw not in {"0", "false", "no", "off"},
            web_evidence_urls=tuple(
                url.strip() for url in web_urls_raw.split(",") if url.strip()
            ),
        )


def build_phase2_model_client(settings: Phase2Settings) -> DeepSeekResearchModelClient:
    return DeepSeekResearchModelClient(
        config=DeepSeekModelConfig(
            base_url=settings.model_base_url,
            model=settings.model_name,
            api_key_env=settings.model_api_key_env,
            timeout_seconds=settings.model_timeout_seconds,
            max_tokens=settings.model_max_tokens,
        )
    )


def build_phase2_capabilities(settings: Phase2Settings) -> ResearchCapabilities:
    literature_readers: list[object] = [
        LocalPdfLiteratureEvidenceReader(settings.literature_root)
    ]
    if settings.online_literature_enabled:
        literature_readers.extend(
            (
                SemanticScholarLiteratureEvidenceReader(),
                ArxivLiteratureEvidenceReader(),
                CrossrefLiteratureEvidenceReader(),
            )
        )
        if settings.web_evidence_urls:
            literature_readers.append(
                WebPageLiteratureEvidenceReader(settings.web_evidence_urls)
            )
    return ResearchCapabilities(
        repository=FileSystemRepositoryEvidenceReader(settings.repository_root),
        data=KuaiRandPureDataEvidenceReader(
            settings.dataset_root,
            denied_roots=(settings.denied_dataset_root,),
        ),
        memory=JsonlExperimentHistoryReader(settings.history_path),
        literature=CompositeLiteratureEvidenceReader(tuple(literature_readers)),
    )
