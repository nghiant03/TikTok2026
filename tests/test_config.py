from pathlib import Path

import pytest
from pydantic import ValidationError

from tiktok2026.config import AppSettings, ModelSettings
from tiktok2026.contracts import RuntimePaths


def test_runtime_root_must_be_outside_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        RuntimePaths.create(tmp_path, tmp_path / ".runtime")


def test_model_settings_support_common_openai_endpoints() -> None:
    settings = ModelSettings(
        base_url="https://api.openai.com/v1",
        model="gpt-4.1",
        api_key_env="OPENAI_API_KEY",
    )
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4.1"


def test_settings_reject_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "repository_root": tmp_path,
                "runtime_root": tmp_path.parent / "runtime",
                "unknown": "forbidden",
            }
        )
