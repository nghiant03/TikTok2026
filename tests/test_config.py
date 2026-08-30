from pathlib import Path

import pytest
from pydantic import ValidationError

from tiktok2026.config import AppSettings, BudgetSettings, ExecutionSettings, ModelSettings
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


def test_model_settings_reasoning_effort_defaults_none() -> None:
    settings = ModelSettings()
    assert settings.reasoning_effort is None


def test_model_settings_accepts_reasoning_effort() -> None:
    settings = ModelSettings(reasoning_effort="high")
    assert settings.reasoning_effort == "high"


def test_settings_reject_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate(
            {
                "repository_root": tmp_path,
                "runtime_root": tmp_path.parent / "runtime",
                "unknown": "forbidden",
            }
        )


def test_settings_load_operator_values(tmp_path: Path) -> None:
    profile = tmp_path / "judged.toml"
    profile.write_text(
        "docker_image = \"controller:sha256\"\n\n"
        "[budget]\nwall_clock_seconds = 90\n",
        encoding="utf-8",
    )

    settings = AppSettings.load(
        repository_root=tmp_path / "repo",
        profile_path=profile,
        overrides={"runtime_root": tmp_path / "runtime"},
    )

    assert settings.budget.wall_clock_seconds == 90
    assert settings.docker_image == "controller:sha256"


def test_execution_settings_have_safe_defaults_and_validate_values() -> None:
    settings = ExecutionSettings()
    assert settings.timeout_seconds == 300
    assert settings.memory_bytes == 1 << 30
    assert settings.cpus == 1.0
    with pytest.raises(ValidationError):
        ExecutionSettings(memory_bytes=0)


def test_budget_allows_three_repairs() -> None:
    assert BudgetSettings().max_repairs == 3
    assert BudgetSettings(max_repairs=3).max_repairs == 3
    with pytest.raises(ValidationError):
        BudgetSettings(max_repairs=4)


def test_development_profile_configures_execution_resources() -> None:
    settings = AppSettings.load(
        repository_root=Path("/tmp/repo"),
        profile_path=Path("config/budgets/development.toml"),
        overrides={"runtime_root": Path("/tmp/tiktok2026-test-runtime")},
    )
    assert settings.execution.memory_bytes == 4_294_967_296
    assert settings.execution.timeout_seconds == 300
    assert settings.execution.cpus == 1.0
