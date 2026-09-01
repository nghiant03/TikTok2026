from pathlib import Path

import pytest
from pydantic import ValidationError

from tiktok2026.config import (
    AppSettings,
    BudgetSettings,
    ExecutionSettings,
    LiteLLMSearchSettings,
    ModelSettings,
    OnlineResearchSettings,
)
from tiktok2026.contracts import CURRENT_EVALUATOR_ID, AgentRole, RuntimePaths


def test_runtime_root_must_be_outside_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        RuntimePaths.create(tmp_path, tmp_path / ".runtime")


def test_production_evaluator_default_is_the_versioned_current_identity() -> None:
    assert AppSettings.model_fields["evaluator_id"].default == CURRENT_EVALUATOR_ID


def test_production_rejects_non_authoritative_evaluator(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="current authoritative evaluator"):
        AppSettings(
            repository_root=tmp_path / "repo",
            runtime_root=tmp_path / "runtime",
            profile="production",
            evaluator_id="custom-evaluator",
        )


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


def test_online_research_is_disabled_and_bounded_by_default() -> None:
    settings = OnlineResearchSettings()
    assert settings.enabled is False
    assert settings.max_searches == 3
    assert settings.max_results_per_search == 5
    with pytest.raises(ValidationError):
        OnlineResearchSettings(max_searches=9)


def test_litellm_search_settings_are_independent_and_bounded() -> None:
    settings = OnlineResearchSettings(provider="litellm_search")
    assert settings.litellm_search == LiteLLMSearchSettings()
    assert settings.litellm_search.base_url == "http://127.0.0.1:4000/v1"
    assert settings.litellm_search.search_tool_name == "research-search"
    with pytest.raises(ValidationError):
        LiteLLMSearchSettings(timeout_seconds=5.0)


def test_online_research_requires_a_research_model(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="requires research model"):
        AppSettings(
            repository_root=tmp_path / "repo",
            runtime_root=tmp_path / "runtime",
            models={},
            online_research=OnlineResearchSettings(enabled=True),
        )


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
    assert settings.gpu_count == 0
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
    assert settings.execution.timeout_seconds == 7_200
    assert settings.execution.cpus == 1.0
    assert settings.execution.gpu_count == 1
    assert settings.budget.gpu_hours == 48.0
    assert settings.budget.wall_clock_seconds == 172_800
    assert settings.budget.tokens == 2_000_000_000
    assert settings.budget.disk_bytes == 107_374_182_400
    assert settings.models[AgentRole.RESEARCH].max_tokens == 32_768


def test_litellm_operator_enables_independent_keyless_search() -> None:
    settings = AppSettings.load(
        repository_root=Path("/tmp/repo"),
        profile_path=Path("config/budgets/development.toml"),
        operator_path=Path("config/litellm/operator-models.toml"),
        overrides={"runtime_root": Path("/tmp/tiktok2026-test-runtime")},
    )
    assert settings.models[AgentRole.RESEARCH].model == "tiktok2026-research"
    assert settings.online_research.enabled is True
    assert settings.online_research.provider == "litellm_search"
    assert settings.online_research.litellm_search.search_tool_name == "research-search"
