from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tiktok2026.contracts import AgentRole, RuntimePaths


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)


class BudgetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_hours: float = Field(default=0.0, ge=0.0)
    wall_clock_seconds: int = Field(default=30, ge=0)
    tokens: int = Field(default=0, ge=0)
    disk_bytes: int = Field(default=104_857_600, ge=0)
    reserved_final_gpu_hours: float = Field(default=0.0, ge=0.0)
    frontier_capacity: int = Field(default=4, gt=0)
    max_repairs: int = Field(default=2, ge=0, le=2)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository_root: Path
    runtime_root: Path
    dataset_root: Path | None = None
    profile: str = "test"
    budget: BudgetSettings = BudgetSettings()
    models: dict[AgentRole, ModelSettings] = {role: ModelSettings() for role in AgentRole}
    docker_image: str = "tiktok2026:local"
    evaluator_id: str = "provisional-within-user-v1"
    mlflow_uri: str | None = None
    plateau_epsilon: float = Field(default=0.002, ge=0.0)
    plateau_patience: int = Field(default=3, ge=1)

    @property
    def paths(self) -> RuntimePaths:
        return RuntimePaths.create(self.repository_root, self.runtime_root)

    @classmethod
    def load(
        cls,
        repository_root: Path,
        profile_path: Path,
        operator_path: Path | None = None,
        overrides: dict[str, object] | None = None,
    ) -> AppSettings:
        values: dict[str, Any] = {
            "repository_root": repository_root,
            "runtime_root": Path(
                os.environ.get(
                    "TIKTOK2026_RUNTIME_ROOT",
                    str(repository_root.resolve().parent / f"{repository_root.name}.runtime"),
                )
            ),
            "dataset_root": (
                Path(os.environ["TIKTOK2026_KUAIRAND_PURE_DATA"])
                if "TIKTOK2026_KUAIRAND_PURE_DATA" in os.environ
                else None
            ),
            "profile": profile_path.stem,
        }
        values.update(_load_toml(profile_path))
        if operator_path is not None:
            values.update(_load_toml(operator_path))
        if overrides:
            values.update(overrides)
        return cls.model_validate(values)


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return {str(key): value for key, value in raw.items()}
