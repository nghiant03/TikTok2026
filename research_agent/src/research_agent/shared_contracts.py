"""Read-only bridge to the authoritative contracts in TikTok2026-main."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def _load_contract_module() -> ModuleType:
    configured_path = os.environ.get("TIKTOK2026_CONTRACTS_FILE")
    contract_path = (
        Path(configured_path)
        if configured_path
        else Path(__file__).resolve().parents[2]
        / "external"
        / "TikTok2026-main"
        / "src"
        / "tiktok2026"
        / "contracts"
        / "models.py"
    )
    if not contract_path.is_file():
        raise ModuleNotFoundError(
            "Cannot locate the bundled tiktok2026 contracts. Restore the external snapshot "
            "or set TIKTOK2026_CONTRACTS_FILE to models.py."
        ) from None

    module_name = "_research_agent_tiktok2026_contracts"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load contract module from {contract_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    return module


_contracts = _load_contract_module()

EvaluationResult = _contracts.EvaluationResult
ExecutionResult = _contracts.ExecutionResult
ExperimentSpec = _contracts.ExperimentSpec
FailureKind = _contracts.FailureKind
Fidelity = _contracts.Fidelity
MetricValue = _contracts.MetricValue
ResourceState = _contracts.ResourceState

__all__ = [
    "EvaluationResult",
    "ExecutionResult",
    "ExperimentSpec",
    "FailureKind",
    "Fidelity",
    "MetricValue",
    "ResourceState",
]
