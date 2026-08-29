import json
import os
import re
import uuid
from pathlib import Path
from typing import cast

from tiktok2026.contracts import ContractModel

SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)(api[_-]?key\s*=\s*)[^\s]+"),
)


def _redact(value: object) -> object:
    if isinstance(value, str):
        for pattern in SECRET_PATTERNS:
            value = pattern.sub(r"\1[REDACTED]", value)
        return value
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return {str(key): _redact(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_redact(item) for item in sequence]
    return value


class RestrictedTraceSink:
    def __init__(self, root: Path) -> None:
        self.root = root

    def record(self, run_id: str, payload: ContractModel) -> Path:
        directory = self.root / run_id
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        serialized = json.dumps(_redact(payload.model_dump(mode="json")), sort_keys=True)
        path = directory / f"trace-{uuid.uuid4().hex}.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
        return path
