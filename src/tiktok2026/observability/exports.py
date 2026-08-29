from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def export_records(
    run_id: str, records: tuple[Mapping[str, object], ...], output: Path
) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda record: str(record.get("event_id", "")))
    jsonl = output / "iterations.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in ordered
        ),
        encoding="utf-8",
    )
    markdown = output / "iterations.md"
    lines = [f"# Run {run_id}", ""]
    for record in ordered:
        lines.extend(
            (
                f"## {record.get('event_id', 'event')}",
                "",
                "```json",
                json.dumps(record, sort_keys=True, indent=2),
                "```",
                "",
            )
        )
    markdown.write_text("\n".join(lines), encoding="utf-8")
    return jsonl, markdown
