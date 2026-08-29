from __future__ import annotations

import csv
from pathlib import Path


class KuaiRandPureAdapter:
    def write_submission(
        self,
        output: Path,
        user_ids: tuple[str, ...],
        video_ids: tuple[str, ...],
        scores: tuple[float, ...],
    ) -> Path:
        if not (len(user_ids) == len(video_ids) == len(scores)):
            raise ValueError("submission arrays must have equal length")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            for row_id, values in enumerate(zip(user_ids, video_ids, scores, strict=True)):
                writer.writerow((row_id, values[0], values[1], f"{values[2]:.8f}"))
        return output
