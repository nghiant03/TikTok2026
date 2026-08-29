from __future__ import annotations

import csv
from pathlib import Path

from tiktok2026.contracts import PredictionRow


class KuaiRandPureAdapter:
    def write_prediction_submission(
        self,
        output: Path,
        predictions: tuple[PredictionRow, ...],
        expected: tuple[PredictionRow, ...],
    ) -> Path:
        if len(predictions) != len(expected) or not predictions:
            raise ValueError("predictions must contain exactly the expected rows")
        for prediction, expected_row in zip(predictions, expected, strict=True):
            prediction_identity = (
                prediction.row_id,
                prediction.row_identity,
                prediction.user_id,
                prediction.item_id,
            )
            expected_identity = (
                expected_row.row_id,
                expected_row.row_identity,
                expected_row.user_id,
                expected_row.item_id,
            )
            if prediction_identity != expected_identity:
                raise ValueError("submission prediction identity mismatch or reordering")
        return self._write_submission(
            output,
            tuple(row.user_id for row in predictions),
            tuple(row.item_id for row in predictions),
            tuple(row.score for row in predictions),
            tuple(row.row_id for row in predictions),
        )

    def _write_submission(
        self,
        output: Path,
        user_ids: tuple[str, ...],
        video_ids: tuple[str, ...],
        scores: tuple[float, ...],
        row_ids: tuple[str, ...],
    ) -> Path:
        if not (len(row_ids) == len(user_ids) == len(video_ids) == len(scores)):
            raise ValueError("submission arrays must have equal length")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row_id", "user_id", "video_id", "score"))
            for row_id, values in zip(
                row_ids, zip(user_ids, video_ids, scores, strict=True), strict=True
            ):
                writer.writerow((row_id, values[0], values[1], f"{values[2]:.8f}"))
        return output
