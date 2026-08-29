from pathlib import Path

from tiktok2026.contracts import FrontierCandidate, LessonRecord
from tiktok2026.memory.lessons import validate_lesson
from tiktok2026.observability.exports import export_records
from tiktok2026.search.frontier import select_frontier
from tiktok2026.search.signatures import normalized_signature


def test_signature_ignores_mapping_order() -> None:
    assert normalized_signature({"a": 1, "b": 2}) == normalized_signature({"b": 2, "a": 1})


def test_frontier_keeps_four_bounded_slots() -> None:
    candidates = (
        FrontierCandidate(experiment_id="champ", slot="champion", score=0.9),
        FrontierCandidate(
            experiment_id="alt-1", slot="alternative", score=0.8, diversity_tags=("a",)
        ),
        FrontierCandidate(
            experiment_id="alt-2", slot="alternative", score=0.7, diversity_tags=("b",)
        ),
        FrontierCandidate(experiment_id="diag", slot="diagnostic", score=0.1),
        FrontierCandidate(experiment_id="extra", slot="alternative", score=0.2),
    )
    selected = select_frontier(candidates)
    assert [item.slot for item in selected] == [
        "champion",
        "alternative",
        "alternative",
        "diagnostic",
    ]


def test_lesson_requires_supporting_experiment() -> None:
    lesson = LessonRecord(
        lesson_id="lesson-1",
        statement="A measured result",
        evidence_strength="moderate",
        experiment_ids=("exp-1",),
    )
    assert validate_lesson(lesson) == lesson


def test_exports_are_deterministic(tmp_path: Path) -> None:
    records = ({"event_id": "b", "value": 2}, {"event_id": "a", "value": 1})
    first = export_records("run-1", records, tmp_path / "first")
    second = export_records("run-1", records, tmp_path / "second")
    assert first[0].read_bytes() == second[0].read_bytes()
    assert first[1].read_bytes() == second[1].read_bytes()
