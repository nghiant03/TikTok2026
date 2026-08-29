from tiktok2026.contracts import LessonRecord


def validate_lesson(lesson: LessonRecord) -> LessonRecord:
    if not lesson.experiment_ids:
        raise ValueError("lesson requires supporting experiments")
    return lesson
