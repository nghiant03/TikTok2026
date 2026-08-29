from tiktok2026.contracts import EvidenceItem, LessonRecord
from tiktok2026.persistence.repositories import ApplicationRepository


class PersistenceMemoryReader:
    def __init__(self, repository: ApplicationRepository) -> None:
        self.repository = repository

    def retrieve(self, query: str, limit: int) -> tuple[EvidenceItem, ...]:
        if limit < 1:
            return ()
        terms = tuple(term.casefold() for term in query.split() if term)
        matches: list[EvidenceItem] = []
        for payload in self.repository.list_json("lesson"):
            lesson = LessonRecord.model_validate_json(payload)
            searchable = " ".join((lesson.statement, *lesson.tags)).casefold()
            if terms and not all(term in searchable for term in terms):
                continue
            experiments = ",".join(lesson.experiment_ids)
            matches.append(
                EvidenceItem(
                    evidence_id=f"memory-{lesson.lesson_id}",
                    kind="memory_lesson",
                    summary=lesson.statement,
                    source_ref=f"lesson:{lesson.lesson_id};experiments:{experiments}",
                )
            )
            if len(matches) == limit:
                break
        return tuple(matches)
