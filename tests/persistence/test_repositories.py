from pathlib import Path

from tiktok2026.contracts import AuditEvent, ExperimentSpec, Fidelity
from tiktok2026.persistence.repositories import ApplicationRepository


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="A deterministic test",
        mechanism="Exercise persistence",
        motivation="Verify canonical storage",
        expected_signal="Records can be reconstructed",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Record persists",
        failure_criteria="Record is unavailable",
    )


def test_experiment_write_is_idempotent(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    repository.put_experiment(spec(), status="proposed")
    repository.put_experiment(spec(), status="proposed")
    assert repository.get_experiment("exp-1") == spec()


def test_final_test_access_can_only_be_claimed_once(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    assert repository.claim_final_test_access("run-1")
    assert not repository.claim_final_test_access("run-1")


def test_audit_event_round_trips(tmp_path: Path) -> None:
    repository = ApplicationRepository(tmp_path / "app.sqlite3")
    repository.initialize()
    event = AuditEvent(
        event_id="event-1",
        run_id="run-1",
        event_type="run_created",
        actor_type="controller",
        actor_id="bootstrap",
        payload={"profile": "test"},
    )
    repository.put_audit_event(event)
    assert repository.list_audit_events("run-1") == (event,)
