import subprocess
from pathlib import Path

import pytest

from tiktok2026.contracts import ExperimentSpec, Fidelity
from tiktok2026.persistence.repositories import ApplicationRepository
from tiktok2026.repository.diffs import validate_diff
from tiktok2026.repository.inspector import RepositoryInspector
from tiktok2026.repository.worktrees import GitWorktreeManager


def run_git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, text=True, capture_output=True
    ).stdout.strip()


def create_repository(path: Path) -> str:
    path.mkdir()
    run_git(path, "init")
    run_git(path, "config", "user.name", "Test")
    run_git(path, "config", "user.email", "test@example.invalid")
    run_git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("base\n", encoding="utf-8")
    run_git(path, "add", "README.md")
    run_git(path, "commit", "-m", "base")
    return run_git(path, "rev-parse", "HEAD")


def spec() -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id="exp-1",
        hypothesis_id="hyp-1",
        hypothesis="Change experiment code",
        mechanism="Edit assigned source",
        motivation="Exercise worktree isolation",
        expected_signal="A source commit is registered",
        implementation_scope=("src/tiktok2026/experiment",),
        fidelity=Fidelity.SMOKE,
        success_criteria="Commit exists",
        failure_criteria="Scope is violated",
    )


def create_manager(repository: Path, runtime: Path, parent: str) -> GitWorktreeManager:
    application = ApplicationRepository(runtime / "application.sqlite3")
    application.initialize()
    return GitWorktreeManager(
        repository,
        runtime,
        approved_parent_validator=lambda candidate: candidate == parent,
        artifact_registry=application,
    )


def test_worktree_is_created_under_runtime_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    head = create_repository(repository)
    manager = create_manager(repository, tmp_path / "runtime", head)
    assignment = manager.create("run-1", spec(), head)
    assert assignment.path.is_relative_to(tmp_path / "runtime" / "worktrees")
    assert assignment.parent_commit == head
    manager.remove(assignment)


def test_diff_rejects_out_of_scope_file() -> None:
    result = validate_diff(("README.md",), ("src/tiktok2026/experiment",))
    assert not result.allowed
    assert result.reason == "outside_implementation_scope"


def test_diff_rejects_scope_traversal() -> None:
    result = validate_diff(
        ("src/tiktok2026/experiment/../../../baseline/evaluate.py",),
        ("src/tiktok2026/experiment",),
    )

    assert not result.allowed
    assert result.reason == "invalid_path"


def test_worktree_rejects_unsafe_run_identity(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    head = create_repository(repository)
    manager = create_manager(repository, tmp_path / "runtime", head)

    with pytest.raises(ValueError, match="safe identifier"):
        manager.create("../escaped", spec(), head)


def test_source_registration_commits_only_validated_scope(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    head = create_repository(repository)
    manager = create_manager(repository, tmp_path / "runtime", head)
    assignment = manager.create("run-1", spec(), head)
    target = assignment.path / "src/tiktok2026/experiment/change.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")

    registration = manager.register_source(
        assignment,
        allowed_scopes=("src/tiktok2026/experiment",),
    )

    assert registration.parent_commit == head
    assert registration.source_commit != head
    assert registration.allowed_scopes == ("src/tiktok2026/experiment",)
    assert registration.patch_artifact_uri is not None
    assert Path(registration.patch_artifact_uri.removeprefix("file://")).read_text(
        encoding="utf-8"
    ).endswith("\n")
    assert run_git(assignment.path, "status", "--porcelain") == ""
    assert run_git(assignment.path, "show", "--format=", "--name-only", "HEAD") == (
        "src/tiktok2026/experiment/change.py"
    )
    manager.remove(assignment)


def test_source_registration_rejects_out_of_scope_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    head = create_repository(repository)
    manager = create_manager(repository, tmp_path / "runtime", head)
    assignment = manager.create("run-1", spec(), head)
    (assignment.path / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside_implementation_scope"):
        manager.register_source(
            assignment,
            allowed_scopes=("src/tiktok2026/experiment",),
        )
    manager.remove(assignment)


def test_source_registration_recovers_after_patch_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    head = create_repository(repository)
    manager = create_manager(repository, tmp_path / "runtime", head)
    assignment = manager.create("run-1", spec(), head)
    target = assignment.path / "src/tiktok2026/experiment/change.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")

    def fail_publish(*args: object, **kwargs: object) -> Path:
        raise OSError("injected patch publication failure")

    monkeypatch.setattr(manager, "_publish_patch", fail_publish)
    with pytest.raises(OSError, match="publication failure"):
        manager.register_source(assignment, ("src/tiktok2026/experiment",))
    assert run_git(assignment.path, "rev-parse", "HEAD") != head

    monkeypatch.undo()
    registration = manager.register_source(
        assignment, ("src/tiktok2026/experiment",)
    )
    assert registration.source_commit == run_git(assignment.path, "rev-parse", "HEAD")
    assert registration.patch_artifact_uri is not None
    manager.remove(assignment)


def test_source_registration_commits_a_revision_after_execution_repair(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    head = create_repository(repository)
    manager = create_manager(repository, tmp_path / "runtime", head)
    assignment = manager.create("run-1", spec(), head)
    target = assignment.path / "src/tiktok2026/experiment/change.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n", encoding="utf-8")
    first = manager.register_source(
        assignment, ("src/tiktok2026/experiment",)
    )

    target.write_text("VALUE = 2\n", encoding="utf-8")
    second = manager.register_source(
        assignment, ("src/tiktok2026/experiment",), first
    )

    assert second.revision == 1
    assert second.registration_id != first.registration_id
    assert second.source_commit != first.source_commit
    assert run_git(assignment.path, "rev-list", "--count", f"{head}..HEAD") == "2"
    assert run_git(assignment.path, "status", "--porcelain") == ""
    manager.remove(assignment)


def test_inspector_bounds_reads(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    create_repository(repository)
    (repository / "large.txt").write_text("abcdefghij", encoding="utf-8")
    assert RepositoryInspector(repository).read("large.txt", max_characters=4) == "abcd"


def test_inspector_reads_the_requested_commit_not_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    first_commit = create_repository(repository)
    target = repository / "README.md"
    target.write_text("checkout\n", encoding="utf-8")
    run_git(repository, "add", "README.md")
    run_git(repository, "commit", "-m", "checkout change")

    assert target.read_text(encoding="utf-8") == "checkout\n"
    assert RepositoryInspector(repository).read_at_commit(
        first_commit, "README.md"
    ) == "base\n"
