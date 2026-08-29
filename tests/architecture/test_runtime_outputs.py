import subprocess
import tomllib
from pathlib import Path

from tiktok2026.contracts import AgentRole

ROOT = Path(__file__).parents[2]


def test_http_stack_is_not_a_runtime_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {
        requirement.split("[")[0].split(">=")[0].split("<")[0]
        for requirement in project["dependencies"]
    }

    assert not {"fastapi", "uvicorn"} & names


def test_source_branch_payloads_are_not_integrated() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    forbidden = (
        "research_agent/",
        "prototype/",
        "submission_final.csv",
        "run_log.jsonl",
        ".pdf",
    )

    assert all(not any(item in path for item in forbidden) for path in tracked)


def test_runtime_has_exactly_four_roles() -> None:
    assert tuple(role.value for role in AgentRole) == (
        "orchestration",
        "research",
        "implementor",
        "validator",
    )


def test_generated_outputs_are_ignored() -> None:
    samples = (
        "submission_final.csv",
        "run_log.jsonl",
        "exports/run/iterations.jsonl",
        "traces/run/trace.json",
        "literature/cache/paper.txt",
    )
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        input="\n".join(samples),
        text=True,
        capture_output=True,
        check=False,
    )

    assert set(result.stdout.splitlines()) == set(samples)
