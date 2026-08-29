from pathlib import Path

from tiktok2026.contracts import ExecutionRequest, FailureKind
from tiktok2026.execution.docker import build_docker_command
from tiktok2026.execution.failures import classify_failure


def request(tmp_path: Path) -> ExecutionRequest:
    source = tmp_path / "source"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    source.mkdir()
    dataset.mkdir()
    output.mkdir()
    return ExecutionRequest(
        execution_id="execution-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        image="tiktok2026:test",
        source_path=source,
        dataset_path=dataset,
        output_path=output,
        timeout_seconds=60,
        memory_bytes=1_000_000,
        cpus=1.0,
    )


def test_docker_command_disables_network_and_mounts_data_read_only(tmp_path: Path) -> None:
    command = build_docker_command(request(tmp_path))
    assert "--network=none" in command
    dataset_mount = command[command.index("--mount") + 1]
    assert "readonly" in dataset_mount or any(
        "dataset" in item and "readonly" in item for item in command
    )


def test_cuda_oom_evidence_is_classified() -> None:
    assert classify_failure(137, "CUDA out of memory", timed_out=False) == FailureKind.CUDA_OOM


def test_timeout_takes_priority() -> None:
    assert classify_failure(-1, "", timed_out=True) == FailureKind.TIMEOUT
