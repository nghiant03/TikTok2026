import asyncio
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    AuthorizedTrainingView,
    DatasetFile,
    load_dataset_manifest,
)
from tiktok2026.contracts import ExecutionRequest, FailureKind
from tiktok2026.execution.docker import (
    DEFAULT_POLICY,
    ArtifactPublicationError,
    AuthorizedTrainingDatasetProvider,
    BoundedOutputCapture,
    DatasetView,
    DockerExecutor,
    ExecutionPolicyError,
    build_docker_command,
    container_name,
    git_output,
    monitor_artifact_quota,
    terminate_process_group,
)
from tiktok2026.execution.failures import classify_failure


def request(tmp_path: Path) -> ExecutionRequest:
    source = tmp_path / "source"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    source.mkdir()
    dataset.mkdir()
    output.mkdir()
    output.chmod(0o777)
    for filename in ("manifest.json", "train.csv", "valid.csv"):
        (dataset / filename).write_text("authorized", encoding="utf-8")
    return ExecutionRequest(
        execution_id="execution-1",
        experiment_id="exp-1",
        source_commit="a" * 40,
        command=("python", "-m", "tiktok2026.experiment.train"),
        image=DEFAULT_POLICY.allowed_image_digests[0],
        source_path=source,
        dataset_path=dataset,
        output_path=output,
        timeout_seconds=60,
        memory_bytes=1_000_000,
        cpus=1.0,
    )


def dataset_view(dataset: Path) -> DatasetView:
    return DatasetView(
        path=dataset,
        manifest_id="manifest-1",
        manifest_sha256="a" * 64,
        manifest_files=("manifest.json",),
        train_files=("train.csv",),
        valid_files=("valid.csv",),
    )


class FakeProvider:
    def __init__(self, view: DatasetView) -> None:
        self.view = view
        self.cleaned = False

    def provide(self, request: ExecutionRequest) -> DatasetView:
        del request
        return self.view

    def cleanup(self, view: DatasetView) -> None:
        assert view is self.view
        self.cleaned = True


class FakeVerifier:
    def verify(self, request: ExecutionRequest) -> None:
        del request


class FakePublisher:
    def publish(
        self,
        *,
        execution_id: str,
        kind: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> str:
        del execution_id, path, sha256, size_bytes
        return f"{kind}-artifact"


class FakeProcess:
    def __init__(self, returncode: int | None) -> None:
        self.pid = 999999
        self.returncode = returncode
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.finished = asyncio.Event()
        if returncode is not None:
            self.finished.set()

    async def wait(self) -> int:
        await self.finished.wait()
        return int(self.returncode or 0)


class FakeLifecycle:
    def __init__(
        self,
        *,
        fail_cleanup: bool = False,
    ) -> None:
        self.fail_cleanup = fail_cleanup
        self.calls: list[str] = []

    async def kill(self, name: str) -> None:
        self.calls.append(f"kill:{name}")
        if self.fail_cleanup:
            raise RuntimeError("kill failed")

    async def remove(self, name: str) -> None:
        self.calls.append(f"remove:{name}")
        if self.fail_cleanup:
            raise RuntimeError("remove failed")


def test_docker_command_disables_network_and_mounts_data_read_only(tmp_path: Path) -> None:
    current = request(tmp_path)
    command = build_docker_command(current, dataset_view=dataset_view(current.dataset_path))
    assert "--network=none" in command
    assert "--rm" not in command
    assert command[command.index("--entrypoint") + 1] == "python"
    image_index = command.index(current.image)
    assert command[image_index + 1 : image_index + 3] == ("-m", "tiktok2026.experiment.train")
    output_mounts = tuple(
        command[index + 1] for index, item in enumerate(command) if item == "--mount"
    )
    assert any(
        "target=/output" in mount
        and f"source={current.output_path.resolve()}" in mount
        and "type=bind" in mount
        and ",rw," in mount
        for mount in output_mounts
    )
    assert not any(item.startswith("/output") for item in command)
    env_index = command.index("--env")
    assert command[env_index + 1] == "HOME=/tmp"
    assert "PYTHONPATH=/workspace/src" in command
    assert command[-2:] == ("--data-root=/dataset", "--data-manifest=/dataset/manifest.json")
    dataset_mount = command[command.index("--mount") + 1]
    assert "readonly" in dataset_mount or any(
        "dataset" in item and "readonly" in item for item in command
    )
    mounts = tuple(command[index + 1] for index, item in enumerate(command) if item == "--mount")
    assert all("bind-recursive=disabled" in mount for mount in mounts)
    assert not any("bind-nonrecursive" in mount for mount in mounts)


def test_cuda_oom_evidence_is_classified() -> None:
    assert classify_failure(137, "CUDA out of memory", timed_out=False) == FailureKind.CUDA_OOM


def test_timeout_takes_priority() -> None:
    assert classify_failure(-1, "", timed_out=True) == FailureKind.TIMEOUT


def test_untrusted_image_and_shell_command_are_rejected(tmp_path: Path) -> None:
    current = request(tmp_path)
    untrusted = current.model_copy(
        update={"image": "untrusted:latest", "command": ("sh", "-c", "rm -rf /")}
    )
    with pytest.raises(ExecutionPolicyError):
        build_docker_command(untrusted, dataset_view=dataset_view(current.dataset_path))


def test_artifact_bind_requires_a_fresh_empty_directory(tmp_path: Path) -> None:
    current = request(tmp_path)
    (current.output_path / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ExecutionPolicyError, match="fresh empty"):
        build_docker_command(current, dataset_view=dataset_view(current.dataset_path))


def test_experiment_package_does_not_eagerly_import_train() -> None:
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import tiktok2026.experiment; "
            "assert 'tiktok2026.experiment.train' not in sys.modules",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert not result.stderr


def test_alternative_data_root_is_rejected(tmp_path: Path) -> None:
    current = request(tmp_path)
    invalid = current.model_copy(
        update={"command": ("python", "-m", "tiktok2026.experiment.train", "--data-root=/bad")}
    )
    with pytest.raises(ExecutionPolicyError):
        build_docker_command(invalid, dataset_view=dataset_view(current.dataset_path))


def test_dataset_view_rejects_unauthorized_and_test_files(tmp_path: Path) -> None:
    current = request(tmp_path)
    (current.dataset_path / "test.csv").write_text("forbidden", encoding="utf-8")
    view = dataset_view(current.dataset_path)
    with pytest.raises(ExecutionPolicyError):
        build_docker_command(current, dataset_view=view)


def test_authorized_training_view_isolated_from_test_files(tmp_path: Path) -> None:
    current = request(tmp_path)
    (current.dataset_path / "test.csv").write_text("held out", encoding="utf-8")
    files = tuple(
        DatasetFile.model_validate(
            {
                "path": filename,
                "sha256": hashlib.sha256(b"authorized").hexdigest(),
                "schema": (),
                "split": split,
            }
        )
        for filename, split in (("train.csv", "train"), ("valid.csv", "valid"))
    )
    verified = AuthorizedTrainingView(
        manifest_id="manifest-1",
        manifest_sha256="a" * 64,
        host_root=current.dataset_path,
        files=files,
    )
    provider = AuthorizedTrainingDatasetProvider(verified)
    staged = provider.provide(current)
    try:
        assert staged.path != current.dataset_path
        assert (staged.path / "manifest.json").is_file()
        assert sorted(path.name for path in staged.path.iterdir()) == [
            "manifest.json",
            "train.csv",
            "valid.csv",
        ]
        assert staged.path.stat().st_mode & 0o777 == 0o755
        assert (staged.path / "manifest.json").stat().st_mode & 0o777 == 0o444
        assert (staged.path / "train.csv").stat().st_mode & 0o777 == 0o444
        assert load_dataset_manifest(staged.path / "manifest.json").manifest_id == "manifest-1"
    finally:
        provider.cleanup(staged)


def test_git_head_output_accepts_trailing_newline(tmp_path: Path) -> None:
    subprocess.run(("git", "init", str(tmp_path)), check=True, capture_output=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "test",
        ),
        check=True,
        capture_output=True,
    )
    commit = subprocess.check_output(
        ("git", "-C", str(tmp_path), "rev-parse", "HEAD"), text=True
    ).strip()
    assert git_output(tmp_path, ("rev-parse", "HEAD"), 40) == commit


def test_executor_requires_authoritative_artifact_publisher(tmp_path: Path) -> None:
    with pytest.raises(ArtifactPublicationError):
        asyncio.run(DockerExecutor().execute(request(tmp_path)))


def test_cancellation_after_launch_cleans_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = request(tmp_path)
    provider = FakeProvider(dataset_view(current.dataset_path))
    lifecycle = FakeLifecycle()

    async def run() -> None:
        process = FakeProcess(returncode=None)

        async def create_process(*args: object, **kwargs: object) -> FakeProcess:
            del args, kwargs
            return process

        async def terminate(process: object, grace: float) -> None:
            del process, grace

        monkeypatch.setattr(
            "tiktok2026.execution.docker.asyncio.create_subprocess_exec",
            create_process,
        )
        monkeypatch.setattr(
            "tiktok2026.execution.docker.terminate_process_group",
            terminate,
        )
        executor = DockerExecutor(
            publisher=FakePublisher(),
            dataset_provider=provider,
            source_verifier=FakeVerifier(),
            lifecycle=lifecycle,
        )
        task = asyncio.create_task(executor.execute(current))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    name = container_name(current.execution_id)
    assert f"kill:{name}" in lifecycle.calls
    assert f"remove:{name}" in lifecycle.calls
    assert provider.cleaned


def test_cleanup_failure_is_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = request(tmp_path)
    lifecycle = FakeLifecycle(fail_cleanup=True)

    async def run() -> None:
        process = FakeProcess(returncode=0)

        async def create_process(*args: object, **kwargs: object) -> FakeProcess:
            del args, kwargs
            return process

        async def terminate(process: object, grace: float) -> None:
            del process, grace

        monkeypatch.setattr(
            "tiktok2026.execution.docker.asyncio.create_subprocess_exec",
            create_process,
        )
        monkeypatch.setattr(
            "tiktok2026.execution.docker.terminate_process_group",
            terminate,
        )
        executor = DockerExecutor(
            publisher=FakePublisher(),
            dataset_provider=FakeProvider(dataset_view(current.dataset_path)),
            source_verifier=FakeVerifier(),
            lifecycle=lifecycle,
        )
        result = await executor.execute(current)
        assert result.failure_kind == FailureKind.DEPENDENCY_ENVIRONMENT
        assert result.exit_code != 0

    asyncio.run(run())


def test_gpu_launch_failure_has_zero_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = request(tmp_path).model_copy(update={"gpu_count": 1})

    async def fail_process(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        raise OSError("docker unavailable")

    async def run() -> None:
        monkeypatch.setattr(
            "tiktok2026.execution.docker.asyncio.create_subprocess_exec", fail_process
        )
        executor = DockerExecutor(
            publisher=FakePublisher(),
            dataset_provider=FakeProvider(dataset_view(current.dataset_path)),
            source_verifier=FakeVerifier(),
            lifecycle=FakeLifecycle(),
        )
        result = await executor.execute(current)
        assert result.gpu_hours == 0.0
        assert result.failure_kind == FailureKind.DEPENDENCY_ENVIRONMENT

    asyncio.run(run())


def test_gpu_execution_stops_without_output_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = request(tmp_path).model_copy(update={"gpu_count": 1})
    lifecycle = FakeLifecycle()

    async def run() -> float:
        process = FakeProcess(returncode=0)

        async def create_process(*args: object, **kwargs: object) -> FakeProcess:
            del args, kwargs
            return process

        async def terminate(process: object, grace: float) -> None:
            del process, grace

        monkeypatch.setattr(
            "tiktok2026.execution.docker.asyncio.create_subprocess_exec", create_process
        )
        monkeypatch.setattr("tiktok2026.execution.docker.terminate_process_group", terminate)
        result = await DockerExecutor(
            publisher=FakePublisher(),
            dataset_provider=FakeProvider(dataset_view(current.dataset_path)),
            source_verifier=FakeVerifier(),
            lifecycle=lifecycle,
        ).execute(current)
        return result.gpu_hours

    assert asyncio.run(run()) < 0.05
    assert f"remove:{container_name(current.execution_id)}" in lifecycle.calls


def test_output_is_bounded_without_buffering(tmp_path: Path) -> None:
    async def capture() -> tuple[int, bool, str]:
        stream = asyncio.StreamReader()
        stream.feed_data(b"0123456789")
        stream.feed_eof()
        capture_file = BoundedOutputCapture(tmp_path / "stdout.log", maximum=4)
        await capture_file.read(stream)
        return capture_file.size, capture_file.truncated, capture_file.text()

    assert asyncio.run(capture()) == (4, True, "0123")


def test_process_group_cleanup_terminates_process() -> None:
    async def cleanup() -> int:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            start_new_session=True,
        )
        await terminate_process_group(process, grace=0.1)
        return int(process.returncode or 0)

    assert asyncio.run(cleanup()) < 0


def test_artifact_quota_monitor_terminates_excess_output(tmp_path: Path) -> None:
    async def monitor() -> bool:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            start_new_session=True,
        )
        (tmp_path / "artifact.bin").write_bytes(b"0123456789")
        stop = asyncio.Event()
        exceeded = await monitor_artifact_quota(
            tmp_path,
            baseline_bytes=0,
            quota_bytes=4,
            stop=stop,
            process=process,
            grace=0.1,
        )
        return exceeded and process.returncode is not None

    assert asyncio.run(monitor())
