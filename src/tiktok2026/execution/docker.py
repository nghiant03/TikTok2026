from __future__ import annotations

import asyncio
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tiktok2026.benchmark.kuaireand_pure.manifest import (
    AuthorizedTrainingView,
    DatasetFile,
    DatasetManifest,
    canonical_manifest_bytes,
    encode_row_identity,
)
from tiktok2026.contracts import (
    ArtifactRetention,
    ExecutionRequest,
    ExecutionResult,
    SourceRegistration,
    WorktreeAssignment,
)
from tiktok2026.execution.failures import classify_failure
from tiktok2026.persistence.artifacts import ArtifactStore
from tiktok2026.persistence.repositories import ApplicationRepository


class ExecutionPolicyError(ValueError):
    """The request cannot be represented by the constrained executor."""


class ArtifactPublicationError(RuntimeError):
    """An execution artifact could not be published."""


class SourceIdentityVerifier(Protocol):
    """Authoritative check for HEAD, cleanliness, assignment, and registration."""

    def verify(self, request: ExecutionRequest) -> None: ...


class DatasetViewProvider(Protocol):
    """Controller-owned provider for an allowlisted dataset view."""

    def provide(self, request: ExecutionRequest) -> DatasetView: ...

    def cleanup(self, view: DatasetView) -> None: ...


class FinalTestAuthorizationAdapter(Protocol):
    """Explicit controller authorization for a final-test dataset view."""

    def authorize(self, request: ExecutionRequest, view: DatasetView) -> bool: ...


class ContainerLifecycle(Protocol):
    async def kill(self, name: str) -> None: ...

    async def remove(self, name: str) -> None: ...


@dataclass(frozen=True)
class DatasetView:
    """A controller-staged directory and its exact authorized file manifest."""

    path: Path
    manifest_id: str
    manifest_sha256: str
    manifest_files: tuple[str, ...]
    train_files: tuple[str, ...]
    valid_files: tuple[str, ...]
    test_files: tuple[str, ...] = ()

    @property
    def authorized_files(self) -> tuple[str, ...]:
        return self.manifest_files + self.train_files + self.valid_files + self.test_files


class AuthorizedTrainingDatasetProvider:
    """Adapter from the verified benchmark training view to the executor seam."""

    def __init__(self, view: AuthorizedTrainingView) -> None:
        if view.container_root != "/dataset":
            raise ExecutionPolicyError("authorized training view must use /dataset")
        self.view = view
        self._stages: set[Path] = set()

    def provide(self, request: ExecutionRequest) -> DatasetView:
        if self.view.host_root.resolve() != request.dataset_path.resolve():
            raise ExecutionPolicyError("dataset request does not match authorized training view")
        train_files = tuple(file.path for file in self.view.files if file.split == "train")
        valid_files = tuple(file.path for file in self.view.files if file.split == "valid")
        if not train_files or not valid_files:
            raise ExecutionPolicyError(
                "authorized training view must contain train and valid files"
            )
        stage = Path(tempfile.mkdtemp(prefix="tiktok2026-dataset-"))
        try:
            files = tuple(self.view.files)
            if len({file.path for file in files}) != len(files):
                raise ExecutionPolicyError("authorized training view contains duplicate files")
            for file in files:
                if not _authorized_relative_path(file.path) or file.path == "manifest.json":
                    raise ExecutionPolicyError("authorized dataset file has an unsafe path")
                source = self.view.host_root / file.path
                destination = stage / file.path
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_resolved = source.resolve(strict=True)
                if (
                    not source.is_file()
                    or source.is_symlink()
                    or source_resolved.parent != self.view.host_root.resolve()
                    and self.view.host_root.resolve() not in source_resolved.parents
                    or _sha256_file(source) != file.sha256
                ):
                    raise ExecutionPolicyError(
                        "authorized dataset file failed checksum verification"
                    )
                shutil.copyfile(source, destination)
                if _sha256_file(destination) != file.sha256:
                    raise ExecutionPolicyError("staged dataset file failed checksum verification")
                destination.chmod(0o444)
                for parent in destination.parents:
                    if parent == stage:
                        break
                    parent.chmod(0o755)
            manifest = _staged_manifest(self.view, files, stage)
            (stage / "manifest.json").write_bytes(canonical_manifest_bytes(manifest))
            (stage / "manifest.json").chmod(0o444)
            stage.chmod(0o755)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        self._stages.add(stage.resolve())
        return DatasetView(
            path=stage,
            manifest_id=self.view.manifest_id,
            manifest_sha256=self.view.manifest_sha256,
            manifest_files=("manifest.json",),
            train_files=train_files,
            valid_files=valid_files,
        )

    def cleanup(self, view: DatasetView) -> None:
        stage = view.path.resolve()
        if stage not in self._stages:
            raise ExecutionPolicyError("dataset stage is not owned by this provider")
        shutil.rmtree(stage, ignore_errors=False)
        self._stages.remove(stage)


class RegisteredGitSourceVerifier:
    """Concrete verifier for the persisted source registration and worktree."""

    def __init__(self, repository: ApplicationRepository, assignment: WorktreeAssignment) -> None:
        self.repository = repository
        self.assignment = assignment

    def verify(self, request: ExecutionRequest) -> None:
        registration = self.repository.get_source_registration(request.experiment_id)
        if registration is None:
            raise ExecutionPolicyError("source registration is unavailable")
        self._verify_registration(request, registration)
        worktree = request.source_path.resolve(strict=True)
        if worktree != self.assignment.path.resolve(strict=True):
            raise ExecutionPolicyError("source path does not match its worktree assignment")
        if git_output(worktree, ("rev-parse", "--show-toplevel"), 4096) != str(worktree):
            raise ExecutionPolicyError("source path is not the assigned Git worktree")
        if git_output(worktree, ("rev-parse", "HEAD"), 40) != request.source_commit:
            raise ExecutionPolicyError("worktree HEAD does not match the registered source")
        if git_output(worktree, ("status", "--porcelain", "--untracked-files=all"), 1):
            raise ExecutionPolicyError("source worktree is not clean")
        if git_output(worktree, ("rev-parse", "--abbrev-ref", "HEAD"), 4096) != (
            self.assignment.branch
        ):
            raise ExecutionPolicyError("worktree branch does not match its assignment")

    def _verify_registration(
        self, request: ExecutionRequest, registration: SourceRegistration
    ) -> None:
        if (
            not registration.eligible
            or registration.experiment_id != request.experiment_id
            or registration.run_id != self.assignment.run_id
            or registration.source_commit != request.source_commit
            or registration.parent_commit != self.assignment.parent_commit
            or not registration.allowed_scopes
            or registration.patch_artifact_id != f"patch-{registration.patch_sha256}"
        ):
            raise ExecutionPolicyError("source registration is not eligible for execution")
        patch = self.repository.get_artifact(registration.patch_artifact_id)
        if (
            patch is None
            or patch.kind != "source_patch"
            or patch.artifact_id != registration.patch_artifact_id
            or patch.run_id != self.assignment.run_id
            or patch.experiment_id != request.experiment_id
            or patch.sha256 != registration.patch_sha256
        ):
            raise ExecutionPolicyError("registered source patch artifact is unavailable")
        patch_path = Path(patch.uri.removeprefix("file://"))
        if (
            not patch_path.is_file()
            or patch.size_bytes != patch_path.stat().st_size
            or _sha256_file(patch_path) != patch.sha256
        ):
            raise ExecutionPolicyError("registered source patch artifact is invalid")


def git_output(worktree: Path, arguments: tuple[str, ...], maximum: int) -> str:
    process = subprocess.Popen(
        ("git", "-C", str(worktree), *arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    output = process.stdout.read(maximum + 2)
    decoded = output.decode("utf-8", errors="strict").strip()
    if len(decoded) > maximum:
        process.kill()
        process.wait()
        raise ExecutionPolicyError("Git identity output exceeded its safety bound")
    try:
        return_code = process.wait(timeout=10.0)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait()
        raise ExecutionPolicyError("Git identity check timed out") from error
    if return_code != 0:
        raise ExecutionPolicyError("Git identity check failed")
    return decoded


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_manifest(
    view: AuthorizedTrainingView, files: tuple[DatasetFile, ...], stage: Path
) -> DatasetManifest:
    dataset_files = [
        {
            "path": file.path,
            "sha256": file.sha256,
            "schema": file.columns,
            "split": file.split,
        }
        for file in sorted(files, key=lambda item: item.path)
    ]
    split_files = {
        split: [file.path for file in files if file.split == split]
        for split in ("train", "valid")
    }
    splits = {
        split: {
            "files": paths,
            "identity_sha256": _staged_split_identity(
                stage, tuple(path for path in paths)
            ),
        }
        for split, paths in split_files.items()
    }
    return DatasetManifest.model_validate(
        {
            "schema_version": "1",
            "manifest_id": view.manifest_id,
            "data_root_env": "TIKTOK2026_FIXED_DATA_ROOT",
            "files": dataset_files,
            "splits": splits,
        }
    )


def _staged_split_identity(stage: Path, paths: tuple[str, ...]) -> str:
    identity_columns = ("row_id", "user_id", "item_id")
    digest = hashlib.sha256()
    try:
        for relative in paths:
            with (stage / relative).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    digest.update(
                        encode_row_identity(
                            tuple(row[column] for column in identity_columns)
                        ).encode()
                    )
                    digest.update(b"\n")
    except (KeyError, UnicodeDecodeError, csv.Error):
        return hashlib.sha256(b"").hexdigest()
    return digest.hexdigest()


class ArtifactPublisher(Protocol):
    def publish(
        self,
        *,
        execution_id: str,
        kind: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> str: ...


class ArtifactStorePublisher:
    """Authoritative publisher adapter with bounded, checksum-verified reads."""

    def __init__(
        self,
        store: ArtifactStore,
        run_id: str,
        experiment_id: str,
        producer: str = "docker-executor",
        maximum_bytes: int = 1 << 20,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.producer = producer
        self.maximum_bytes = maximum_bytes

    def publish(
        self,
        *,
        execution_id: str,
        kind: str,
        path: Path,
        sha256: str,
        size_bytes: int,
    ) -> str:
        del execution_id
        if size_bytes < 0 or size_bytes > self.maximum_bytes:
            raise ArtifactPublicationError("artifact exceeds the publication bound")
        content = _read_verified_bytes(path, sha256, size_bytes, self.maximum_bytes)
        record = self.store.publish_bytes(
            self.run_id,
            self.experiment_id,
            kind,
            path.name,
            content,
            self.producer,
            ArtifactRetention.RUN,
        )
        if record.sha256 != sha256 or record.size_bytes != size_bytes:
            raise ArtifactPublicationError("authoritative artifact record failed byte verification")
        if self.store.repository.get_artifact(record.artifact_id) != record:
            raise ArtifactPublicationError("authoritative artifact record was not persisted")
        return record.artifact_id


def _read_verified_bytes(
    path: Path, expected_sha256: str, expected_size: int, maximum: int
) -> bytes:
    digest = hashlib.sha256()
    content = bytearray()
    with path.open("rb") as handle:
        while chunk := handle.read(min(_READ_CHUNK_BYTES, maximum + 1 - len(content))):
            content.extend(chunk)
            digest.update(chunk)
            if len(content) > maximum:
                raise ArtifactPublicationError("artifact read exceeded its safety bound")
    if len(content) != expected_size or digest.hexdigest() != expected_sha256:
        raise ArtifactPublicationError("artifact bytes failed checksum verification")
    return bytes(content)


@dataclass(frozen=True)
class ExecutionPolicy:
    """The non-negotiable limits and allowlists for a Docker invocation."""

    allowed_image_digests: tuple[str, ...] = (
        "tiktok2026:test@sha256:" + "0" * 64,
    )
    allowed_module_prefix: str = "tiktok2026.experiment."
    max_command_arguments: int = 32
    max_timeout_seconds: int = 7 * 24 * 60 * 60
    max_memory_bytes: int = 1 << 40
    max_cpus: float = 64.0
    max_gpu_count: int = 16
    disk_bytes: int = 1 << 30
    artifact_quota_bytes: int = 1 << 30
    max_output_bytes: int = 20_000
    pids_limit: int = 256
    termination_grace_seconds: float = 2.0
    container_user: str = "65532:65532"
    environment: tuple[tuple[str, str], ...] = (
        ("HOME", "/tmp"),
        ("LANG", "C.UTF-8"),
        ("PYTHONUNBUFFERED", "1"),
        ("PYTHONPATH", "/workspace/src"),
        ("TMPDIR", "/tmp"),
    )


DEFAULT_POLICY = ExecutionPolicy()
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:\-]*@sha256:[0-9a-f]{64}$")
_MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_SAFE_ARGUMENT_RE = re.compile(r"^[A-Za-z0-9_.:/=@%+,\-]+$")
_SAFE_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SAFE_USER_RE = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")
_READ_CHUNK_BYTES = 8192
_CONTAINER_PATH_ROOTS = ("/workspace", "/dataset", "/output", "/tmp")


def _resolved_directory(path: Path, name: str) -> Path:
    if not path.is_absolute():
        raise ExecutionPolicyError(f"{name} mount must be an absolute path")
    if path.is_symlink():
        raise ExecutionPolicyError(f"{name} mount must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ExecutionPolicyError(f"{name} mount is not accessible") from error
    if not resolved.is_dir():
        raise ExecutionPolicyError(f"{name} mount must be a directory")
    if any(character in str(resolved) for character in ("\x00", "\n", ",")):
        raise ExecutionPolicyError(f"{name} mount contains an unsafe character")
    return resolved


def _is_safe_container_path(value: str) -> bool:
    return any(value == root or value.startswith(f"{root}/") for root in _CONTAINER_PATH_ROOTS)


def _authorized_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and "\\" not in value
        and all(part not in ("", ".", "..") for part in path.parts)
    )


def validate_dataset_view(
    request: ExecutionRequest,
    view: DatasetView,
    final_test_authorizer: FinalTestAuthorizationAdapter | None = None,
) -> Path:
    """Verify that a controller-provided view contains exactly authorized files."""

    dataset = _resolved_directory(view.path, "dataset view")
    if not view.manifest_id or not _SHA256_RE.fullmatch(view.manifest_sha256):
        raise ExecutionPolicyError("dataset view provenance is invalid")
    request_dataset = _resolved_directory(request.dataset_path, "dataset request")
    if dataset == request_dataset:
        # The request path is permitted only when the controller has explicitly
        # described every file in that directory through the view.
        pass
    elif request_dataset in dataset.parents or dataset in request_dataset.parents:
        raise ExecutionPolicyError("dataset view must not overlap the request dataset")

    authorized = view.authorized_files
    if not authorized or len(set(authorized)) != len(authorized):
        raise ExecutionPolicyError("dataset view must contain a unique authorized file list")
    for relative in authorized:
        if not _authorized_relative_path(relative):
            raise ExecutionPolicyError("dataset view contains an unsafe file name")
        file_path = dataset / relative
        try:
            resolved = file_path.resolve(strict=True)
        except OSError as error:
            raise ExecutionPolicyError("authorized dataset file is missing") from error
        if resolved.parent != dataset and dataset not in resolved.parents:
            raise ExecutionPolicyError("authorized dataset file escapes the view")
        if not resolved.is_file() or file_path.is_symlink():
            raise ExecutionPolicyError("authorized dataset entries must be regular files")

    found: set[str] = set()
    for entry in dataset.rglob("*"):
        relative = entry.relative_to(dataset).as_posix()
        if entry.is_symlink():
            raise ExecutionPolicyError("dataset view must not contain symlinks")
        if entry.is_file():
            found.add(relative)
    if found != set(authorized):
        raise ExecutionPolicyError("dataset view contains files outside its authorization")

    if view.test_files and (
        final_test_authorizer is None or not final_test_authorizer.authorize(request, view)
    ):
        raise ExecutionPolicyError("test files require explicit final-test authorization")
    return dataset


def validate_source_identity(request: ExecutionRequest) -> None:
    """Validate the identity supplied by the repository/registration seam.

    The executor does not resolve commits itself: the repository boundary owns
    registration.  It does reject abbreviated or malformed identities before a
    container can be started.
    """

    if not _COMMIT_RE.fullmatch(request.source_commit):
        raise ExecutionPolicyError("source_commit must be a full lowercase Git commit SHA")


def validate_execution_request(
    request: ExecutionRequest,
    policy: ExecutionPolicy = DEFAULT_POLICY,
    dataset_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    validate_source_identity(request)
    if (
        not _IMAGE_DIGEST_RE.fullmatch(request.image)
        or request.image not in policy.allowed_image_digests
    ):
        raise ExecutionPolicyError("image must be an allowlisted immutable digest")
    if not request.command:
        raise ExecutionPolicyError("command must not be empty")
    if len(request.command) > policy.max_command_arguments:
        raise ExecutionPolicyError("command has too many arguments")
    if request.command[0] not in ("python", "python3") or request.command[1:2] != ("-m",):
        raise ExecutionPolicyError("command must use the approved Python module template")
    if len(request.command) < 3:
        raise ExecutionPolicyError("command must specify a module")
    module = request.command[2]
    if (
        not _MODULE_RE.fullmatch(module)
        or not module.startswith(policy.allowed_module_prefix)
    ):
        raise ExecutionPolicyError("command module is not allowlisted")
    for argument in request.command[3:]:
        safe_flag = argument.startswith("--") and "=" in argument
        if argument.startswith(("--data-root=", "--data-manifest=")):
            raise ExecutionPolicyError("data root and manifest are executor-controlled")
        if (
            not argument
            or not _SAFE_ARGUMENT_RE.fullmatch(argument)
            or ".." in argument
            or argument.startswith(("~", "-")) and not safe_flag
        ):
            # Flags are accepted only as --name=value.  This prevents an
            # argument from becoming a second command or an interpreter flag.
            raise ExecutionPolicyError("command contains an unsafe argument")
        if argument.startswith("/") and not _is_safe_container_path(argument):
            raise ExecutionPolicyError("command path is outside the container sandboxes")
        if safe_flag:
            flag_value = argument.partition("=")[2]
            if flag_value.startswith("/") and not _is_safe_container_path(flag_value):
                raise ExecutionPolicyError("command path is outside the container sandboxes")

    if request.timeout_seconds > policy.max_timeout_seconds:
        raise ExecutionPolicyError("timeout exceeds the execution policy")
    if request.memory_bytes > policy.max_memory_bytes:
        raise ExecutionPolicyError("memory request exceeds the execution policy")
    if request.cpus > policy.max_cpus:
        raise ExecutionPolicyError("CPU request exceeds the execution policy")
    if request.gpu_count > policy.max_gpu_count:
        raise ExecutionPolicyError("GPU request exceeds the execution policy")
    if (
        policy.disk_bytes <= 0
        or policy.artifact_quota_bytes <= 0
        or policy.max_output_bytes <= 0
        or policy.pids_limit <= 0
    ):
        raise ExecutionPolicyError("execution policy contains an invalid limit")
    if not _SAFE_USER_RE.fullmatch(policy.container_user):
        raise ExecutionPolicyError("container user must be a non-root numeric uid/gid")
    for name, value in policy.environment:
        if not _SAFE_ENV_NAME_RE.fullmatch(name) or "\x00" in value or "\n" in value:
            raise ExecutionPolicyError("environment is not in the safe allowlist")

    source = _resolved_directory(request.source_path, "source")
    dataset = _resolved_directory(dataset_path or request.dataset_path, "dataset")
    output = _resolved_directory(request.output_path, "artifact")
    mounts = (source, dataset, output)
    for index, mount in enumerate(mounts):
        for other in mounts[index + 1 :]:
            if mount == other or mount in other.parents or other in mount.parents:
                raise ExecutionPolicyError("source, dataset, and artifact mounts must be distinct")
    if not os.access(output, os.W_OK) or not output.stat().st_mode & 0o002:
        raise ExecutionPolicyError("artifact mount is not writable")
    if any(output.iterdir()):
        raise ExecutionPolicyError("artifact mount must be a fresh empty directory")
    return source, dataset, output


def container_name(execution_id: str) -> str:
    """Return a deterministic Docker-safe name without exposing caller input."""

    return f"tiktok2026-{hashlib.sha256(execution_id.encode()).hexdigest()[:32]}"


def build_docker_command(
    request: ExecutionRequest,
    policy: ExecutionPolicy = DEFAULT_POLICY,
    dataset_view: DatasetView | None = None,
    final_test_authorizer: FinalTestAuthorizationAdapter | None = None,
) -> tuple[str, ...]:
    if dataset_view is None:
        raise ExecutionPolicyError("a controller-provided dataset view is required")
    dataset = validate_dataset_view(request, dataset_view, final_test_authorizer)
    source, _, output = validate_execution_request(request, policy, dataset)
    command = [
        "docker",
        "run",
        "--name",
        container_name(request.execution_id),
        "--network=none",
        "--read-only",
        "--security-opt=no-new-privileges:true",
        "--cap-drop=ALL",
        "--user",
        policy.container_user,
        "--pids-limit",
        str(policy.pids_limit),
        "--cpus",
        str(request.cpus),
        "--memory",
        str(request.memory_bytes),
        "--storage-opt",
        f"size={policy.disk_bytes}",
        "--mount",
        f"type=bind,source={source},target=/workspace,readonly,bind-recursive=disabled",
        "--mount",
        f"type=bind,source={dataset},target=/dataset,readonly,bind-recursive=disabled",
        "--mount",
        f"type=bind,source={output},target=/output,rw,bind-recursive=disabled",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=67108864",
        "--workdir",
        "/workspace",
        "--label",
        f"tiktok2026.source_commit={request.source_commit}",
        "--entrypoint",
        request.command[0],
    ]
    for name, value in policy.environment:
        command.extend(("--env", f"{name}={value}"))
    command.append("--env")
    command.append(f"TIKTOK2026_SOURCE_COMMIT={request.source_commit}")
    if request.gpu_count:
        command.extend(("--gpus", str(request.gpu_count)))
    command.extend(
        (
            request.image,
            *request.command[1:],
            "--data-root=/dataset",
            "--data-manifest=/dataset/manifest.json",
        )
    )
    return tuple(command)


class BoundedOutputCapture:
    def __init__(self, path: Path, maximum: int) -> None:
        self.path = path
        self.maximum = maximum
        self.size = 0
        self.truncated = False
        self._digest = hashlib.sha256()
        self._file = path.open("wb")

    async def read(self, reader: asyncio.StreamReader) -> None:
        try:
            while chunk := await reader.read(_READ_CHUNK_BYTES):
                remaining = self.maximum - self.size
                if remaining > 0:
                    accepted = chunk[:remaining]
                    self._file.write(accepted)
                    self._digest.update(accepted)
                    self.size += len(accepted)
                if len(chunk) > max(remaining, 0):
                    self.truncated = True
        finally:
            self._file.close()

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def text(self) -> str:
        return self.path.read_bytes().decode(errors="replace")

    def close(self) -> None:
        self._file.close()


async def terminate_process_group(process: asyncio.subprocess.Process, grace: float) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=grace)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


async def _run_docker_control(arguments: tuple[str, ...]) -> None:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *arguments,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin"},
        start_new_session=True,
    )
    try:
        return_code = await asyncio.wait_for(process.wait(), timeout=10.0)
    except TimeoutError:
        await terminate_process_group(process, grace=1.0)
        raise RuntimeError(f"Docker control command timed out: {arguments[0]}") from None
    if return_code != 0:
        raise RuntimeError(f"Docker control command failed: {arguments[0]}")


class DockerContainerLifecycle:
    async def kill(self, name: str) -> None:
        await _run_docker_control(("kill", name))

    async def remove(self, name: str) -> None:
        await _run_docker_control(("rm", "--force", name))


def _directory_size(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        try:
            total += entry.stat().st_size
        except OSError:
            continue
    return total


async def monitor_artifact_quota(
    path: Path,
    baseline_bytes: int,
    quota_bytes: int,
    stop: asyncio.Event,
    process: asyncio.subprocess.Process,
    grace: float,
) -> bool:
    while not stop.is_set():
        if _directory_size(path) - baseline_bytes > quota_bytes:
            await terminate_process_group(process, grace)
            return True
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.1)
        except TimeoutError:
            continue
    return False


class DockerExecutor:
    def __init__(
        self,
        policy: ExecutionPolicy = DEFAULT_POLICY,
        publisher: ArtifactPublisher | None = None,
        dataset_provider: DatasetViewProvider | None = None,
        source_verifier: SourceIdentityVerifier | None = None,
        final_test_authorizer: FinalTestAuthorizationAdapter | None = None,
        lifecycle: ContainerLifecycle | None = None,
    ) -> None:
        self.policy = policy
        self.publisher = publisher
        self.dataset_provider = dataset_provider
        self.source_verifier = source_verifier
        self.final_test_authorizer = final_test_authorizer
        self.lifecycle = lifecycle or DockerContainerLifecycle()

    def _publish(
        self, execution_id: str, kind: str, path: Path, sha256: str, size_bytes: int
    ) -> str:
        publisher = self.publisher
        if publisher is None:
            raise ArtifactPublicationError("an authoritative artifact publisher is required")
        try:
            return publisher.publish(
                execution_id=execution_id,
                kind=kind,
                path=path,
                sha256=sha256,
                size_bytes=size_bytes,
            )
        except Exception as error:
            raise ArtifactPublicationError(f"could not publish {kind} artifact") from error

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        if self.publisher is None:
            raise ArtifactPublicationError("an authoritative artifact publisher is required")
        if self.dataset_provider is None:
            raise ExecutionPolicyError("a controller-provided dataset view is required")
        if self.source_verifier is None:
            raise ExecutionPolicyError("an authoritative source verifier is required")
        try:
            dataset_view = self.dataset_provider.provide(request)
        except Exception as error:
            raise ExecutionPolicyError("dataset view could not be provided") from error
        try:
            dataset = validate_dataset_view(request, dataset_view, self.final_test_authorizer)
            self.source_verifier.verify(request)
            _, _, output = validate_execution_request(request, self.policy, dataset)
            return await self._execute_materialized(request, dataset_view, dataset, output)
        finally:
            self.dataset_provider.cleanup(dataset_view)

    async def _execute_materialized(
        self,
        request: ExecutionRequest,
        dataset_view: DatasetView,
        dataset: Path,
        output: Path,
    ) -> ExecutionResult:
        start = time.monotonic()
        prefix = hashlib.sha256(request.execution_id.encode()).hexdigest()[:16]
        name = container_name(request.execution_id)
        baseline_bytes = _directory_size(output)
        docker_command = build_docker_command(
            request,
            self.policy,
            dataset_view,
            self.final_test_authorizer,
        )
        stdout_capture = BoundedOutputCapture(
            output / f"{prefix}.stdout.log", self.policy.max_output_bytes
        )
        stderr_capture = BoundedOutputCapture(
            output / f"{prefix}.stderr.log", self.policy.max_output_bytes
        )
        process: asyncio.subprocess.Process | None = None
        container_started: float | None = None
        timed_out = False
        quota_exceeded = False
        launch_error: str | None = None
        cleanup_errors: list[str] = []
        monitor: asyncio.Task[bool] | None = None
        monitor_stop = asyncio.Event()

        async def cleanup_container() -> None:
            for operation in (self.lifecycle.kill, self.lifecycle.remove):
                try:
                    await operation(name)
                except Exception as error:
                    cleanup_errors.append(f"{type(error).__name__}: {error}")

        try:
            try:
                process = await asyncio.create_subprocess_exec(
                    *docker_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={"PATH": "/usr/bin:/bin"},
                    start_new_session=True,
                )
            except OSError as error:
                launch_error = f"{type(error).__name__}: {error}"
            if process is not None:
                container_started = time.monotonic()
                assert process.stdout is not None
                assert process.stderr is not None
                readers = (
                    asyncio.create_task(stdout_capture.read(process.stdout)),
                    asyncio.create_task(stderr_capture.read(process.stderr)),
                )
                monitor = asyncio.create_task(
                    monitor_artifact_quota(
                        output,
                        baseline_bytes,
                        self.policy.artifact_quota_bytes,
                        monitor_stop,
                        process,
                        self.policy.termination_grace_seconds,
                    )
                )
                try:
                    await asyncio.wait_for(process.wait(), timeout=request.timeout_seconds)
                except TimeoutError:
                    timed_out = True
                    await terminate_process_group(process, self.policy.termination_grace_seconds)
                    await cleanup_container()
                except asyncio.CancelledError:
                    await terminate_process_group(process, self.policy.termination_grace_seconds)
                    raise
                finally:
                    monitor_stop.set()
                    quota_exceeded = await monitor
                    await asyncio.gather(*readers)
        except asyncio.CancelledError:
            if process is not None:
                await terminate_process_group(process, self.policy.termination_grace_seconds)
                await cleanup_container()
            raise
        except Exception:
            if process is not None:
                await terminate_process_group(process, self.policy.termination_grace_seconds)
                await cleanup_container()
            raise
        finally:
            # A failed process creation must not leave the capture files open.
            if process is None:
                stdout_capture.close()
                stderr_capture.close()

        container_elapsed = (
            time.monotonic() - container_started if container_started is not None else 0.0
        )
        try:
            if process is not None and not timed_out and not quota_exceeded:
                try:
                    await self.lifecycle.remove(name)
                except Exception as error:
                    cleanup_errors.append(
                        f"container removal {type(error).__name__}: {error}"
                    )
                    await cleanup_container()
        except asyncio.CancelledError:
            if process is not None:
                await terminate_process_group(process, self.policy.termination_grace_seconds)
                await cleanup_container()
            raise

        elapsed = time.monotonic() - start
        if process is None:
            returncode = 127
        else:
            returncode = process.returncode if process.returncode is not None else -1
        gpu_hours = (
            container_elapsed * request.gpu_count / 3600.0
            if process is not None and request.gpu_count
            else 0.0
        )
        artifact_bytes = max(0, _directory_size(output) - baseline_bytes)
        quota_exceeded = quota_exceeded or artifact_bytes > self.policy.artifact_quota_bytes
        if quota_exceeded and not timed_out:
            if process is not None and process.returncode is None:
                await terminate_process_group(process, self.policy.termination_grace_seconds)
            await cleanup_container()
        cleanup_failed = bool(cleanup_errors)
        exit_code = (
            -1
            if timed_out
            else 1
            if quota_exceeded or cleanup_failed
            else int(returncode)
        )
        stdout_id = self._publish(
            request.execution_id,
            "stdout",
            stdout_capture.path,
            stdout_capture.sha256,
            stdout_capture.size,
        )
        stderr_id = self._publish(
            request.execution_id,
            "stderr",
            stderr_capture.path,
            stderr_capture.sha256,
            stderr_capture.size,
        )
        evidence = {
            "command": list(request.command),
            "elapsed_seconds": elapsed,
            "exit_code": exit_code,
            "image": request.image,
            "resolved_image_digest": request.image,
            "container_name": name,
            "source_commit": request.source_commit,
            "dataset_manifest_id": dataset_view.manifest_id,
            "dataset_manifest_sha256": dataset_view.manifest_sha256,
            "dataset_authorized_files": list(dataset_view.authorized_files),
            "container_mounts": {
                "source": {"target": "/workspace", "read_only": True},
                "dataset": {"target": "/dataset", "read_only": True},
                "artifacts": {"target": "/output", "read_only": False},
            },
            "memory_limit_bytes": request.memory_bytes,
            "cpu_limit": request.cpus,
            "gpu_count": request.gpu_count,
            "resource_measurement_basis": "allocated_limits_only",
            "allocated_gpu_hours": gpu_hours,
            "measured_gpu_hours": None,
            "measured_peak_cpu": None,
            "measured_peak_memory_bytes": None,
            "disk_limit_bytes": self.policy.disk_bytes,
            "artifact_quota_bytes": self.policy.artifact_quota_bytes,
            "artifact_output_bytes": artifact_bytes,
            "stdout_bytes": stdout_capture.size,
            "stderr_bytes": stderr_capture.size,
            "stdout_truncated": stdout_capture.truncated,
            "stderr_truncated": stderr_capture.truncated,
            "environment_allowlist": [name for name, _ in self.policy.environment],
        }
        if launch_error is not None:
            evidence["launch_error"] = launch_error
        if quota_exceeded:
            evidence["failure_evidence"] = "artifact output quota exceeded"
        if cleanup_errors:
            evidence["container_cleanup_errors"] = cleanup_errors
        evidence_path = output / f"{prefix}.resource-evidence.json"
        evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
        evidence_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        evidence_id = self._publish(
            request.execution_id,
            "resource-evidence",
            evidence_path,
            evidence_digest,
            evidence_path.stat().st_size,
        )
        combined_evidence = (
            f"{stdout_capture.text()}\n{stderr_capture.text()}\n{launch_error or ''}"
        )
        failure = (
            None
            if exit_code == 0
            else classify_failure(exit_code, combined_evidence, timed_out)
        )
        return ExecutionResult(
            execution_id=request.execution_id,
            experiment_id=request.experiment_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=exit_code,
            elapsed_seconds=elapsed,
            gpu_hours=gpu_hours,
            artifact_ids=(stdout_id, stderr_id, evidence_id),
            failure_kind=failure,
        )
