from __future__ import annotations

import asyncio
import time

from tiktok2026.contracts import ExecutionRequest, ExecutionResult
from tiktok2026.execution.failures import classify_failure


def build_docker_command(request: ExecutionRequest) -> tuple[str, ...]:
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cpus",
        str(request.cpus),
        "--memory",
        str(request.memory_bytes),
        "--mount",
        f"type=bind,source={request.source_path.resolve()},target=/workspace,readonly",
        "--mount",
        f"type=bind,source={request.dataset_path.resolve()},target=/dataset,readonly",
        "--mount",
        f"type=bind,source={request.output_path.resolve()},target=/output",
        "--workdir",
        "/workspace",
    ]
    if request.gpu_count:
        command.extend(("--gpus", str(request.gpu_count)))
    command.extend((request.image, *request.command))
    return tuple(command)


class DockerExecutor:
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        start = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *build_docker_command(request),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=request.timeout_seconds
            )
        except TimeoutError:
            timed_out = True
            process.kill()
            stdout, stderr = await process.communicate()
        exit_code = -1 if timed_out else int(process.returncode or 0)
        evidence = (stdout + stderr).decode(errors="replace")[-20_000:]
        failure = None if exit_code == 0 else classify_failure(exit_code, evidence, timed_out)
        return ExecutionResult(
            execution_id=request.execution_id,
            experiment_id=request.experiment_id,
            source_commit=request.source_commit,
            command=request.command,
            exit_code=exit_code,
            elapsed_seconds=time.monotonic() - start,
            gpu_hours=0.0,
            failure_kind=failure,
        )
