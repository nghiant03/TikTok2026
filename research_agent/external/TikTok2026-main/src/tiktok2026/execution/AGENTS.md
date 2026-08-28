# Execution Instructions

## Authority

Execution deterministically runs validated, committed source in the pinned project Docker image. It owns process lifecycle, timeouts, GPU allocation, resource telemetry, artifact capture, and failure evidence.

## Rules

- Execute only a registered Git commit from its assigned worktree and an approved typed command template.
- Mount datasets and protected benchmark inputs read-only. Mount only the experiment artifact directory writable.
- Enforce fidelity, timeout, CPU/RAM/GPU, disk, environment, network, and process limits before launch.
- Default network access during training and evaluation is disabled.
- Terminate the entire process group on timeout, cancellation, or budget exhaustion.
- Never classify an invalid execution as a scientific negative result.
- Persist stdout, stderr, exit status, image digest, command, environment allowlist, timing, GPU usage, peak resources, and artifact hashes.
- Agents do not receive arbitrary shell or Docker access.

## Dependencies and tests

Depend on contracts, pure policies, and injected artifact/resource backends. Never depend on LangGraph or agent code. Test timeout cleanup, cancellation, OOM/error classification evidence, read-only mounts, command allowlists, artifact limits, and budget denial.
