from tiktok2026.contracts import FailureKind


def classify_failure(exit_code: int, evidence: str, timed_out: bool) -> FailureKind:
    text = evidence.lower()
    if timed_out:
        return FailureKind.TIMEOUT
    if "cuda out of memory" in text:
        return FailureKind.CUDA_OOM
    if "memoryerror" in text or "cannot allocate memory" in text:
        return FailureKind.CPU_OOM
    if "no space left" in text or "artifact output quota exceeded" in text:
        return FailureKind.DISK
    if exit_code == 137:
        return FailureKind.CPU_OOM
    if "nan" in text or "diverg" in text:
        return FailureKind.NAN_DIVERGENCE
    if "syntaxerror" in text or "modulenotfounderror" in text or "importerror" in text:
        return FailureKind.SYNTAX_IMPORT
    if "no such file" in text or "filenotfounderror" in text:
        return FailureKind.MISSING_PATH
    if exit_code != 0:
        return FailureKind.DEPENDENCY_ENVIRONMENT
    raise ValueError("successful execution has no failure classification")
