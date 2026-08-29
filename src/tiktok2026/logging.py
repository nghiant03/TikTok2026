from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def configure_logging(runtime_root: Path | None = None) -> None:
    logger.remove()

    level = os.getenv("TIKTOK2026_LOG_LEVEL", "INFO").upper()

    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )

    if runtime_root is not None:
        log_dir = runtime_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_dir / "tiktok2026.log",
            level=level,
            rotation="50 MB",
            retention="7 days",
            enqueue=True,
            backtrace=True,
            diagnose=False,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | {message}"
            ),
        )
