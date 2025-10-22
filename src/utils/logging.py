from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


_LOGGER_INITIALIZED = False


def configure_logging(level: str = "INFO") -> None:
    global _LOGGER_INITIALIZED
    if _LOGGER_INITIALIZED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    _LOGGER_INITIALIZED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if not _LOGGER_INITIALIZED:
        configure_logging()
    return logging.getLogger(name or "src")
