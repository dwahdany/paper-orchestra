"""Logging setup for entry points.  Library modules only call ``getLogger``."""

from __future__ import annotations

import logging

LOGGER_NAME = "paper_orchestra"


def setup_logging(level: int | str = logging.INFO, *, rich: bool = True) -> None:
    """Send ``paper_orchestra.*`` logs to the console.  Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler: logging.Handler
    if rich:
        from rich.logging import RichHandler

        handler = RichHandler(
            show_path=False, rich_tracebacks=False, markup=False, log_time_format="%H:%M:%S"
        )
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
    logger.addHandler(handler)
