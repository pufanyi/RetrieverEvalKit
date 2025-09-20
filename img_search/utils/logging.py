import sys

from loguru import logger
from rich.logging import RichHandler


def setup_logger():
    """Set up the logger with RichHandler."""
    logger.remove()
    logger.add(
        RichHandler(rich_tracebacks=True, show_path=False, show_time=False),
        format="{message}",
        level="INFO",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
    logger.add(
        sys.stderr,
        level="ERROR",
        format="<red>{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}</red>",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
