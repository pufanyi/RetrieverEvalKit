import logging
import sys
from typing import Any

import rich
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax


class _InterceptHandler(logging.Handler):
    """Route standard logging records through Loguru."""

    def emit(
        self, record: logging.LogRecord
    ) -> None:  # pragma: no cover - thin wrapper
        try:
            loguru_level = logger.level(record.levelname).name
        except ValueError:
            loguru_level = record.levelno

        logger.opt(depth=2, exception=record.exc_info).log(
            loguru_level, record.getMessage()
        )


def print_config(cfg: DictConfig, with_logging_cfg: bool = False):
    if with_logging_cfg:
        conf_yaml = OmegaConf.to_yaml(cfg).strip()
    else:
        conf_dict = OmegaConf.to_container(cfg, resolve=True)
        conf_dict.pop("logging")
        conf_yaml = OmegaConf.to_yaml(conf_dict).strip()

    rich.print(
        Panel.fit(
            Syntax(
                conf_yaml,
                "yaml",
                line_numbers=False,
                theme="dracula",
                padding=(1, 2),
            ),
            title="[bold magenta]Config[/bold magenta]",
            border_style="green",
        )
    )


def setup_logger(logging_cfg: DictConfig):
    """Set up the logger based on the provided hydra config."""
    logger.remove()

    # The config is now structured under 'handlers'
    for handler_cfg in logging_cfg.handlers:
        handler_dict: dict[str, Any] = dict(handler_cfg)
        sink = handler_dict.pop("sink")

        if sink == "rich":
            # Pop RichHandler specific keys to avoid passing them to logger.add
            rich_tracebacks = handler_dict.pop("rich_tracebacks", True)
            show_path = handler_dict.pop("show_path", False)
            show_time = handler_dict.pop("show_time", False)

            logger.add(
                RichHandler(
                    rich_tracebacks=rich_tracebacks,
                    show_path=show_path,
                    show_time=show_time,
                ),
                **handler_dict,
            )
        else:
            # For other sinks like sys.stderr
            actual_sink = sys.stderr if sink == "sys.stderr" else sink
            logger.add(
                actual_sink,
                **handler_dict,
            )

    logging.basicConfig(
        level=logging.NOTSET, handlers=[_InterceptHandler()], force=True
    )
