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

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
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


def _resolve_log_level(level: Any, module_name: str) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        upper_level = level.upper()
        resolved_level = getattr(logging, upper_level, None)
        if isinstance(resolved_level, int):
            return resolved_level

    raise ValueError(
        f"Invalid log level '{level}' for module '{module_name}'. "
        "Expected an int or standard logging level name."
    )


def setup_logger(logging_cfg: DictConfig):
    """Set up the logger based on the provided hydra config."""
    logger.remove()

    for handler_cfg in logging_cfg.handlers:
        handler_dict: dict[str, Any] = dict(handler_cfg)
        sink = handler_dict.pop("sink")

        if sink == "rich":
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
            actual_sink = sys.stderr if sink == "sys.stderr" else sink
            logger.add(actual_sink, **handler_dict)

    logging.basicConfig(
        level=logging.NOTSET,
        handlers=[_InterceptHandler()],
        force=True,
    )

    module_levels_cfg = logging_cfg.get("module_levels")
    if module_levels_cfg is None:
        module_levels: dict[str, Any] = {
            "sentence_transformers": "WARNING",
            "transformers": "WARNING",
            "datasets": "WARNING",
        }
    else:
        module_levels = OmegaConf.to_container(module_levels_cfg, resolve=True)
        if not isinstance(module_levels, dict):
            raise TypeError("logging.module_levels must be a mapping")

    for module_name, configured_level in module_levels.items():
        std_logger = logging.getLogger(module_name)
        std_logger.handlers.clear()
        std_logger.propagate = True
        std_logger.setLevel(_resolve_log_level(configured_level, module_name))
