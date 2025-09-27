
import logging
from unittest.mock import MagicMock, call, patch

import pytest
from omegaconf import OmegaConf

from img_search.utils import logging as logging_utils


@pytest.mark.parametrize(
    "level_input, expected_output",
    [
        ("INFO", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("WaRnInG", logging.WARNING),
        (logging.ERROR, logging.ERROR),
    ],
)
def test_resolve_log_level_valid(level_input, expected_output):
    """Test that valid log level names and integers are resolved correctly."""
    assert logging_utils._resolve_log_level(level_input, "test_module") == expected_output


@pytest.mark.parametrize("invalid_level", ["INVALID_LEVEL", None, object()])
def test_resolve_log_level_invalid(invalid_level):
    """Test that invalid log levels raise a ValueError."""
    with pytest.raises(ValueError, match="Invalid log level"):
        logging_utils._resolve_log_level(invalid_level, "test_module")


@patch("img_search.utils.logging.rich.print")
def test_print_config(mock_rich_print: MagicMock):
    """Test that the config is printed correctly, respecting the with_logging_cfg flag."""
    cfg = OmegaConf.create({
        "key": "value",
        "logging": {"level": "INFO"}
    })

    # Case 1: with_logging_cfg = False (default)
    logging_utils.print_config(cfg)
    mock_rich_print.assert_called_once()
    panel = mock_rich_print.call_args[0][0]
    syntax = panel.renderable
    assert "logging:" not in syntax.code
    assert "key: value" in syntax.code

    # Case 2: with_logging_cfg = True
    mock_rich_print.reset_mock()
    logging_utils.print_config(cfg, with_logging_cfg=True)
    mock_rich_print.assert_called_once()
    panel = mock_rich_print.call_args[0][0]
    syntax = panel.renderable
    assert "logging:" in syntax.code
    assert "level: INFO" in syntax.code


@patch("img_search.utils.logging.logger")
@patch("img_search.utils.logging.logging.basicConfig")
def test_setup_logger(mock_basic_config: MagicMock, mock_loguru_logger: MagicMock):
    """Verify that setup_logger configures loguru and the standard logging bridge."""
    logging_cfg = OmegaConf.create({
        "handlers": [
            {"sink": "sys.stderr", "level": "INFO"},
            {"sink": "rich", "level": "DEBUG"},
        ],
        "module_levels": {
            "my_app": "DEBUG",
            "third_party": "WARNING",
        },
    })

    logging_utils.setup_logger(logging_cfg)

    # Check that loguru is configured
    mock_loguru_logger.remove.assert_called_once()
    assert mock_loguru_logger.add.call_count == 2
    mock_loguru_logger.add.assert_any_call(logging_utils.sys.stderr, level="INFO")
    # The second call is to RichHandler, which is harder to assert directly

    # Check that the standard logging bridge is set up
    mock_basic_config.assert_called_once()
    assert "handlers" in mock_basic_config.call_args.kwargs
    handler = mock_basic_config.call_args.kwargs["handlers"][0]
    assert isinstance(handler, logging_utils._InterceptHandler)

    # Check that module levels are set
    assert logging.getLogger("my_app").level == logging.DEBUG
    assert logging.getLogger("third_party").level == logging.WARNING
