import logging
from pathlib import Path


def setup_logging(logging_config):
    """
    Configure application logging to both console and file.
    """

    log_level = logging_config.get("level", "INFO").upper()
    log_file = Path(logging_config["log_file"])
    log_format = logging_config.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Create logs directory
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(log_format)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Avoid duplicate handlers if setup_logging is called more than once
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger