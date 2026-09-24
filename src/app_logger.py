import logging

from src.config import load_config
from src.logging_config import setup_logging


def initialize_logging():
    """
    Load logging configuration and initialize
    application logging.
    """

    config = load_config()

    setup_logging(
        config["logging"]
    )

    logger = logging.getLogger(
        __name__
    )

    logger.info(
        "Application logging initialized."
    )

    return logger