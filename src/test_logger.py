import logging

from src.app_logger import (
    initialize_logging,
)


initialize_logging()

logger = logging.getLogger(
    "test"
)

logger.debug(
    "This is a DEBUG message."
)

logger.info(
    "This is an INFO message."
)

logger.warning(
    "This is a WARNING message."
)

logger.error(
    "This is an ERROR message."
)