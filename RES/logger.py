# logger.py
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = "RESouce_builder.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

def setup_logger(name: str = None, level=logging.INFO,reset=False) -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name (str, optional): Name of the logger. Defaults to None (root logger).
        level (int, optional): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.

    Logging Levels:
    - DEBUG      : Detailed information, for debugging.
    - INFO       : Confirmation that things are working as expected.
    - WARNING    : An indication that something unexpected happened.
    - ERROR      : More serious problem, the software has not been able to perform some function.
    - CRITICAL   : A serious error, indicating the program itself may be unable to continue running.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if reset:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
    if not logger.hasHandlers():  # Avoid duplicate handlers
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Rotating file handler (5MB per file, 3 backups)
        fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
