import logging
import os
from logging.handlers import RotatingFileHandler

def get_logger(name, log_file='logs/feature_engineer.log', level=logging.INFO):
    """
    Configures and returns a logger with console and file handlers.

    Args:
        name (str): Name of the logger (e.g., module name).
        log_file (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation (max 5MB, keep 3 backups)
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger