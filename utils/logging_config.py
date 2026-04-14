import logging
import os

def setup_logging(log_level=logging.INFO, log_file='project.log'):
    """
    Configures the logging for the entire project.

    Parameters:
    - log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    - log_file (str): Path to the log file.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)

    c_handler.setLevel(log_level)
    f_handler.setLevel(log_level)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)