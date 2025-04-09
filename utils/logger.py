"""
File: logger.py
Location: project/utils/logger.py

Description:
    Sets up logging and a TensorBoard writer using torch.utils.tensorboard.
    This module initializes:
      - A Python logger that logs to both the console and a log file.
      - A TensorBoard SummaryWriter for logging training metrics.

Usage:
    Import the setup functions/classes from this module in your main training script.
"""

import logging
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def setup_logger(log_dir="experiments/logs", log_file="train.log"):
    """
    Sets up a logger that writes log messages to both console and a file.

    Args:
        log_dir (str): Directory where the log file will be saved.
        log_file (str): Log file name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Create a custom logger
    logger = logging.getLogger("GAN_Logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create handlers: one for console and one for the file.
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_path)

    # Set level for handlers
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    logger.info(f"Logger is set up. Log file: {log_path}")
    return logger


def setup_tensorboard(log_dir="tensorboard"):
    """
    Sets up a TensorBoard SummaryWriter.

    Args:
        log_dir (str): Directory to store TensorBoard logs.

    Returns:
        SummaryWriter: A TensorBoard writer instance.
    """
    # Append current date-time to the log directory for versioning.
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = os.path.join(log_dir, current_time)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    return writer


# Example usage of the logger and TensorBoard writer:
if __name__ == "__main__":
    logger = setup_logger()
    writer = setup_tensorboard()
    logger.info("This is an info message for testing.")
    writer.add_scalar("Test/Example", 1.0, 0)
    writer.close()