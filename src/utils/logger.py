"""
Project: Book-NCF
Module: src.utils.logger
Description: Structured logging and metrics history tracker for experiments.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List


def setup_logger(name: str = "BookNCF", log_dir: str = "logs", log_file: str = "experiment.log") -> logging.Logger:
    """
    Sets up a logger that outputs both to standard output and a file.

    Args:
        name: Name of the logger.
        log_dir: Directory where log file is stored.
        log_file: Name of the log file.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logger is invoked multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger


class MetricTracker:
    """
    Records training loss and evaluation metrics (HR@K, NDCG@K) across epochs.
    Allows exporting history to JSON for downstream visualization.
    """

    def __init__(self, output_path: str = "logs/training_history.json"):
        self.output_path = output_path
        self.history: List[Dict[str, Any]] = []

    def log_epoch(self, epoch: int, train_loss: float, metrics: Dict[str, float]) -> None:
        """
        Record a single epoch's outcome.
        """
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics
        }
        self.history.append(record)
        self.save()

    def save(self) -> None:
        """
        Saves current history into JSON format.
        """
        parent_dir = os.path.dirname(self.output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
