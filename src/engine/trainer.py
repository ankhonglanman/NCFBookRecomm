"""
Project: Book-NCF
Module: src.engine.trainer
Description: Model trainer with dynamic negative sampling per epoch,
             early stopping, and best model checkpointing.
"""

import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NCFDataset
from src.engine.evaluator import Evaluator
from src.utils.logger import MetricTracker


class Trainer:
    """
    Manages NCF model training pipeline:
    - Dynamic negative sampling per epoch
    - Mini-batch gradient descent
    - Leave-One-Out validation (HR@K, NDCG@K)
    - Early stopping and model checkpoint persistence
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: NCFDataset,
        evaluator: Evaluator,
        config: Dict[str, Any],
        logger: Any,
        metric_tracker: Optional[MetricTracker] = None
    ):
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.config = config
        self.logger = logger
        self.metric_tracker = metric_tracker

        self.train_cfg = config.get("train", {})
        self.batch_size = self.train_cfg.get("batch_size", 256)
        self.num_epochs = self.train_cfg.get("num_epochs", 20)
        self.lr = self.train_cfg.get("learning_rate", 0.001)
        self.weight_decay = self.train_cfg.get("weight_decay", 1e-5)
        self.patience = self.train_cfg.get("early_stopping_patience", 5)
        self.monitor_metric = self.train_cfg.get("monitor_metric", "ndcg@10")
        self.save_dir = self.train_cfg.get("save_dir", "checkpoints")
        self.checkpoint_name = self.train_cfg.get("checkpoint_name", "best_model.pt")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.train_cfg.get("device") == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = self._build_optimizer()

        os.makedirs(self.save_dir, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_type = self.train_cfg.get("optimizer", "adam").lower()
        if opt_type == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        elif opt_type == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_epoch(self, epoch: int) -> float:
        """Runs one training epoch after resampling negatives."""
        # 1. Dynamic Negative Sampling
        self.dataset.sample_negatives()

        # 2. Re-create DataLoader for newly sampled dataset
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda")
        )

        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{self.num_epochs:02d}", leave=False)
        for users, items, labels in pbar:
            users = users.to(self.device)
            items = items.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(users, items)
            loss = self.criterion(predictions, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    def fit(self) -> Dict[str, Any]:
        """Runs complete training workflow with evaluation and early stopping."""
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__} | Epochs: {self.num_epochs} | Batch size: {self.batch_size}")

        best_metric_val = -1.0
        best_epoch = -1
        patience_counter = 0

        best_checkpoint_path = os.path.join(self.save_dir, self.checkpoint_name)
        last_checkpoint_path = os.path.join(self.save_dir, "last_model.pt")

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on test set
            eval_metrics = self.evaluator.evaluate(self.model)
            elapsed = time.time() - epoch_start

            metric_str = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in eval_metrics.items()])
            self.logger.info(f"Epoch {epoch:02d}/{self.num_epochs:02d} [{elapsed:.1f}s] - Loss: {train_loss:.4f} | {metric_str}")

            if self.metric_tracker:
                self.metric_tracker.log_epoch(epoch, train_loss, eval_metrics)

            current_score = eval_metrics.get(self.monitor_metric, 0.0)

            # Check if this is the best epoch
            if current_score > best_metric_val:
                best_metric_val = current_score
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": eval_metrics,
                    "config": self.config
                }, best_checkpoint_path)
                self.logger.info(f"--> Saved best checkpoint ({self.monitor_metric}={best_metric_val:.4f}) to {best_checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered! No improvement for {self.patience} consecutive epochs.")
                    break

        # Save last model checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.config
        }, last_checkpoint_path)

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.2f} mins. Best {self.monitor_metric}: {best_metric_val:.4f} at epoch {best_epoch}.")

        return {
            "best_epoch": best_epoch,
            "best_metric_value": best_metric_val,
            "best_checkpoint": best_checkpoint_path
        }
