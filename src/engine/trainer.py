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
        val_evaluator: Evaluator,
        config: Dict[str, Any],
        logger: Any,
        test_evaluator: Optional[Evaluator] = None,
        metric_tracker: Optional[MetricTracker] = None,
        resume_checkpoint: Optional[str] = None
    ):
        self.model = model
        self.dataset = dataset
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.config = config
        self.logger = logger
        self.metric_tracker = metric_tracker
        self.resume_checkpoint = resume_checkpoint

        self.train_cfg = config.get("train", {})
        self.batch_size = self.train_cfg.get("batch_size", 256)
        self.num_epochs = self.train_cfg.get("num_epochs", 20)
        self.lr = self.train_cfg.get("learning_rate", 0.001)
        self.weight_decay = self.train_cfg.get("weight_decay", 1e-5)
        self.patience = self.train_cfg.get("early_stopping_patience", 5)
        raw_monitor = self.train_cfg.get("monitor_metric", "ndcg@10")
        self.monitor_metric = raw_monitor.replace("val_", "")
        self.save_dir = self.train_cfg.get("save_dir", "checkpoints")
        self.checkpoint_name = self.train_cfg.get("checkpoint_name", "best_model.pt")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.train_cfg.get("device") == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.criterion = nn.BCELoss()
        self.optimizer = self._build_optimizer()

        os.makedirs(self.save_dir, exist_ok=True)

        self.initial_best_score = -1.0
        if self.resume_checkpoint:
            self._load_resume_checkpoint()

    def _load_resume_checkpoint(self) -> None:
        if not os.path.exists(self.resume_checkpoint):
            self.logger.warning(f"Resume checkpoint not found at {self.resume_checkpoint}. Training from scratch.")
            return

        self.logger.info(f"Resuming training from checkpoint: {self.resume_checkpoint}")
        ckpt = torch.load(self.resume_checkpoint, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
                self.logger.info(f"Loaded optimizer state with fine-tuning learning rate: {self.lr}")
            except Exception as e:
                self.logger.warning(f"Could not load optimizer state ({e}). Fresh optimizer will be used.")

        prev_metrics = ckpt.get("val_metrics", ckpt.get("metrics", {}))
        self.initial_best_score = prev_metrics.get(self.monitor_metric, -1.0)
        prev_epoch = ckpt.get("epoch", 0)
        self.logger.info(f"Loaded checkpoint from previous Epoch {prev_epoch}. Baseline {self.monitor_metric.upper()}: {self.initial_best_score:.4f}")

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

        best_metric_val = getattr(self, "initial_best_score", -1.0)
        best_epoch = -1
        patience_counter = 0

        best_checkpoint_path = os.path.join(self.save_dir, self.checkpoint_name)
        last_checkpoint_path = os.path.join(self.save_dir, "last_model.pt")

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_metrics = self.val_evaluator.evaluate(self.model)
            elapsed = time.time() - epoch_start

            metric_str = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(f"Epoch {epoch:02d}/{self.num_epochs:02d} [{elapsed:.1f}s] - Loss: {train_loss:.4f} | Val {metric_str}")

            if self.metric_tracker:
                self.metric_tracker.log_epoch(epoch, train_loss, val_metrics)

            current_score = val_metrics.get(self.monitor_metric, 0.0)

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
                    "val_metrics": val_metrics,
                    "config": self.config
                }, best_checkpoint_path)
                self.logger.info(f"--> Saved best checkpoint (Val {self.monitor_metric}={best_metric_val:.4f}) to {best_checkpoint_path}")
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
        self.logger.info(f"Training completed in {total_time / 60:.2f} mins. Best Val {self.monitor_metric}: {best_metric_val:.4f} at epoch {best_epoch}.")

        # Final Evaluation on Held-out Test Set
        final_test_metrics = {}
        if self.test_evaluator is not None and os.path.exists(best_checkpoint_path):
            self.logger.info("=" * 80)
            self.logger.info(f"RUNNING FINAL UNBIASED TEST EVALUATION (Using Best Checkpoint from Epoch {best_epoch})...")
            best_ckpt = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(best_ckpt["model_state_dict"])
            
            final_test_metrics = self.test_evaluator.evaluate(self.model)
            test_metric_str = " | ".join([f"{k.upper()}: {v:.4f}" for k, v in final_test_metrics.items()])
            self.logger.info(f"FINAL TEST SET METRICS: {test_metric_str}")
            self.logger.info("=" * 80)

            if self.metric_tracker:
                self.metric_tracker.log_test(final_test_metrics)

        return {
            "best_epoch": best_epoch,
            "best_val_metric": best_metric_val,
            "final_test_metrics": final_test_metrics,
            "best_checkpoint": best_checkpoint_path
        }
