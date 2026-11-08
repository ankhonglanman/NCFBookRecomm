"""
Project: Book-NCF
Module: src.engine.evaluator
Description: Batched evaluation of Hit Ratio (HR@K) and NDCG@K on test_negatives.
"""

from typing import List, Dict
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class Evaluator:
    """
    Evaluates recommendation performance using the Leave-One-Out ranking protocol.
    For each test user, evaluates 1 positive item against N negative items (e.g. 99 negatives).
    Computes Hit Ratio@K (HR@K) and Normalized Discounted Cumulative Gain@K (NDCG@K).
    """

    def __init__(
        self,
        test_negatives_path: str,
        top_k_list: List[int] = None,
        batch_size: int = 100,
        device: str = "cpu"
    ):
        """
        Args:
            test_negatives_path: Path to CSV containing [user_idx, positive_item_idx, neg_0, ...].
            top_k_list: List of K values to compute metrics for (e.g. [5, 10, 20]).
            batch_size: Number of test users processed in parallel.
            device: Computation device ("cuda" or "cpu").
        """
        if top_k_list is None:
            top_k_list = [5, 10, 20]

        self.test_negatives_path = test_negatives_path
        self.top_k_list = sorted(top_k_list)
        self.batch_size = batch_size
        self.device = torch.device(device)

        self._load_test_data()

    def _load_test_data(self) -> None:
        """Loads and pre-parses the test negatives table."""
        df = pd.read_csv(self.test_negatives_path)
        self.users = df["user_idx"].values
        self.pos_items = df["positive_item_idx"].values
        neg_cols = [c for c in df.columns if c.startswith("neg_")]
        self.neg_items = df[neg_cols].values  # shape: (num_users, num_negatives)
        self.num_users = len(self.users)

    @torch.no_grad()
    def evaluate(self, model: nn.Module) -> Dict[str, float]:
        """
        Runs evaluation across all test users and returns aggregated HR@K and NDCG@K.

        Args:
            model: PyTorch model (GMF, MLP, or NeuMF) to evaluate.

        Returns:
            Dictionary containing metrics, e.g. {"hr@5": 0.42, "ndcg@5": 0.31, ...}
        """
        model.eval()
        hits = {k: 0 for k in self.top_k_list}
        ndcgs = {k: 0.0 for k in self.top_k_list}

        num_neg = self.neg_items.shape[1]
        num_candidates = num_neg + 1  # 1 positive + N negatives

        for i in range(0, self.num_users, self.batch_size):
            end_idx = min(i + self.batch_size, self.num_users)
            current_batch_size = end_idx - i

            batch_users = self.users[i:end_idx]
            batch_pos = self.pos_items[i:end_idx]
            batch_negs = self.neg_items[i:end_idx]

            # Construct candidate items array: shape (current_batch_size, num_candidates)
            candidates = np.hstack([batch_pos[:, np.newaxis], batch_negs])

            # Flatten users and items for model scoring
            u_flat = np.repeat(batch_users, num_candidates)
            i_flat = candidates.flatten()

            u_tensor = torch.tensor(u_flat, dtype=torch.long, device=self.device)
            i_tensor = torch.tensor(i_flat, dtype=torch.long, device=self.device)

            scores = model(u_tensor, i_tensor).view(current_batch_size, num_candidates)

            # Positive item is at index 0 for each row in candidate list
            # Rank candidate items: highest score gets rank 1
            # Sort descending along candidate dimension
            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            
            # Find the rank of index 0 (the true positive item)
            # rank is 0-indexed initially
            pos_ranks = (sorted_indices == 0).nonzero()[:, 1].cpu().numpy()

            for rank in pos_ranks:
                for k in self.top_k_list:
                    if rank < k:
                        hits[k] += 1
                        ndcgs[k] += 1.0 / math.log2(rank + 2)  # (rank + 1) + 1 for 1-based log2(rank + 1)

        metrics = {}
        for k in self.top_k_list:
            metrics[f"hr@{k}"] = round(hits[k] / self.num_users, 4)
            metrics[f"ndcg@{k}"] = round(ndcgs[k] / self.num_users, 4)

        return metrics
