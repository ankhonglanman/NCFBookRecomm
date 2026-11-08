"""
Project: Book-NCF
Module: src.data.dataset
Description: PyTorch Dataset for NCF with dynamic negative sampling per epoch.
"""

import random
from collections import defaultdict
from typing import Dict, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    """
    PyTorch Dataset implementing Dynamic Negative Sampling.
    Before every epoch, `sample_negatives()` is invoked to sample `num_negatives`
    unobserved items for every positive interaction.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        num_negatives: int = 4,
        seed: int = 42
    ):
        """
        Args:
            train_df: DataFrame with 'user_idx' and 'item_idx' columns.
            num_users: Total number of distinct users.
            num_items: Total number of distinct items.
            num_negatives: Ratio of negative samples per positive interaction.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.seed = seed

        # Extract unique positive pairs
        self.train_users = train_df["user_idx"].values
        self.train_items = train_df["item_idx"].values

        # Build user -> set of interacted items for O(1) membership check
        self.user_pos_items: Dict[int, Set[int]] = defaultdict(set)
        for u, i in zip(self.train_users, self.train_items):
            self.user_pos_items[int(u)].add(int(i))

        self.users: np.ndarray = np.array([], dtype=np.int64)
        self.items: np.ndarray = np.array([], dtype=np.int64)
        self.labels: np.ndarray = np.array([], dtype=np.float32)

        # Initial sample of negatives
        self.sample_negatives()

    def sample_negatives(self) -> None:
        """
        Generates dynamic negative samples for the upcoming training epoch.
        For each positive (u, i), samples `num_negatives` items not in user_pos_items[u].
        """
        users_list = []
        items_list = []
        labels_list = []

        num_pos = len(self.train_users)

        for idx in range(num_pos):
            u = int(self.train_users[idx])
            i = int(self.train_items[idx])

            # Positive sample
            users_list.append(u)
            items_list.append(i)
            labels_list.append(1.0)

            # Negative samples
            pos_set = self.user_pos_items[u]
            for _ in range(self.num_negatives):
                neg_item = random.randint(0, self.num_items - 1)
                while neg_item in pos_set:
                    neg_item = random.randint(0, self.num_items - 1)
                
                users_list.append(u)
                items_list.append(neg_item)
                labels_list.append(0.0)

        self.users = np.array(users_list, dtype=np.int64)
        self.items = np.array(items_list, dtype=np.int64)
        self.labels = np.array(labels_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.users[idx], dtype=torch.long),
            torch.tensor(self.items[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )
