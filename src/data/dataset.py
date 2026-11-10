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

    def sample_negatives(self) -> None:
        """
        Fast vectorized dynamic negative sampling.
        Generates dynamic negative samples for the upcoming training epoch using NumPy.
        """
        num_pos = len(self.train_users)
        total_samples = num_pos * (1 + self.num_negatives)

        users = np.empty(total_samples, dtype=np.int64)
        items = np.empty(total_samples, dtype=np.int64)
        labels = np.zeros(total_samples, dtype=np.float32)

        # 1. Fill positive samples
        users[:num_pos] = self.train_users
        items[:num_pos] = self.train_items
        labels[:num_pos] = 1.0

        # 2. Vectorized negative sampling for all (u, neg) pairs
        users_neg = np.repeat(self.train_users, self.num_negatives)
        rand_items = np.random.randint(0, self.num_items, size=num_pos * self.num_negatives, dtype=np.int64)

        # Fast collision rejection: only adjust the ~1% colliding items
        for idx in range(len(users_neg)):
            u = int(users_neg[idx])
            neg = int(rand_items[idx])
            if neg in self.user_pos_items[u]:
                while neg in self.user_pos_items[u]:
                    neg = random.randint(0, self.num_items - 1)
                rand_items[idx] = neg

        users[num_pos:] = users_neg
        items[num_pos:] = rand_items

        # Convert once to contiguous PyTorch tensors for zero-copy DataLoader slicing
        self.users_tensor = torch.from_numpy(users)
        self.items_tensor = torch.from_numpy(items)
        self.labels_tensor = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.users_tensor[idx],
            self.items_tensor[idx],
            self.labels_tensor[idx]
        )
