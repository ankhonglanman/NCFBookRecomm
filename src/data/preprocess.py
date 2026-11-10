"""
Project: Book-NCF
Module: src.data.preprocess
Description: K-core filtering, contiguous ID mapping, Leave-One-Out train/test split,
             and negative sampling for Leave-One-Out evaluation.
"""

import os
import pickle
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Set


class DataPreprocessor:
    """
    Handles preprocessing of raw rating records for Neural Collaborative Filtering:
    1. Rating binarization (implicit feedback)
    2. Iterative K-core filtering on users and books
    3. Re-indexing user and book IDs to contiguous 0-indexed integers
    4. Leave-One-Out splitting per user
    5. Test negative sampling (1 positive + N negatives per user)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        
        self.ratings_path = self.data_cfg.get("ratings_path", "data/raw/ratings.csv")
        self.processed_dir = self.data_cfg.get("processed_dir", "data/processed")
        self.train_path = self.data_cfg.get("train_path", os.path.join(self.processed_dir, "train.csv"))
        self.test_negatives_path = self.data_cfg.get("test_negatives_path", os.path.join(self.processed_dir, "test_negatives.csv"))
        self.mappings_path = self.data_cfg.get("mappings_path", os.path.join(self.processed_dir, "id_mappings.pkl"))
        
        self.user_col = self.data_cfg.get("user_col", "user_id")
        self.item_col = self.data_cfg.get("item_col", "book_id")
        self.rating_col = self.data_cfg.get("rating_col", "rating")
        self.timestamp_col = self.data_cfg.get("timestamp_col", "timestamp")
        
        self.min_user_ratings = self.data_cfg.get("min_user_ratings", 5)
        self.min_item_ratings = self.data_cfg.get("min_item_ratings", 5)
        self.min_rating_threshold = self.data_cfg.get("min_rating_threshold", 1.0)
        self.num_negatives_test = self.data_cfg.get("num_negatives_test", 99)
        self.seed = config.get("train", {}).get("seed", 42)

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw ratings from disk and converts to implicit feedback."""
        if not os.path.exists(self.ratings_path):
            raise FileNotFoundError(f"Ratings file not found at: {self.ratings_path}")
        
        df = pd.read_csv(self.ratings_path)
        print(f"Loaded raw ratings: {len(df):,} rows from {self.ratings_path}")

        # Ensure required columns exist
        for col in [self.user_col, self.item_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing in ratings dataset.")

        # Filter by rating threshold if explicit ratings
        if self.rating_col in df.columns:
            df = df[df[self.rating_col] >= self.min_rating_threshold].copy()
            print(f"Rows after filtering rating >= {self.min_rating_threshold}: {len(df):,}")

        # Drop duplicates
        df = df.drop_duplicates(subset=[self.user_col, self.item_col]).reset_index(drop=True)
        return df

    def filter_k_core(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterative K-core filtering: keeps only users with >= min_user_ratings
        and items with >= min_item_ratings until stable.
        """
        print(f"Starting K-core filtering (min_user={self.min_user_ratings}, min_item={self.min_item_ratings})...")
        while True:
            initial_count = len(df)
            
            user_counts = df[self.user_col].value_counts()
            df = df[df[self.user_col].isin(user_counts[user_counts >= self.min_user_ratings].index)]
            
            item_counts = df[self.item_col].value_counts()
            df = df[df[self.item_col].isin(item_counts[item_counts >= self.min_item_ratings].index)]
            
            if len(df) == initial_count:
                break

        print(f"After K-core filtering: {len(df):,} interactions, "
              f"{df[self.user_col].nunique():,} users, {df[self.item_col].nunique():,} items.")
        return df.reset_index(drop=True)

    def create_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Maps raw user & item IDs to continuous zero-based indices."""
        unique_users = df[self.user_col].unique()
        unique_items = df[self.item_col].unique()

        user2idx = {raw_id: idx for idx, raw_id in enumerate(unique_users)}
        idx2user = {idx: raw_id for raw_id, idx in user2idx.items()}

        item2idx = {raw_id: idx for idx, raw_id in enumerate(unique_items)}
        idx2item = {idx: raw_id for raw_id, idx in item2idx.items()}

        df["user_idx"] = df[self.user_col].map(user2idx)
        df["item_idx"] = df[self.item_col].map(item2idx)

        mappings = {
            "user2idx": user2idx,
            "idx2user": idx2user,
            "item2idx": item2idx,
            "idx2item": idx2item,
            "num_users": len(user2idx),
            "num_items": len(item2idx)
        }
        return df, mappings

    def leave_one_out_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data using Leave-One-Out protocol:
        - The latest interaction per user (or last row) is held out as test item.
        - All prior interactions form the training set.
        """
        # Sort by timestamp if column exists
        if self.timestamp_col in df.columns:
            df = df.sort_values(by=[self.user_col, self.timestamp_col])
        else:
            # Deterministic shuffle per user to pick a test item
            df = df.sample(frac=1.0, random_state=self.seed)

        # For each user, last record is test, rest are train
        test_df = df.groupby("user_idx").tail(1).copy()
        train_df = df.drop(test_df.index).copy()

        print(f"Leave-One-Out split: {len(train_df):,} train samples, {len(test_df):,} test users.")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def generate_test_negatives(
        self,
        full_df: pd.DataFrame,
        test_df: pd.DataFrame,
        num_items: int
    ) -> pd.DataFrame:
        """
        For each test user, sample `num_negatives_test` unobserved items to form the 100-item candidate list.
        Returns a DataFrame with columns: ['user_idx', 'positive_item_idx', 'neg_0', 'neg_1', ..., 'neg_N-1']
        """
        print(f"Sampling {self.num_negatives_test} test negative items per user...")
        # Map user to all interacted items
        user_interacted = full_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

        records = []
        random.seed(self.seed)

        # Vectorized tuple iteration with progress bar
        u_indices = test_df["user_idx"].values
        pos_indices = test_df["item_idx"].values

        for idx in tqdm(range(len(test_df)), desc="Generating test negatives"):
            u = int(u_indices[idx])
            pos_item = int(pos_indices[idx])
            interacted = user_interacted.get(u, set())

            # Fast rejection sampling (num_items is typically much larger than num_negatives)
            neg_set = set()
            while len(neg_set) < self.num_negatives_test:
                rand_item = random.randint(0, num_items - 1)
                if rand_item not in interacted:
                    neg_set.add(rand_item)

            records.append([u, pos_item] + list(neg_set))

        col_names = ["user_idx", "positive_item_idx"] + [f"neg_{i}" for i in range(self.num_negatives_test)]
        return pd.DataFrame(records, columns=col_names)

    def run(self) -> None:
        """Executes full preprocessing pipeline and saves artifacts."""
        os.makedirs(self.processed_dir, exist_ok=True)
        
        df = self.load_raw_data()
        df = self.filter_k_core(df)
        df, mappings = self.create_mappings(df)

        train_df, test_df = self.leave_one_out_split(df)
        test_neg_df = self.generate_test_negatives(df, test_df, mappings["num_items"])

        # Save artifacts
        train_export = train_df[["user_idx", "item_idx"]].copy()
        train_export["interaction"] = 1
        train_export.to_csv(self.train_path, index=False)
        print(f"Saved training data to: {self.train_path}")

        test_neg_df.to_csv(self.test_negatives_path, index=False)
        print(f"Saved test negative evaluation dataset to: {self.test_negatives_path}")

        with open(self.mappings_path, "wb") as f:
            pickle.dump(mappings, f)
        print(f"Saved mappings to: {self.mappings_path}")
        print("Preprocessing successfully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Book-NCF raw ratings.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    preprocessor = DataPreprocessor(cfg)
    preprocessor.run()
