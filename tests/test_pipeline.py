"""
Project: Book-NCF
Module: tests.test_pipeline
Description: Unit tests verifying models, dataset dynamic negative sampling,
             and evaluation logic.
"""

import unittest
import torch
import pandas as pd
import numpy as np

from src.models.gmf import GMF
from src.models.mlp import MLP
from src.models.neumf import NeuMF
from src.data.dataset import NCFDataset


class TestBookNCFPipeline(unittest.TestCase):

    def setUp(self):
        self.num_users = 50
        self.num_items = 100
        self.batch_size = 16

        # Dummy train interactions
        self.train_data = []
        for u in range(self.num_users):
            for i in range(5):
                self.train_data.append({"user_idx": u, "item_idx": (u + i) % self.num_items})
        self.train_df = pd.DataFrame(self.train_data)

    def test_gmf_forward(self):
        model = GMF(num_users=self.num_users, num_items=self.num_items, latent_dim=16)
        users = torch.randint(0, self.num_users, (self.batch_size,))
        items = torch.randint(0, self.num_items, (self.batch_size,))

        preds = model(users, items)
        self.assertEqual(preds.shape, (self.batch_size,))
        self.assertTrue((preds >= 0.0).all() and (preds <= 1.0).all())

    def test_mlp_forward(self):
        model = MLP(num_users=self.num_users, num_items=self.num_items, latent_dim=16, hidden_dims=[32, 16])
        users = torch.randint(0, self.num_users, (self.batch_size,))
        items = torch.randint(0, self.num_items, (self.batch_size,))

        preds = model(users, items)
        self.assertEqual(preds.shape, (self.batch_size,))
        self.assertTrue((preds >= 0.0).all() and (preds <= 1.0).all())

    def test_neumf_forward(self):
        model = NeuMF(num_users=self.num_users, num_items=self.num_items, latent_dim_gmf=16, latent_dim_mlp=16)
        users = torch.randint(0, self.num_users, (self.batch_size,))
        items = torch.randint(0, self.num_items, (self.batch_size,))

        preds = model(users, items)
        self.assertEqual(preds.shape, (self.batch_size,))
        self.assertTrue((preds >= 0.0).all() and (preds <= 1.0).all())

    def test_dynamic_negative_sampling(self):
        num_neg = 4
        dataset = NCFDataset(
            train_df=self.train_df,
            num_users=self.num_users,
            num_items=self.num_items,
            num_negatives=num_neg,
            seed=42
        )
        # Expected size: num_pos * (1 + num_neg)
        expected_len = len(self.train_df) * (1 + num_neg)
        self.assertEqual(len(dataset), expected_len)

        # Resample negatives
        dataset.sample_negatives()
        self.assertEqual(len(dataset), expected_len)

        # Check item get
        u, i, label = dataset[0]
        self.assertIsInstance(u, torch.Tensor)
        self.assertIsInstance(i, torch.Tensor)
        self.assertIn(label.item(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
