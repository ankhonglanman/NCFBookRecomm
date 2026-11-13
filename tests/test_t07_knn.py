"""Unit tests for scripts/t07_knn.py (place under tests/)."""
import unittest

import numpy as np
import pandas as pd

from scripts.t07_knn import build_user_item_matrix, predict_user_knn


class TestUserKNN(unittest.TestCase):
    def test_excludes_self_and_uses_other_users(self):
        # User 0 has book 0. User 1 shares book 0 and also has book 1.
        train = pd.DataFrame({"user_idx": [0, 1, 1, 2], "item_idx": [0, 0, 1, 2]})
        R = build_user_item_matrix(train, num_users=3, num_items=4)
        scores, diagnostics = predict_user_knn(
            R,
            eval_users=np.array([0, 1, 2]),
            candidates=np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            k=10,
        )
        np.testing.assert_allclose(scores[0], [1, 1, 0], atol=1e-6)
        np.testing.assert_allclose(scores[1], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(scores[2], [0, 0, 0], atol=1e-6)
        self.assertEqual(diagnostics["users_without_neighbors"], 1)

    def test_k_one_selects_closest_neighbor(self):
        train = pd.DataFrame({
            "user_idx": [0, 0, 1, 1, 1, 2, 2],
            "item_idx": [0, 1, 0, 1, 2, 0, 3],
        })
        R = build_user_item_matrix(train, num_users=3, num_items=4)
        scores, _ = predict_user_knn(
            R, np.array([0]), np.array([[2, 3]]), k=1
        )
        self.assertAlmostEqual(float(scores[0, 0]), 1.0)
        self.assertAlmostEqual(float(scores[0, 1]), 0.0)

    def test_rejects_invalid_k(self):
        R = build_user_item_matrix(
            pd.DataFrame({"user_idx": [0], "item_idx": [0]}), 1, 1
        )
        with self.assertRaises(ValueError):
            predict_user_knn(R, np.array([0]), np.array([[0]]), k=0)


if __name__ == "__main__":
    unittest.main()
