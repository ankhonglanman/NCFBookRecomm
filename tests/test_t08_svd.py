
import unittest
import numpy as np
import pandas as pd

from scripts.t08_svd import (
    build_user_item_matrix,
    score_candidates
)


class TestT08SVD(unittest.TestCase):

    def test_binary_matrix(self):

        train = pd.DataFrame({
            "user_idx": [0, 0, 1],
            "item_idx": [1, 1, 2]
        })

        R = build_user_item_matrix(
            train,
            num_users=3,
            num_items=4
        )

        self.assertEqual(R.shape, (3, 4))
        self.assertEqual(R.nnz, 2)
        self.assertEqual(R[0, 1], 1.0)

    def test_candidate_scores(self):

        user_factors = np.array([
            [1, 2],
            [2, 1]
        ], dtype=np.float32)

        item_factors = np.array([
            [1, 0],
            [0, 1],
            [1, 1]
        ], dtype=np.float32)

        users = np.array([0, 1])

        candidates = np.array([
            [0, 2],
            [1, 2]
        ])

        scores = score_candidates(
            user_factors,
            item_factors,
            users,
            candidates,
            batch_size=1
        )

        expected = np.array([
            [1, 3],
            [1, 3]
        ])

        np.testing.assert_allclose(scores, expected)


if __name__ == "__main__":
    unittest.main()
