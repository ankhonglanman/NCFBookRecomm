
import unittest
import numpy as np

from src.engine.baseline_evaluator import evaluate_ranking


class TestBaselineEvaluator(unittest.TestCase):

    def test_known_ranks(self):
        candidates = np.tile(np.arange(30), (3, 1))
        scores = np.tile(
            np.arange(30, 0, -1, dtype=float),
            (3, 1)
        )
        positives = np.array([0, 7, 24])

        result = evaluate_ranking(
            scores, candidates, positives,
            top_ks=(5, 10, 20)
        )

        self.assertAlmostEqual(result["hr@5"], 1/3)
        self.assertAlmostEqual(result["hr@10"], 2/3)
        self.assertAlmostEqual(result["hr@20"], 2/3)

        expected = (1 + 1/np.log2(9)) / 3
        self.assertAlmostEqual(result["ndcg@10"], expected)

    def test_tie_breaking(self):
        candidates = np.arange(29, -1, -1).reshape(1, 30)
        scores = np.full((1, 30), 0.5)

        result = evaluate_ranking(
            scores, candidates, np.array([0])
        )
        self.assertEqual(result["hr@5"], 1.0)

    def test_invalid_positive(self):
        candidates = np.arange(30).reshape(1, 30)
        scores = np.zeros((1, 30))

        with self.assertRaises(ValueError):
            evaluate_ranking(
                scores, candidates, np.array([99])
            )


if __name__ == "__main__":
    unittest.main()
