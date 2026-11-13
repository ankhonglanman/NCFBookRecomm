
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.engine.baseline_evaluator import evaluate_ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=200)
    args = parser.parse_args()

    # 1. Load train and validation
    print("=== LOADING DATA ===")

    train = pd.read_csv("data/processed/train.csv")
    val_all = pd.read_csv(
        "data/processed/val_negatives.csv"
    )

    val = val_all.sample(
        n=min(args.users, len(val_all)),
        random_state=42
    ).reset_index(drop=True)

    neg_cols = [
        c for c in val.columns if c.startswith("neg_")
    ]

    positives = val["positive_item_idx"].to_numpy(
        dtype=np.int64
    )

    candidates = np.column_stack([
        positives,
        val[neg_cols].to_numpy(dtype=np.int64)
    ])

    # 2. Build sparse user-item matrix
    print("=== BUILDING USER-ITEM MATRIX ===")

    num_users = max(
        int(train["user_idx"].max()),
        int(val["user_idx"].max())
    ) + 1

    num_items = max(
        int(train["item_idx"].max()),
        int(candidates.max())
    ) + 1

    R = csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (
                train["user_idx"].to_numpy(),
                train["item_idx"].to_numpy()
            )
        ),
        shape=(num_users, num_items)
    )

    R.sum_duplicates()
    R.data[:] = 1.0

    print("Matrix shape:", R.shape)
    print("Interactions:", R.nnz)

    # 3. Item-based Cosine Similarity
    print("=== COMPUTING COSINE REPRESENTATIONS ===")

    book_vectors = normalize(
        R.T.tocsr(),
        norm="l2",
        axis=1
    )

    # 4. Score candidates
    print("=== PREDICTING SCORES ===")

    scores = np.zeros(
        candidates.shape,
        dtype=np.float32
    )

    for row, user in enumerate(val["user_idx"]):
        user = int(user)
        history = R.getrow(user).indices

        if len(history) == 0:
            continue

        profile = np.asarray(
            book_vectors[history].sum(axis=0)
        ).ravel()

        scores[row] = np.asarray(
            book_vectors[candidates[row]] @ profile
        ).ravel()

    # 5. Evaluate
    print("=== EVALUATING COSINE CF ===")

    results = evaluate_ranking(
        scores=scores,
        candidate_items=candidates,
        positive_items=positives,
        top_ks=(5, 10, 20)
    )

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("Zero score ratio:", np.mean(scores == 0))

    # 6. Save results
    Path("reports").mkdir(exist_ok=True)

    pd.DataFrame([{
        "model": "ItemCosine",
        "dataset": "Validation",
        **results
    }]).to_csv(
        "reports/t06_cosine_results.csv",
        index=False
    )

    print("Saved: reports/t06_cosine_results.csv")
    print("=== T06 COMPLETED ===")


if __name__ == "__main__":
    main()
