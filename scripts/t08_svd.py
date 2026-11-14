
"""
T08 - TruncatedSVD Book Recommendation

Run:
python -m scripts.t08_svd --users 200 --factors 32
python -m scripts.t08_svd --users 5000 --factors 32
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.engine.baseline_evaluator import evaluate_ranking


# ==========================================
# 1. BUILD USER-ITEM MATRIX
# ==========================================

def build_user_item_matrix(train, num_users, num_items):

    R = csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (
                train["user_idx"].to_numpy(dtype=np.int64),
                train["item_idx"].to_numpy(dtype=np.int64)
            )
        ),
        shape=(num_users, num_items),
        dtype=np.float32
    )

    # Loai bo duplicate va chuyen ve binary
    R.sum_duplicates()
    R.data[:] = 1.0

    return R


# ==========================================
# 2. SCORE CANDIDATE BOOKS
# ==========================================

def score_candidates(
    user_factors,
    item_factors,
    users,
    candidates,
    batch_size=256
):

    scores = np.empty(
        candidates.shape,
        dtype=np.float32
    )

    for start in range(0, len(users), batch_size):

        end = min(start + batch_size, len(users))

        # (batch, factors)
        U_batch = user_factors[users[start:end]]

        # (batch, candidates, factors)
        V_batch = item_factors[candidates[start:end]]

        # Dot product for each user-candidate pair
        scores[start:end] = np.einsum(
            "bd,bcd->bc",
            U_batch,
            V_batch,
            optimize=True
        )

    return scores


# ==========================================
# 3. MAIN PIPELINE
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    if args.users <= 0 or args.factors <= 0:
        raise ValueError("users va factors phai > 0")

    if args.batch_size <= 0:
        raise ValueError("batch-size phai > 0")

    # ======================================
    # LOAD DATA
    # ======================================

    print("=== T08: TRUNCATED SVD ===")
    print("=== LOADING DATA ===")

    train = pd.read_csv(
        "data/processed/train.csv"
    )

    val_all = pd.read_csv(
        "data/processed/val_negatives.csv"
    )

    # Same users as T04, T06, T07
    val = val_all.sample(
        n=min(args.users, len(val_all)),
        random_state=42
    ).reset_index(drop=True)

    users = val["user_idx"].to_numpy(
        dtype=np.int64
    )

    positives = val["positive_item_idx"].to_numpy(
        dtype=np.int64
    )

    neg_cols = sorted(
        [c for c in val.columns if c.startswith("neg_")],
        key=lambda c: int(c.split("_")[1])
    )

    candidates = np.column_stack([
        positives,
        val[neg_cols].to_numpy(dtype=np.int64)
    ])

    if np.any(
        np.diff(np.sort(candidates, axis=1), axis=1) == 0
    ):
        raise ValueError("Duplicate candidate items")

    print("Train interactions:", len(train))
    print("Evaluation users:", len(val))
    print("Candidates shape:", candidates.shape)

    # ======================================
    # BUILD MATRIX
    # ======================================

    print("\n=== BUILDING USER-ITEM MATRIX ===")

    num_users = max(
        int(train["user_idx"].max()),
        int(users.max())
    ) + 1

    num_items = max(
        int(train["item_idx"].max()),
        int(candidates.max())
    ) + 1

    R = build_user_item_matrix(
        train,
        num_users,
        num_items
    )

    print("Matrix shape:", R.shape)
    print("Nonzero interactions:", R.nnz)

    if args.factors >= min(R.shape):
        raise ValueError("So latent factors qua lon")

    # Check held-out positives
    overlap = sum(
        R[int(u), int(p)] > 0
        for u, p in zip(users, positives)
    )

    print("Train/Validation positive overlap:", overlap)

    if overlap > 0:
        raise ValueError("Data leakage: positive overlap")

    # ======================================
    # TRAIN TRUNCATED SVD
    # ======================================

    print("\n=== TRAINING SVD ===")

    start_time = time.perf_counter()

    svd = TruncatedSVD(
        n_components=args.factors,
        algorithm="randomized",
        n_iter=7,
        random_state=args.seed
    )

    # User representations: (users, factors)
    user_factors = svd.fit_transform(R).astype(
        np.float32
    )

    # Book representations: (books, factors)
    item_factors = svd.components_.T.astype(
        np.float32
    )

    train_time = time.perf_counter() - start_time

    print("User factors shape:", user_factors.shape)
    print("Item factors shape:", item_factors.shape)
    print("Training time:", round(train_time, 2), "seconds")

    print(
        "Explained variance ratio:",
        round(float(svd.explained_variance_ratio_.sum()), 4)
    )

    # ======================================
    # SCORE CANDIDATES
    # ======================================

    print("\n=== PREDICTING SCORES ===")

    score_start = time.perf_counter()

    scores = score_candidates(
        user_factors,
        item_factors,
        users,
        candidates,
        batch_size=args.batch_size
    )

    scoring_time = time.perf_counter() - score_start

    print("Scores shape:", scores.shape)
    print("Scoring time:", round(scoring_time, 2), "seconds")

    print("Min score:", float(scores.min()))
    print("Max score:", float(scores.max()))

    if not np.isfinite(scores).all():
        raise ValueError("Invalid scores: NaN/Inf")

    # ======================================
    # EVALUATE
    # ======================================

    print("\n=== RANKING EVALUATION ===")

    results = evaluate_ranking(
        scores=scores,
        candidate_items=candidates,
        positive_items=positives,
        top_ks=(5, 10, 20)
    )

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    zero_ratio = float(np.mean(scores == 0))

    print("Zero score ratio:", round(zero_ratio, 4))

    # ======================================
    # SAVE RESULTS
    # ======================================

    Path("reports").mkdir(exist_ok=True)

    output = {
        "model": "TruncatedSVD",
        "dataset": "Validation",
        **results,
        "factors": args.factors,
        "seed": args.seed,
        "train_time_sec": round(train_time, 3),
        "scoring_time_sec": round(scoring_time, 3),
        "zero_score_ratio": zero_ratio
    }

    df = pd.DataFrame([output])

    # Ket qua moi nhat
    df.to_csv(
        "reports/t08_svd_results.csv",
        index=False
    )

    # Giu rieng tung lan chay
    run_path = (
        f"reports/t08_svd_k{args.factors}_"
        f"{len(val)}users.csv"
    )

    df.to_csv(run_path, index=False)

    print("\nSaved: reports/t08_svd_results.csv")
    print("Saved:", run_path)

    print("=== T08 COMPLETED ===")


if __name__ == "__main__":
    main()
