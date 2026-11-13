"""T07: User-based KNN Collaborative Filtering for Book-NCF.

Ranking baseline using binary interactions and cosine similarity.
The script trains only on data/processed/train.csv, evaluates on the
same reproducible validation sample as T04 and T06, and does not build
an all-user x all-user dense similarity matrix.

Usage (from repository root):
    python -m scripts.t07_knn --users 200 --k 10
    python -m scripts.t07_knn --users 5000 --k 10
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from src.engine.baseline_evaluator import evaluate_ranking


def build_user_item_matrix(train, num_users, num_items):
    """Construct a CSR matrix containing only observed training interactions."""
    matrix = csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (
                train["user_idx"].to_numpy(dtype=np.int64),
                train["item_idx"].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(num_users, num_items),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def predict_user_knn(R, eval_users, candidates, k=10):
    """Score candidate books from the k most similar OTHER users.

    Similarity is cosine between binary training histories. Neighbor
    ratings are aggregated as a similarity-weighted mean of their
    implicit interactions. Zero-overlap users are not used as neighbors.

    Returns:
        scores: (number of users, number of candidates) float array.
        diagnostics: useful information about the prediction run.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(eval_users) != len(candidates):
        raise ValueError("users and candidates have different lengths")

    R = R.tocsr()
    X = normalize(R, norm="l2", axis=1, copy=True).tocsr()
    XT = X.T  # sparse transpose; do NOT form a dense user x user matrix

    scores = np.zeros(candidates.shape, dtype=np.float32)
    neighbor_counts = []
    empty_user_count = 0

    for row, raw_user in enumerate(eval_users):
        user = int(raw_user)
        # Only users with overlapping observed training books receive
        # a positive cosine similarity. The result is a sparse row.
        sim_row = (X.getrow(user) @ XT).tocsr()
        neighbor_ids = sim_row.indices
        similarities = sim_row.data

        # Exclude target user itself: otherwise the model can copy the
        # query user's history and cease to be a proper KNN baseline.
        mask = (neighbor_ids != user) & (similarities > 0)
        neighbor_ids = neighbor_ids[mask]
        similarities = similarities[mask]

        if len(neighbor_ids) == 0:
            neighbor_counts.append(0)
            empty_user_count += 1
            continue

        # Deterministic tie breaking by user ID after descending similarity.
        order = np.lexsort((neighbor_ids, -similarities))[:k]
        neighbor_ids = neighbor_ids[order]
        weights = similarities[order].astype(np.float32, copy=False)
        neighbor_counts.append(len(neighbor_ids))

        # Per-item weighted mean of neighbor interactions (0 or 1).
        # Materialize only a 10,000-book score vector for one user;
        # never an all-users / all-items dense matrix.
        weighted_items = np.asarray(R[neighbor_ids].T @ weights).ravel()
        scores[row] = weighted_items[candidates[row]] / weights.sum()

        if (row + 1) % 500 == 0:
            print(f"Scored {row + 1}/{len(eval_users)} users")

    diagnostics = {
        "users_without_neighbors": empty_user_count,
        "avg_neighbors": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        "zero_score_ratio": float(np.mean(scores == 0)) if scores.size else 0.0,
    }
    return scores, diagnostics


def main():
    parser = argparse.ArgumentParser(description="T07 User-based KNN ranking baseline")
    parser.add_argument("--users", type=int, default=200, help="number of sampled evaluation users")
    parser.add_argument("--k", type=int, default=10, help="number of nearest neighbors")
    parser.add_argument("--seed", type=int, default=42, help="seed used for deterministic evaluation sample")
    parser.add_argument("--eval-file", default="data/processed/val_negatives.csv")
    parser.add_argument("--output", default="reports/t07_knn_results.csv")
    args = parser.parse_args()
    if args.users <= 0 or args.k <= 0:
        parser.error("--users and --k must both be positive")

    start = time.perf_counter()
    print("=== T07: USER-BASED KNN COSINE ===")
    train = pd.read_csv("data/processed/train.csv")
    eval_all = pd.read_csv(args.eval_file)
    val = eval_all.sample(
        n=min(args.users, len(eval_all)), random_state=args.seed
    ).reset_index(drop=True)

    neg_cols = [c for c in val.columns if c.startswith("neg_")]
    if not neg_cols:
        raise ValueError("No neg_* columns in evaluation CSV")
    eval_users = val["user_idx"].to_numpy(dtype=np.int64)
    positives = val["positive_item_idx"].to_numpy(dtype=np.int64)
    candidates = np.column_stack((
        positives, val[neg_cols].to_numpy(dtype=np.int64)
    ))

    num_users = max(int(train["user_idx"].max()), int(eval_all["user_idx"].max())) + 1
    num_items = max(int(train["item_idx"].max()), int(candidates.max())) + 1
    R = build_user_item_matrix(train, num_users, num_items)
    print(f"Train matrix: {R.shape}, interactions: {R.nnz:,}")
    print(f"Eval users: {len(eval_users)}, candidates/user: {candidates.shape[1]}, k={args.k}")

    print("=== FINDING NEIGHBORS AND SCORING ===")
    scores, diagnostics = predict_user_knn(R, eval_users, candidates, k=args.k)
        
    # ==========================================
    # T07 SCORE DIAGNOSTICS
    # ==========================================

    print("\n=== T07 SCORE DIAGNOSTICS ===")

    positive_scores = scores[:, 0]
    negative_scores = scores[:, 1:]

    print("Positive score > 0:",
        np.mean(positive_scores > 0))

    print("Negative score > 0:",
        np.mean(negative_scores > 0))

    print("Users with all-zero scores:",
        np.sum(np.all(scores == 0, axis=1)))

    print("Average positive score:",
        np.mean(positive_scores))

    print("Average negative score:",
        np.mean(negative_scores))

    ranks = []

    for i in range(len(positives)):
        order = np.lexsort((
            candidates[i],
            -scores[i]
        ))

        rank = np.flatnonzero(order == 0)[0] + 1
        ranks.append(rank)

    ranks = np.array(ranks)

    print("Positive ranked #1:",
        np.sum(ranks == 1))

    print("Positive in Top-10:",
        np.sum(ranks <= 10))

    print("Top-10 hits with positive score = 0:",
        np.sum(
            (ranks <= 10) &
            (positive_scores == 0)
        ))

    print("Average positive rank:",
        np.mean(ranks))

    print("=== END DIAGNOSTICS ===")

    print("=== RANKING EVALUATION ===")
    metrics = evaluate_ranking(
        scores=scores,
        candidate_items=candidates,
        positive_items=positives,
        top_ks=(5, 10, 20),
    )
    for name, value in metrics.items():
        print(f"{name}: {int(value) if name == 'num_users' else f'{value:.4f}'}")
    print(f"Zero score ratio: {diagnostics['zero_score_ratio']:.4f}")
    print(f"Users with no neighbors: {diagnostics['users_without_neighbors']}")
    print(f"Average neighbor count: {diagnostics['avg_neighbors']:.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "model": "UserKNN_Cosine",
        "dataset": "Validation" if "val" in Path(args.eval_file).stem else "Test",
        **metrics,
        "k": args.k,
        "seed": args.seed,
        "candidates_per_user": candidates.shape[1],
        **diagnostics,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    pd.DataFrame([output]).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("=== T07 COMPLETED ===")


if __name__ == "__main__":
    main()
