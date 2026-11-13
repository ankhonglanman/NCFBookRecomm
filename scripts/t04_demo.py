
import numpy as np
import pandas as pd
from pathlib import Path

from src.engine.baseline_evaluator import evaluate_ranking


# ==========================================
# 1. LOAD DATA
# ==========================================

print("=== LOADING DATA ===")

train = pd.read_csv(
    "data/processed/train.csv"
)


val_all = pd.read_csv(
    "data/processed/val_negatives.csv"
)

val = val_all.sample(
    n=min(5000, len(val_all)),
    random_state=42
).reset_index(drop=True)


print("Train interactions:", len(train))
print("Validation users:", len(val))


# ==========================================
# 2. PREPARE CANDIDATES
# ==========================================

neg_cols = [
    col for col in val.columns
    if col.startswith("neg_")
]

positive_items = val[
    "positive_item_idx"
].to_numpy(dtype=np.int64)

negative_items = val[
    neg_cols
].to_numpy(dtype=np.int64)

candidates = np.column_stack([
    positive_items,
    negative_items
])

print("Candidates shape:", candidates.shape)


# ==========================================
# 3. POPULARITY BASELINE
# ==========================================

print("\n=== POPULARITY MODEL ===")

# Đếm số lượt tương tác của từng sách
popularity = train[
    "item_idx"
].value_counts()

# Sách xuất hiện nhiều hơn có điểm cao hơn
scores = popularity.reindex(
    candidates.ravel(),
    fill_value=0
).to_numpy().reshape(candidates.shape)

print("Scores shape:", scores.shape)

print("\n=== POPULARITY DIAGNOSTICS ===")

print("Positive popularity mean:", scores[:, 0].mean())
print("Negative popularity mean:", scores[:, 1:].mean())
print("Maximum popularity:", popularity.max())

print("\n=== RANDOM BASELINE ===")

rng = np.random.default_rng(42)

random_scores = rng.random(candidates.shape)

random_results = evaluate_ranking(
    scores=random_scores,
    candidate_items=candidates,
    positive_items=positive_items,
    top_ks=(5, 10, 20)
)

for metric, value in random_results.items():
    print(f"{metric}: {value}")

# ==========================================
# DEBUG: KIEM TRA DIEM DU DOAN
# ==========================================

print("\n========== DEBUG T04 ==========")

print("So users:", len(val))
print("So books trong train:", train["item_idx"].nunique())

print("\n5 positive book IDs:")
print(positive_items[:5])

print("\nDiem positive cua 5 users dau:")
print(scores[:5, 0])

print("\nDiem 10 candidates cua user dau:")
print(scores[0, :10])

print("\nSo diem khac 0:",
      np.count_nonzero(scores))

print("Tong so diem:",
      scores.size)

print("\nSo positive co diem > 0:",
      np.sum(scores[:, 0] > 0))

print("Diem positive trung binh:",
      np.mean(scores[:, 0]))

print("Diem negative lon nhat (trung binh):",
      np.mean(np.max(scores[:, 1:], axis=1)))

# Tinh thu hang positive doc lap voi evaluator
ranks = []

for user_scores, user_items, pos in zip(
    scores, candidates, positive_items
):
    order = np.lexsort((user_items, -user_scores))
    ranked_items = user_items[order]

    rank = np.flatnonzero(ranked_items == pos)[0] + 1
    ranks.append(rank)

ranks = np.array(ranks)

print("\n10 thu hang positive dau tien:")
print(ranks[:10])

print("So positive trong Top-5:", np.sum(ranks <= 5))
print("So positive trong Top-10:", np.sum(ranks <= 10))
print("So positive trong Top-20:", np.sum(ranks <= 20))

print("Thu hang trung binh:", np.mean(ranks))

print("\n========== END DEBUG ==========")


# ==========================================
# 4. EVALUATE
# ==========================================

print("\n=== EVALUATION RESULTS ===")

results = evaluate_ranking(
    scores=scores,
    candidate_items=candidates,
    positive_items=positive_items,
    top_ks=(5, 10, 20)
)

for metric, value in results.items():
    if metric == "num_users":
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.4f}")


# ==========================================
# 5. SAVE RESULTS
# ==========================================

Path("reports").mkdir(exist_ok=True)

output = {
    "model": "Popularity",
    "dataset": "Validation",
    **results
}

pd.DataFrame([output]).to_csv(
    "reports/t04_popularity_results.csv",
    index=False
)

print("\nSaved: reports/t04_popularity_results.csv")

# ==========================================
# RANDOM BASELINE - SANITY CHECK
# ==========================================

print("\n=== RANDOM BASELINE ===")

rng = np.random.default_rng(42)

random_scores = rng.random(candidates.shape)

random_results = evaluate_ranking(
    scores=random_scores,
    candidate_items=candidates,
    positive_items=positive_items,
    top_ks=(5, 10, 20)
)

print("\n=== COMPARISON ===")
print(f"{'Metric':<12} {'Popularity':>12} {'Random':>12}")

for k in [5, 10, 20]:
    for metric in [f"hr@{k}", f"ndcg@{k}"]:
        print(
            f"{metric:<12} "
            f"{results[metric]:>12.4f} "
            f"{random_results[metric]:>12.4f}"
        )

# Lưu cả hai mô hình để đối chiếu
comparison = pd.DataFrame([
    {
        "model": "Popularity",
        "dataset": "Validation",
        **results
    },
    {
        "model": "Random",
        "dataset": "Validation",
        **random_results
    }
])

comparison.to_csv(
    "reports/t04_baseline_comparison.csv",
    index=False
)

print("\nSaved: reports/t04_baseline_comparison.csv")

print("=== T04.4 COMPLETED ===")
