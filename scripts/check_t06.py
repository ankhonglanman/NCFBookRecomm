
import pandas as pd
import numpy as np

train = pd.read_csv("data/processed/train.csv")
val_all = pd.read_csv("data/processed/val_negatives.csv")
test = pd.read_csv(
    "data/processed/test_negatives.csv",
    usecols=["user_idx", "positive_item_idx"]
)

val = val_all.sample(
    n=min(5000, len(val_all)),
    random_state=42
).reset_index(drop=True)

neg_cols = [c for c in val.columns if c.startswith("neg_")]
negative_items = val[neg_cols].to_numpy(dtype=np.int64)

# Tập sách người dùng đã tương tác trong train
train_history = (
    train.groupby("user_idx")["item_idx"]
    .agg(set).to_dict()
)

# Positive được giữ lại cho test
test_positive = dict(zip(
    test["user_idx"],
    test["positive_item_idx"]
))

train_val_overlap = 0
negative_overlap = 0
duplicate_candidates = 0

for idx, row in val.iterrows():
    user = int(row["user_idx"])
    positive = int(row["positive_item_idx"])
    negatives = negative_items[idx]

    history = train_history.get(user, set())

    if positive in history:
        train_val_overlap += 1

    # Kiểm tra negative có chứa sách đã tương tác
    held_out_test = test_positive.get(user, -1)

    for item in negatives:
        if (
            int(item) in history
            or int(item) == positive
            or int(item) == held_out_test
        ):
            negative_overlap += 1

    candidates = np.concatenate([[positive], negatives])
    if len(np.unique(candidates)) != len(candidates):
        duplicate_candidates += 1

print("=== T06 DATA VALIDATION ===")
print("Users:", len(val))
print("Train/Val positive overlap:", train_val_overlap)
print("Negative/known positive overlap:", negative_overlap)
print("Users with duplicate candidates:", duplicate_candidates)
print("=== FINISHED ===")
