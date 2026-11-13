
import pandas as pd
from pathlib import Path

# Đường dẫn dữ liệu
DATA_DIR = Path("data/processed")

train_path = DATA_DIR / "train.csv"
val_path = DATA_DIR / "val_negatives.csv"
test_path = DATA_DIR / "test_negatives.csv"

# Kiểm tra file có tồn tại
for path in [train_path, val_path, test_path]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

# Đọc dữ liệu
train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)

print("DATASET SUMMARY")
print("Train shape:", train.shape)
print("Validation shape:", val.shape)
print("Test shape:", test.shape)

print("\nTRAIN SAMPLE")
print(train.head())

print("\nVALIDATION SAMPLE")
print(val.iloc[:5, :8])

print("\nSTATISTICS")
print("Users:", train["user_idx"].nunique())
print("Books:", train["item_idx"].nunique())
print("Train interactions:", len(train))

print("\nMISSING VALUES")
print("Train:", train.isnull().sum().sum())
print("Validation:", val.isnull().sum().sum())
print("Test:", test.isnull().sum().sum())

print("\nEVALUATION FORMAT")
neg_cols = [c for c in val.columns if c.startswith("neg_")]

print("Negative candidates per user:", len(neg_cols))
print("Total candidates per user:", len(neg_cols) + 1)

print("\nDATA LEAKAGE CHECK")

train_pairs = train[["user_idx", "item_idx"]].drop_duplicates()

def count_overlap(eval_df):
    positives = eval_df[
        ["user_idx", "positive_item_idx"]
    ].rename(columns={"positive_item_idx": "item_idx"})

    overlap = positives.merge(
        train_pairs,
        on=["user_idx", "item_idx"],
        how="inner"
    )
    return len(overlap)

print("Train/Validation overlap:", count_overlap(val))
print("Train/Test overlap:", count_overlap(test))

print("\nFINISHED")
