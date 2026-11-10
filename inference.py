"""
Project: Book-NCF (Neural Collaborative Filtering for Book Recommendation)
File: inference.py
Description: Generates personalized Top-K book recommendations for a specified User ID.
             Filters out previously interacted books and decorates results
             with metadata (title, author) from books.csv if available.
"""

import os
import pickle
import argparse
import yaml
import torch
import numpy as np
import pandas as pd

from train import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Book-NCF Top-K Recommendation Inference")
    parser.add_argument("--user_id", type=str, required=True, help="Target user ID (as in raw dataset or mapped index)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations to return")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to configuration YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to trained checkpoint")
    parser.add_argument("--books_path", type=str, default=None, help="Path to books.csv metadata file")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, config: dict, num_users: int, num_items: int, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_cfg = checkpoint.get("config", config)
    model_type = saved_cfg.get("model", {}).get("model_type", "neumf")

    model = build_model(model_type, num_users, num_items, saved_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_type


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    mappings_path = data_cfg.get("mappings_path", "data/processed/id_mappings.pkl")
    train_path = data_cfg.get("train_path", "data/processed/train.csv")
    books_path = args.books_path or data_cfg.get("books_path", "data/raw/books.csv")

    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"ID mappings not found at {mappings_path}. Please run preprocessing first.")

    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    user2idx = mappings["user2idx"]
    idx2item = mappings["idx2item"]
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]

    # Match User ID: check as raw string, int, or direct index
    raw_uid = args.user_id
    user_idx = None

    if raw_uid in user2idx:
        user_idx = user2idx[raw_uid]
    else:
        try:
            int_uid = int(raw_uid)
            if int_uid in user2idx:
                user_idx = user2idx[int_uid]
            elif 0 <= int_uid < num_users:
                user_idx = int_uid
        except ValueError:
            pass

    if user_idx is None:
        raise ValueError(f"User '{raw_uid}' not found in trained dataset mappings. Total users: {num_users}")

    print(f"\n=======================================================")
    print(f" Generating Top-{args.top_k} Book Recommendations")
    print(f" Target User ID: {raw_uid} (Internal Index: {user_idx})")
    print(f"=======================================================")

    # Find items already interacted by user to exclude them
    interacted_items = set()
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        user_history = train_df[train_df["user_idx"] == user_idx]["item_idx"].values
        interacted_items = set(user_history)
        print(f"User has interacted with {len(interacted_items)} items in training set.")

    # Candidate pool: all unobserved items
    all_items = np.arange(num_items)
    unseen_items = np.array([i for i in all_items if i not in interacted_items], dtype=np.int64)

    if len(unseen_items) == 0:
        print("User has already interacted with all available books!")
        return

    # Load model
    device = "cuda" if torch.cuda.is_available() and config["train"].get("device") == "cuda" else "cpu"
    model, model_type = load_model_from_checkpoint(args.checkpoint, config, num_users, num_items, device)
    print(f"Using trained model: {model_type.upper()} on {device}")

    # Predict scores in batches
    user_tensor = torch.full((len(unseen_items),), fill_value=user_idx, dtype=torch.long, device=device)
    item_tensor = torch.tensor(unseen_items, dtype=torch.long, device=device)

    batch_size = 1024
    scores_list = []
    with torch.no_grad():
        for i in range(0, len(unseen_items), batch_size):
            u_batch = user_tensor[i:i + batch_size]
            it_batch = item_tensor[i:i + batch_size]
            batch_scores = model(u_batch, it_batch)
            scores_list.append(batch_scores.cpu().numpy())

    all_scores = np.concatenate(scores_list)

    # Rank top-K
    top_k = min(args.top_k, len(unseen_items))
    top_indices = np.argpartition(all_scores, -top_k)[-top_k:]
    # Sort partitioned top indices by score descending
    sorted_top_indices = top_indices[np.argsort(-all_scores[top_indices])]

    recommended_item_indices = unseen_items[sorted_top_indices]
    recommended_scores = all_scores[sorted_top_indices]
    recommended_raw_book_ids = [idx2item[idx] for idx in recommended_item_indices]

    # Load book metadata if available
    books_meta = None
    if os.path.exists(books_path):
        try:
            books_meta = pd.read_csv(books_path)
            book_col = data_cfg.get("item_col", "book_id")
            meta_id_col = data_cfg.get("book_meta_id_col", "id" if "id" in books_meta.columns else book_col)
            if meta_id_col in books_meta.columns:
                books_meta[meta_id_col] = books_meta[meta_id_col].astype(str)
                books_meta = books_meta.set_index(meta_id_col)
            elif book_col in books_meta.columns:
                books_meta[book_col] = books_meta[book_col].astype(str)
                books_meta = books_meta.set_index(book_col)
        except Exception as e:
            print(f"Warning: Could not parse books metadata ({e})")

    # Display recommendations
    print(f"\nTop-{top_k} Recommended Books:")
    print("-" * 75)
    print(f"{'Rank':<5} | {'Score':<8} | {'Book ID':<12} | {'Title / Details'}")
    print("-" * 75)

    for rank, (b_id, score) in enumerate(zip(recommended_raw_book_ids, recommended_scores), start=1):
        str_bid = str(b_id)
        title_str = "N/A"
        if books_meta is not None and str_bid in books_meta.index:
            row = books_meta.loc[str_bid]
            title = row.get("title", row.get("original_title", "Unknown Title"))
            author = row.get("authors", row.get("author", ""))
            title_str = f"{title}" + (f" (by {author})" if author else "")

        print(f"{rank:<5} | {score:<8.4f} | {str_bid:<12} | {title_str}")
    print("-" * 75)


if __name__ == "__main__":
    main()
