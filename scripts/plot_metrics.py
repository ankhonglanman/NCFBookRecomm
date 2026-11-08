"""
Project: Book-NCF
Module: scripts.plot_metrics
Description: Plots training loss and HR@K / NDCG@K curves from training_history.json.
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(history_path: str = "logs/training_history.json", save_path: str = "logs/training_curves.png"):
    if not os.path.exists(history_path):
        print(f"Error: {history_path} does not exist. Train a model first!")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print("History file is empty.")
        return

    df = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", color="#1f77b4", linewidth=2, label="Train Loss")
    axes[0].set_title("Training Loss across Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Cross-Entropy Loss", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Ranking metrics
    metric_cols = [c for c in df.columns if c.startswith("hr@") or c.startswith("ndcg@")]
    colors = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]

    for idx, col in enumerate(metric_cols):
        c = colors[idx % len(colors)]
        linestyle = "--" if "ndcg" in col else "-"
        axes[1].plot(df["epoch"], df[col], marker="s", linestyle=linestyle, color=c, linewidth=2, label=col.upper())

    axes[1].set_title("Leave-One-Out Test Metrics", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Metric Value", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved metric curves plot to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Book-NCF training curves")
    parser.add_argument("--history", type=str, default="logs/training_history.json")
    parser.add_argument("--output", type=str, default="logs/training_curves.png")
    args = parser.parse_args()

    plot_metrics(args.history, args.output)
