# Book-NCF: Neural Collaborative Filtering for Book Recommendation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research and production-grade implementation of personalized book recommendation using **Neural Collaborative Filtering (NCF)**, following the seminal paper by Xiangnan He et al. (WWW 2017). Designed for academic and production workflows, this repository provides a complete pipeline from raw data processing, feature transformation, deep neural model training (GMF, MLP, NeuMF), through to ranking evaluation and real-time inference.

---

## 🌟 Key Features

- **Modular & Decoupled Architecture**:
  - `GMF` (Generalized Matrix Factorization): Models linear user-item interactions via element-wise (Hadamard) product.
  - `MLP` (Multi-Layer Perceptron): Models high-order non-linear latent feature interactions.
  - `NeuMF` (Neural Matrix Factorization): Fuses GMF and MLP branches, with full support for loading pretrained weights from both standalone branches.
- **Robust Recommendation Preprocessing**:
  - Iterative K-core filtering for users and items to eliminate extreme sparsity.
  - Implicit feedback binarization for ranking tasks.
  - Rigorous **Leave-One-Out (LOO)** protocol: the latest interaction per user is held out for testing.
  - Unbiased ranking evaluation candidate set: **1 Positive Item + 99 Negative Items** per test user.
- **Dynamic Negative Sampling**: Resamples unobserved items on every training epoch to avoid static sample bias and enhance generalization.
- **Standard Ranking Metrics**:
  - **HR@K** (Hit Ratio at Rank K).
  - **NDCG@K** (Normalized Discounted Cumulative Gain at Rank K).
- **Practical Top-K Inference**: Automatically filters out items the user has already read, scores all unseen candidate books, and annotates recommendations with real metadata from `books.csv`.

---

## 📂 Project Structure

```text
book-ncf/
├── configs/
│   └── config.yaml          # Centralized hyperparameters & paths configuration
├── data/
│   ├── raw/                 # Raw dataset files (ratings.csv, books.csv)
│   └── processed/           # Processed files (train.csv, test_negatives.csv, id_mappings.pkl)
├── notebooks/
│   ├── eda.ipynb            # Exploratory Data Analysis (long-tail distribution, sparsity)
│   └── colab_runner.ipynb   # Execution notebook for Google Colab (Drive mount, CLI commands)
├── scripts/
│   ├── generate_sample_data.py # Synthetic sample data generator for immediate testing
│   └── plot_metrics.py      # Script to plot loss and HR/NDCG evaluation curves
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── preprocess.py    # K-core filtering, continuous ID mapping, Leave-One-Out split
│   │   └── dataset.py       # PyTorch Dataset supporting dynamic negative sampling per epoch
│   ├── models/
│   │   ├── gmf.py           # Standalone Generalized Matrix Factorization branch
│   │   ├── mlp.py           # Standalone Multi-Layer Perceptron branch
│   │   └── neumf.py         # NeuMF ensemble (combines GMF + MLP with pretrained weight loader)
│   ├── engine/
│   │   ├── trainer.py       # Trainer class handling epochs, early stopping, and checkpoints
│   │   └── evaluator.py     # Batched HR@K and NDCG@K evaluation on test negatives
│   └── utils/
│       ├── seed.py          # Deterministic seed fixing (torch, numpy, random, cudnn)
│       └── logger.py        # Stream/File logging and metric history JSON export
├── tests/
│   └── test_pipeline.py     # Unit tests verifying forward passes and dataset sampling
├── train.py                 # CLI entry-point accepting flags and config overrides
├── inference.py             # CLI script generating Top-K book recommendations for a target user
├── requirements.txt         # Pinned dependency packages
├── .gitignore               # Ignores model checkpoints, caches, logs, and raw CSV files
├── AGENT.md                 # Technical guide and conventions for AI Agents and developers
└── README.md                # Project documentation and quick-start guide
```

---

## 🏗️ Model Architecture (NeuMF Framework)

```text
                                  Prediction: y_ui
                                         ▲
                                         │
                             [ Output Layer (Linear + Sigmoid) ]
                                         ▲
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
        [ GMF Element-wise Product ]                     [ MLP Layer 3 ]
                 ▲                                               ▲
          ┌──────┴──────┐                                        │
          │             │                                 [ MLP Layer 2 ]
          │             │                                        ▲
          │             │                                        │
          │             │                                 [ MLP Layer 1 ]
          │             │                                        ▲
          │             │                                 ┌──────┴──────┐
          │             │                                 │ Concatenate │
          │             │                                 └──────┬──────┘
    GMF User Emb   GMF Item Emb                     MLP User Emb   MLP Item Emb
          ▲             ▲                                 ▲             ▲
          │             │                                 │             │
      User ID        Item ID                           User ID       Item ID
```

---

## 🚀 Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate    # On Linux/macOS
# .venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📖 Quick-Start Guide

### Step 1: Prepare Data
Place your raw CSV datasets into `data/raw/`:
- `data/raw/ratings.csv` (contains at least: `user_id`, `book_id`, `rating`, and optionally `timestamp`)
- `data/raw/books.csv` (contains book metadata: `book_id`, `title`, `authors`, etc.)

*(Optional)* If you don't have a dataset yet, you can instantly generate a synthetic sample dataset:
```bash
python scripts/generate_sample_data.py
```

### Step 2: Run Data Preprocessing
Run K-core filtering, index mapping, and Leave-One-Out split generation:
```bash
python -m src.data.preprocess --config configs/config.yaml
```
Output artifacts are written to `data/processed/`:
- `train.csv`
- `test_negatives.csv`
- `id_mappings.pkl`

### Step 3: Train Models

Train with default configurations from `configs/config.yaml`:
```bash
# Train NeuMF model from scratch
python train.py --config configs/config.yaml --model neumf

# Train standalone GMF branch
python train.py --config configs/config.yaml --model gmf --epochs 15

# Train standalone MLP branch
python train.py --config configs/config.yaml --model mlp --epochs 15
```

To train NeuMF initialized with pretrained weights from standalone GMF and MLP models:
```bash
python train.py --config configs/config.yaml --model neumf \
    --pretrained_gmf checkpoints/gmf_best.pt \
    --pretrained_mlp checkpoints/mlp_best.pt
```

### Step 4: Top-K Recommendation Inference
Generate personalized Top-10 book recommendations for any User ID:
```bash
python inference.py --user_id 42 --top_k 10 --checkpoint checkpoints/best_model.pt
```

### Step 5: Plot Training & Metric Curves
```bash
python scripts/plot_metrics.py --history logs/training_history.json --output logs/training_curves.png
```

---

## 📊 References & Citations

1. **He, Xiangnan, et al.** "Neural collaborative filtering." *Proceedings of the 26th international conference on world wide web*. 2017.
2. Goodbooks-10k / Book-Crossing Benchmark Datasets.
