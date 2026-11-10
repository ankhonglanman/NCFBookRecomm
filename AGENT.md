# AGENT.md: Guidelines for AI Agents & Developers

This document serves as the primary technical reference and operating manual for AI coding agents and human engineers working on the **Book-NCF** codebase.

---

## 1. Project Overview & Architecture Principles

- **Objective**: Neural Collaborative Filtering (NCF) for Top-K book recommendations under implicit feedback.
- **Key References**: Xiangnan He et al. (WWW 2017) *Neural Collaborative Filtering*.
- **Core Design Principles**:
  - **Single Responsibility Principle**: Data processing (`src/data`), model implementations (`src/models`), evaluation and training (`src/engine`), and utilities (`src/utils`) are strictly decoupled.
  - **Reproducibility**: All random seeds across Python, NumPy, and PyTorch (CPU & CUDA) must be controlled via `src/utils/seed.py`.
  - **No Hardcoding**: All hyperparameters, data paths, negative sampling ratios, batch sizes, and thresholds must be declared in `configs/config.yaml` or injected via CLI arguments in `train.py`.
  - **Batched Operations**: Ranking evaluation (`src/engine/evaluator.py`) and inference (`inference.py`) must be executed in vectorized or batched tensor format to avoid OOM and slow loops.

---

## 2. Directory Layout & Key Responsibilities

```text
├── configs/
│   └── config.yaml          # Single source of truth for all hyperparameters & paths
├── data/
│   ├── raw/                 # Raw datasets (ratings.csv, books.csv) - Untracked by git
│   └── processed/           # Processed datasets (train.csv, val_negatives.csv, test_negatives.csv, id_mappings.pkl)
├── notebooks/
│   ├── eda.ipynb            # Exploratory data analysis (sparsity, long-tail distributions)
│   └── colab_runner.ipynb   # Complete execution flow for Google Colab environments
├── src/
│   ├── __init__.py          # Package root
│   ├── data/
│   │   ├── preprocess.py    # K-core filtering, ID mapping, Leave-One-Out split
│   │   └── dataset.py       # Dynamic negative sampling PyTorch Dataset
│   ├── models/
│   │   ├── gmf.py           # Generalized Matrix Factorization module
│   │   ├── mlp.py           # Multi-Layer Perceptron module
│   │   └── neumf.py         # NeuMF module combining GMF & MLP (+ pretrained loading)
│   ├── engine/
│   │   ├── trainer.py       # Training loop, early stopping, checkpointing
│   │   └── evaluator.py     # HR@K, NDCG@K metrics on test_negatives
│   └── utils/
│       ├── seed.py          # Deterministic seeding helper
│       └── logger.py        # Stream/File logging & JSON metric history recorder
├── train.py                 # CLI entry-point for experiments
├── inference.py             # CLI entry-point for Top-K inference for a target user
├── requirements.txt         # Pinned / minimal required packages
└── .gitignore               # Ignored artifacts (models, caches, data files)
```

---

## 3. Data Flow & Contract Specifications

### 3.1. Raw Input Expectations
- `data/raw/ratings.csv`: Must have at least `user_id`, `book_id`, and `rating`. An optional `timestamp` column will be used for temporal Leave-One-Out splitting.
- `data/raw/books.csv`: Contains metadata for displaying titles, authors, etc.

### 3.2. Processed Artifacts
- `id_mappings.pkl`: Dictionary holding:
  - `user2idx`: mapping from raw user ID to continuous integer `[0, num_users - 1]`.
  - `idx2user`: inverse mapping.
  - `item2idx`: mapping from raw book ID to continuous integer `[0, num_items - 1]`.
  - `idx2item`: inverse mapping.
  - `num_users`: total distinct users after K-core filtering.
  - `num_items`: total distinct items after K-core filtering.
- `train.csv`: Training set containing `user_idx`, `item_idx`, and implicit interaction `1` (interactions $1 \dots T-2$).
- `val_negatives.csv`: Validation set containing `user_idx`, positive item ($T-1$), and 99 negative items (`neg_0` to `neg_98`) used for early stopping and hyperparameter tuning.
- `test_negatives.csv`: Evaluation set containing `user_idx`, positive item ($T$), and 99 negative items (`neg_0` to `neg_98`) used strictly for final unbiased reporting.

---

## 4. Development & Extension Protocols

### Adding a New Model Architecture
1. Implement your model inside `src/models/new_model.py`.
2. Inherit from `torch.nn.Module`.
3. Provide a standard `forward(user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor` returning 1D predicted interaction probabilities (shape: `(batch_size,)`).
4. Register the model in `src/models/__init__.py` and in `build_model()` inside `train.py`.
5. Add appropriate hyperparameters in `configs/config.yaml`.

### Modifying Negative Sampling
- Dynamic negative sampling happens inside `NCFDataset.sample_negatives()` before every epoch.
- If implementing popularity-biased or hard negative sampling, modify `NCFDataset` while preserving the return contract: `(user_tensor, item_tensor, label_tensor)`.

### Checkpoint and Experiment Logging
- Models are automatically saved under `checkpoints/best_model.pt` based on the monitored metric (default: `ndcg@10`).
- Experiment logs are appended to `logs/experiment.log`.
- Complete epoch history (Loss, HR@5/10/20, NDCG@5/10/20) is recorded into `logs/training_history.json`.

---

## 5. Standard CLI Commands

```bash
# Run data preprocessing
python -m src.data.preprocess --config configs/config.yaml

# Train NeuMF (From Scratch)
python train.py --config configs/config.yaml --model neumf --epochs 20

# Train GMF
python train.py --config configs/config.yaml --model gmf --epochs 15

# Train MLP
python train.py --config configs/config.yaml --model mlp --epochs 15

# Train NeuMF with Pretrained Weights
python train.py --config configs/config.yaml --model neumf \
    --pretrained_gmf checkpoints/gmf_best.pt \
    --pretrained_mlp checkpoints/mlp_best.pt

# Resume / Fine-tune from an existing checkpoint
python train.py --resume checkpoints/best_model.pt --epochs 10 --lr 0.0002

# Run Top-K Inference for User ID 123
python inference.py --user_id 123 --top_k 10 --checkpoint checkpoints/best_model.pt
```
