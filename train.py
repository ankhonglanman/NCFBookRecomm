"""
Project: Book-NCF (Neural Collaborative Filtering for Book Recommendation)
File: train.py
Description: Main entry-point script to train GMF, MLP, or NeuMF models.
             Parses CLI arguments, loads configs, orchestrates dataset,
             evaluator, trainer, and saves checkpoints.
"""

import os
import argparse
import pickle
import yaml
import pandas as pd
import torch

from src.utils.seed import set_seed
from src.utils.logger import setup_logger, MetricTracker
from src.data.dataset import NCFDataset
from src.data.preprocess import DataPreprocessor
from src.models.gmf import GMF
from src.models.mlp import MLP
from src.models.neumf import NeuMF
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Book-NCF Recommendation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, choices=["gmf", "mlp", "neumf"], default=None, help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Target device")
    parser.add_argument("--pretrained_gmf", type=str, default=None, help="Pretrained GMF weights for NeuMF")
    parser.add_argument("--pretrained_mlp", type=str, default=None, help="Pretrained MLP weights for NeuMF")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_type: str, num_users: int, num_items: int, config: dict):
    model_cfg = config.get("model", {})
    model_type = model_type.lower()

    if model_type == "gmf":
        latent_dim = model_cfg.get("latent_dim_gmf", 32)
        return GMF(num_users=num_users, num_items=num_items, latent_dim=latent_dim)

    elif model_type == "mlp":
        latent_dim = model_cfg.get("latent_dim_mlp", 32)
        hidden_dims = model_cfg.get("mlp_hidden_dims", [64, 32, 16])
        dropout = model_cfg.get("dropout", 0.2)
        return MLP(num_users=num_users, num_items=num_items, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)

    elif model_type == "neumf":
        latent_dim_gmf = model_cfg.get("latent_dim_gmf", 32)
        latent_dim_mlp = model_cfg.get("latent_dim_mlp", 32)
        hidden_dims = model_cfg.get("mlp_hidden_dims", [64, 32, 16])
        dropout = model_cfg.get("dropout", 0.2)
        
        model = NeuMF(
            num_users=num_users,
            num_items=num_items,
            latent_dim_gmf=latent_dim_gmf,
            latent_dim_mlp=latent_dim_mlp,
            mlp_hidden_dims=hidden_dims,
            dropout=dropout
        )

        gmf_path = model_cfg.get("pretrained_gmf_path")
        mlp_path = model_cfg.get("pretrained_mlp_path")
        if gmf_path and mlp_path:
            model.load_pretrained_weights(gmf_path, mlp_path)
            
        return model

    else:
        raise ValueError(f"Unsupported model architecture: {model_type}")


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI Overrides
    if args.model:
        config["model"]["model_type"] = args.model
    if args.epochs:
        config["train"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["learning_rate"] = args.lr
    if args.device:
        config["train"]["device"] = args.device
    if args.pretrained_gmf:
        config["model"]["pretrained_gmf_path"] = args.pretrained_gmf
    if args.pretrained_mlp:
        config["model"]["pretrained_mlp_path"] = args.pretrained_mlp

    # Set random seed
    seed = config.get("train", {}).get("seed", 42)
    set_seed(seed)

    # Setup logger and metric tracker
    log_cfg = config.get("logging", {})
    logger = setup_logger(
        name="BookNCF",
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "experiment.log")
    )
    metric_tracker = MetricTracker(
        output_path=os.path.join(log_cfg.get("log_dir", "logs"), log_cfg.get("history_file", "training_history.json"))
    )

    logger.info("Initializing Book-NCF Training Pipeline...")

    # Data check
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "data/processed/train.csv")
    val_neg_path = data_cfg.get("val_negatives_path", "data/processed/val_negatives.csv")
    test_neg_path = data_cfg.get("test_negatives_path", "data/processed/test_negatives.csv")
    mappings_path = data_cfg.get("mappings_path", "data/processed/id_mappings.pkl")

    if not os.path.exists(train_path) or not os.path.exists(mappings_path) or not os.path.exists(test_neg_path):
        logger.warning("Processed files not found. Attempting automatic preprocessing...")
        preprocessor = DataPreprocessor(config)
        preprocessor.run()

    logger.info(f"Loading id mappings from: {mappings_path}")
    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    num_users = mappings["num_users"]
    num_items = mappings["num_items"]
    logger.info(f"Dataset stats: {num_users:,} unique users, {num_items:,} unique books")

    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df):,} training interactions.")

    dataset = NCFDataset(
        train_df=train_df,
        num_users=num_users,
        num_items=num_items,
        num_negatives=data_cfg.get("num_negatives_train", 4),
        seed=seed
    )

    eval_cfg = config.get("evaluation", {})
    target_device = "cuda" if torch.cuda.is_available() and config["train"].get("device") == "cuda" else "cpu"
    
    # 3-Way Split Evaluators: Validation & Test
    eval_val_path = val_neg_path if os.path.exists(val_neg_path) else test_neg_path
    val_evaluator = Evaluator(
        test_negatives_path=eval_val_path,
        top_k_list=eval_cfg.get("top_k", [5, 10, 20]),
        batch_size=eval_cfg.get("eval_batch_size", 100),
        device=target_device
    )

    test_evaluator = None
    if os.path.exists(val_neg_path) and os.path.exists(test_neg_path):
        test_evaluator = Evaluator(
            test_negatives_path=test_neg_path,
            top_k_list=eval_cfg.get("top_k", [5, 10, 20]),
            batch_size=eval_cfg.get("eval_batch_size", 100),
            device=target_device
        )

    # Initialize model
    model_type = config.get("model", {}).get("model_type", "neumf")
    model = build_model(model_type, num_users, num_items, config)

    # Trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        val_evaluator=val_evaluator,
        config=config,
        logger=logger,
        test_evaluator=test_evaluator,
        metric_tracker=metric_tracker
    )

    results = trainer.fit()
    logger.info(f"Experiment finished successfully: {results}")


if __name__ == "__main__":
    main()
