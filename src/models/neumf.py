"""
Project: Book-NCF
Module: src.models.neumf
Description: Neural Matrix Factorization (NeuMF) combining GMF and MLP branches,
             with support for loading pretrained weights from standalone models.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .gmf import GMF
from .mlp import MLP


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) model.
    Combines linear interactions from Generalized Matrix Factorization (GMF)
    and non-linear representations from Multi-Layer Perceptron (MLP).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        """
        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            latent_dim_gmf: Latent dimension for GMF branch embeddings.
            latent_dim_mlp: Latent dimension for MLP branch embeddings.
            mlp_hidden_dims: List of hidden layer dimensions for MLP.
            dropout: Dropout probability in MLP layers.
        """
        super().__init__()
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [64, 32, 16]

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_gmf = latent_dim_gmf
        self.latent_dim_mlp = latent_dim_mlp
        self.mlp_hidden_dims = mlp_hidden_dims

        # GMF Branch Embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embedding_gmf = nn.Embedding(num_items, latent_dim_gmf)

        # MLP Branch Embeddings
        self.user_embedding_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, latent_dim_mlp)

        # MLP Layers
        layers = []
        input_dim = latent_dim_mlp * 2
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*layers)

        # Final NeuMF output layer combining GMF and MLP representations
        self.affine_output = nn.Linear(latent_dim_gmf + mlp_hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """Random weight initialization for from-scratch training."""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.kaiming_uniform_(self.affine_output.weight, nonlinearity="sigmoid")
        if self.affine_output.bias is not None:
            nn.init.zeros_(self.affine_output.bias)

    def load_pretrained_weights(
        self,
        gmf_path: str,
        mlp_path: str,
        alpha: float = 0.5,
        map_location: str = "cpu"
    ) -> None:
        """
        Loads pretrained weights from separate GMF and MLP checkpoint files.

        Args:
            gmf_path: Path to saved GMF state_dict or model checkpoint.
            mlp_path: Path to saved MLP state_dict or model checkpoint.
            alpha: Weighting factor between GMF and MLP for final linear layer.
            map_location: Device to map loaded tensors to.
        """
        print(f"Loading pretrained GMF weights from: {gmf_path}")
        gmf_checkpoint = torch.load(gmf_path, map_location=map_location)
        gmf_dict = gmf_checkpoint.get("model_state_dict", gmf_checkpoint)

        self.user_embedding_gmf.weight.data.copy_(gmf_dict["user_embedding.weight"])
        self.item_embedding_gmf.weight.data.copy_(gmf_dict["item_embedding.weight"])

        print(f"Loading pretrained MLP weights from: {mlp_path}")
        mlp_checkpoint = torch.load(mlp_path, map_location=map_location)
        mlp_dict = mlp_checkpoint.get("model_state_dict", mlp_checkpoint)

        self.user_embedding_mlp.weight.data.copy_(mlp_dict["user_embedding.weight"])
        self.item_embedding_mlp.weight.data.copy_(mlp_dict["item_embedding.weight"])

        # Copy MLP hidden layers
        for key in mlp_dict:
            if key.startswith("mlp_layers."):
                self.state_dict()[key].data.copy_(mlp_dict[key])

        # Concatenate prediction layer weights
        gmf_weight = gmf_dict["affine_output.weight"]
        mlp_weight = mlp_dict["affine_output.weight"]
        combined_weight = torch.cat([alpha * gmf_weight, (1.0 - alpha) * mlp_weight], dim=1)
        self.affine_output.weight.data.copy_(combined_weight)

        # Average bias
        if self.affine_output.bias is not None and "affine_output.bias" in gmf_dict and "affine_output.bias" in mlp_dict:
            combined_bias = 0.5 * (gmf_dict["affine_output.bias"] + mlp_dict["affine_output.bias"])
            self.affine_output.bias.data.copy_(combined_bias)

        print("Pretrained weights loaded successfully into NeuMF.")

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for NeuMF.

        Args:
            user_indices: Tensor of shape (batch_size,) with user IDs.
            item_indices: Tensor of shape (batch_size,) with item IDs.

        Returns:
            Predicted interaction probabilities of shape (batch_size,).
        """
        # GMF Branch
        gmf_user = self.user_embedding_gmf(user_indices)
        gmf_item = self.item_embedding_gmf(item_indices)
        phi_gmf = gmf_user * gmf_item

        # MLP Branch
        mlp_user = self.user_embedding_mlp(user_indices)
        mlp_item = self.item_embedding_mlp(item_indices)
        phi_mlp = self.mlp_layers(torch.cat([mlp_user, mlp_item], dim=-1))

        # Fusion layer
        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        logits = self.affine_output(fusion)
        rating_pred = self.sigmoid(logits)
        return rating_pred.view(-1)
