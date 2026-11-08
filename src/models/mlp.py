"""
Project: Book-NCF
Module: src.models.mlp
Description: Multi-Layer Perceptron (MLP) standalone branch.
"""

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model for Collaborative Filtering.
    Concatenates user and item embedding vectors and feeds them into
    deep fully-connected layers with non-linear activations and dropout.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            latent_dim: Size of initial user and item embedding vectors.
            hidden_dims: List specifying hidden layer neuron dimensions.
            dropout: Dropout probability between dense layers.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        layers = []
        input_dim = latent_dim * 2  # Concatenation of user and item embeddings
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.mlp_layers = nn.Sequential(*layers)
        self.affine_output = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """Weight initialization with Kaiming Normal for ReLU activations."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.kaiming_uniform_(self.affine_output.weight, nonlinearity="sigmoid")
        if self.affine_output.bias is not None:
            nn.init.zeros_(self.affine_output.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP.

        Args:
            user_indices: Tensor of shape (batch_size,) with user IDs.
            item_indices: Tensor of shape (batch_size,) with item IDs.

        Returns:
            Predicted interaction probabilities of shape (batch_size,).
        """
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        interaction = torch.cat([user_emb, item_emb], dim=-1)
        mlp_vector = self.mlp_layers(interaction)
        logits = self.affine_output(mlp_vector)
        rating_pred = self.sigmoid(logits)
        return rating_pred.view(-1)
