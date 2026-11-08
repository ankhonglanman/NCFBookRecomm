"""
Project: Book-NCF
Module: src.models.gmf
Description: Generalized Matrix Factorization (GMF) standalone branch.
"""

import torch
import torch.nn as nn


class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF) model.
    Computes an element-wise product of user and item latent vectors,
    projected through a linear output layer with a Sigmoid activation.
    """

    def __init__(self, num_users: int, num_items: int, latent_dim: int = 32):
        """
        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            latent_dim: Dimensionality of latent embedding vectors.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim

        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.affine_output = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        """Weight initialization following standard NCF protocol."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.affine_output.weight, nonlinearity="sigmoid")
        if self.affine_output.bias is not None:
            nn.init.zeros_(self.affine_output.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GMF.

        Args:
            user_indices: Tensor of shape (batch_size,) with user IDs.
            item_indices: Tensor of shape (batch_size,) with item IDs.

        Returns:
            Predicted interaction probabilities of shape (batch_size,).
        """
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        # Element-wise product (Hadamard product)
        element_product = user_emb * item_emb
        logits = self.affine_output(element_product)
        rating_pred = self.sigmoid(logits)
        return rating_pred.view(-1)
