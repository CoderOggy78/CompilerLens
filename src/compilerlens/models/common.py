"""Common neural network layers and building blocks for CompilerLens models."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron with LayerNorm, ReLU activations, and Dropout."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        layers: List[nn.Module] = []

        curr_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if use_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Modulates node representation h using conditioning vector c:
        output = gamma(c) * h + beta(c)
    """

    def __init__(self, node_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.scale_proj = nn.Linear(condition_dim, node_dim)
        self.shift_proj = nn.Linear(condition_dim, node_dim)

        # Initialize to identity transform (scale=1, shift=0)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Args:

        h: Node representations of shape (..., node_dim)
        c: Conditioning representations of shape (..., condition_dim)
        """
        gamma = self.scale_proj(c) + 1.0  # (..., node_dim)
        beta = self.shift_proj(c)         # (..., node_dim)
        return gamma * h + beta


class GlobalPooling(nn.Module):
    """Graph pooling module combining mean and max pooling across nodes."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.out_dim = in_dim * 2

    def forward(
        self,
        node_embeds: torch.Tensor,
        graph_node_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Args:

        node_embeds: (Total_Nodes, D)
        graph_node_idx: (Total_Nodes,) indicating graph membership [0 ... num_graphs - 1]
        num_graphs: Number of graphs in the batch.

        Returns:
            (num_graphs, 2 * D) pooled graph embedding.
        """
        device = node_embeds.device
        dim = node_embeds.size(-1)

        # Mean pooling
        sum_embeds = torch.zeros((num_graphs, dim), device=device, dtype=node_embeds.dtype)
        counts = torch.zeros((num_graphs, 1), device=device, dtype=node_embeds.dtype)
        ones = torch.ones((graph_node_idx.size(0), 1), device=device, dtype=node_embeds.dtype)

        sum_embeds.index_add_(0, graph_node_idx, node_embeds)
        counts.index_add_(0, graph_node_idx, ones)
        mean_embeds = sum_embeds / torch.clamp(counts, min=1.0)

        # Max pooling
        max_embeds = torch.full((num_graphs, dim), -1e9, device=device, dtype=node_embeds.dtype)
        # Scatter reduce with max
        max_embeds.scatter_reduce_(0, graph_node_idx.unsqueeze(-1).expand(-1, dim), node_embeds, reduce="amax", include_self=False)
        # Handle empty cases if any
        max_embeds = torch.where(max_embeds < -1e8, torch.zeros_like(max_embeds), max_embeds)

        return torch.cat([mean_embeds, max_embeds], dim=-1)
