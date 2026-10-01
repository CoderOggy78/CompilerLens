"""GraphSAGE baseline model for computation graph encoding and configuration ranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from compilerlens.models.common import MLP, GlobalPooling


class GraphSAGELayer(nn.Module):
    """A single GraphSAGE message passing layer with mean neighbor aggregation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

        h: (N, in_dim) node representations
        edge_index: (2, E) directed edges [src, dst]
        """
        num_nodes = h.size(0)
        device = h.device

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            # Aggregate incoming messages by destination node
            msg = h[src]  # (E, in_dim)
            agg = torch.zeros((num_nodes, h.size(1)), device=device, dtype=h.dtype)
            counts = torch.zeros((num_nodes, 1), device=device, dtype=h.dtype)
            ones = torch.ones((edge_index.size(1), 1), device=device, dtype=h.dtype)

            agg.index_add_(0, dst, msg)
            counts.index_add_(0, dst, ones)
            neigh_mean = agg / torch.clamp(counts, min=1.0)
        else:
            neigh_mean = torch.zeros_like(h)

        concat_h = torch.cat([h, neigh_mean], dim=-1)
        out = self.linear(concat_h)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.norm(out + self.res_proj(h))
        return out


class GraphSAGEModel(nn.Module):
    """Standard GraphSAGE baseline: encodes graph connectivity without configuration conditioning."""

    def __init__(
        self,
        node_feat_dim: int = 140,
        config_feat_dim: int = 24,
        num_opcodes: int = 128,
        opcode_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.opcode_embed = nn.Embedding(num_opcodes, opcode_embed_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim - opcode_embed_dim),
            nn.LayerNorm(hidden_dim - opcode_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pooling = GlobalPooling(hidden_dim)

        self.config_encoder = MLP(
            in_dim=config_feat_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.head = MLP(
            in_dim=(hidden_dim * 2) + hidden_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward_batch(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        """Compute candidate score predictions for all graphs in the batch."""
        node_feat = batch["node_feat"]
        node_opcode = torch.clamp(batch["node_opcode"], 0, self.opcode_embed.num_embeddings - 1)
        graph_node_idx = batch["graph_node_idx"]
        num_graphs = batch["num_graphs"]
        fwd_edges = batch["fwd_edge_index"]
        rev_edges = batch["rev_edge_index"]
        graph_configs = batch["graph_configs"]

        # Undirected edge union for standard GraphSAGE baseline
        if fwd_edges.size(1) > 0 and rev_edges.size(1) > 0:
            edges = torch.cat([fwd_edges, rev_edges], dim=1)
        else:
            edges = fwd_edges

        # Initial node representations
        h = torch.cat([self.opcode_embed(node_opcode), self.node_proj(node_feat)], dim=-1)

        # Message passing layers
        for layer in self.layers:
            h = layer(h, edges)

        # Global graph pooling
        pooled_graphs = self.pooling(h, graph_node_idx, num_graphs)  # (num_graphs, 2 * hidden_dim)

        graph_predictions = []
        for g_idx in range(num_graphs):
            configs = graph_configs[g_idx]
            c_emb = self.config_encoder(configs)

            g_rep = pooled_graphs[g_idx : g_idx + 1].expand(configs.size(0), -1)
            combined = torch.cat([g_rep, c_emb], dim=-1)
            scores = self.head(combined).squeeze(-1)
            graph_predictions.append(scores)

        return graph_predictions
