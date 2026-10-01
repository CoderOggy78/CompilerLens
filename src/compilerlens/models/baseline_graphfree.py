"""Graph-free neural baseline: pools node representations without graph message passing."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from compilerlens.models.common import MLP, GlobalPooling


class GraphFreeModel(nn.Module):
    """Neural baseline that completely ignores graph edge connectivity.

    Encodes individual node attributes and pools them globally, testing whether graph
    message passing is actually needed to rank compiler configurations.
    """

    def __init__(
        self,
        node_feat_dim: int = 140,
        config_feat_dim: int = 24,
        num_opcodes: int = 128,
        opcode_embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Opcode embedding
        self.opcode_embed = nn.Embedding(num_opcodes, opcode_embed_dim)
        # Node continuous feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim - opcode_embed_dim),
            nn.LayerNorm(hidden_dim - opcode_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Graph pooling
        self.pooling = GlobalPooling(hidden_dim)

        # Configuration encoder
        self.config_encoder = MLP(
            in_dim=config_feat_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Readout head: pooled graph representation (2 * hidden_dim) + config (hidden_dim)
        self.head = MLP(
            in_dim=(hidden_dim * 2) + hidden_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward_batch(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        """Compute predicted log normalized runtimes for a collated batch of graphs.

        Returns:
            List of 1D tensors, one per graph in the batch, shape (C_g,).
        """
        node_feat = batch["node_feat"]
        node_opcode = torch.clamp(batch["node_opcode"], 0, self.opcode_embed.num_embeddings - 1)
        graph_node_idx = batch["graph_node_idx"]
        num_graphs = batch["num_graphs"]
        graph_configs = batch["graph_configs"]

        # Initial node representations
        opcode_emb = self.opcode_embed(node_opcode)
        feat_emb = self.node_proj(node_feat)
        h = torch.cat([opcode_emb, feat_emb], dim=-1)  # (Total_Nodes, hidden_dim)

        # Pool across nodes of each graph (without any message passing)
        pooled_graphs = self.pooling(h, graph_node_idx, num_graphs)  # (num_graphs, 2 * hidden_dim)

        graph_predictions = []
        for g_idx in range(num_graphs):
            configs = graph_configs[g_idx]  # (C_g, 24)
            c_emb = self.config_encoder(configs)  # (C_g, hidden_dim)

            g_rep = pooled_graphs[g_idx : g_idx + 1]  # (1, 2 * hidden_dim)
            g_rep_expanded = g_rep.expand(configs.size(0), -1)  # (C_g, 2 * hidden_dim)

            combined = torch.cat([g_rep_expanded, c_emb], dim=-1)  # (C_g, 3 * hidden_dim)
            scores = self.head(combined).squeeze(-1)  # (C_g,)
            graph_predictions.append(scores)

        return graph_predictions
