"""Configuration-conditioned, direction-aware Graph Neural Network for compiler configuration ranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compilerlens.models.common import MLP, FiLM, GlobalPooling


class DirectionAwareMessagePassingLayer(nn.Module):
    """Direction-aware graph convolution layer with separate forward and reverse message paths.

    Maintains distinct relation projections for:
      - Forward dataflow edges (producers -> consumers)
      - Reverse dependency edges (consumers -> producers)
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.fwd_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rev_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.comb_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        fwd_edge_index: torch.Tensor,
        rev_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

        h: Node representations of shape (N, hidden_dim)
        fwd_edge_index: (2, E) forward edges
        rev_edge_index: (2, E) reverse edges
        """
        num_nodes = h.size(0)
        device = h.device

        # 1. Forward message passing
        if fwd_edge_index.size(1) > 0:
            src_fwd, dst_fwd = fwd_edge_index[0], fwd_edge_index[1]
            msg_fwd = self.fwd_proj(h[src_fwd])  # (E, D)
            agg_fwd = torch.zeros((num_nodes, self.hidden_dim), device=device, dtype=h.dtype)
            counts_fwd = torch.zeros((num_nodes, 1), device=device, dtype=h.dtype)
            ones_fwd = torch.ones((fwd_edge_index.size(1), 1), device=device, dtype=h.dtype)

            agg_fwd.index_add_(0, dst_fwd, msg_fwd)
            counts_fwd.index_add_(0, dst_fwd, ones_fwd)
            mean_fwd = agg_fwd / torch.clamp(counts_fwd, min=1.0)
        else:
            mean_fwd = torch.zeros_like(h)

        # 2. Reverse message passing
        if rev_edge_index.size(1) > 0:
            src_rev, dst_rev = rev_edge_index[0], rev_edge_index[1]
            msg_rev = self.rev_proj(h[src_rev])
            agg_rev = torch.zeros((num_nodes, self.hidden_dim), device=device, dtype=h.dtype)
            counts_rev = torch.zeros((num_nodes, 1), device=device, dtype=h.dtype)
            ones_rev = torch.ones((rev_edge_index.size(1), 1), device=device, dtype=h.dtype)

            agg_rev.index_add_(0, dst_rev, msg_rev)
            counts_rev.index_add_(0, dst_rev, ones_rev)
            mean_rev = agg_rev / torch.clamp(counts_rev, min=1.0)
        else:
            mean_rev = torch.zeros_like(h)

        # 3. Combine directional messages
        comb_msg = self.comb_proj(torch.cat([mean_fwd, mean_rev], dim=-1))

        # 4. Residual update & LayerNorm
        update_in = torch.cat([h, comb_msg], dim=-1)
        h_next = self.norm(h + self.update_mlp(update_in))
        return h_next


class ConfigConditionedGNN(nn.Module):
    """Configuration-conditioned, direction-aware GNN for learned compiler cost modeling.

    Integrates candidate compiler configurations directly into the graph message passing
    via feature-wise linear modulation (FiLM) or concatenation, propagating configuration
    semantics throughout the IR dependency graph.
    """

    def __init__(
        self,
        node_feat_dim: int = 140,
        config_feat_dim: int = 24,
        num_opcodes: int = 128,
        opcode_embed_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        conditioning: str = "film",  # "film" or "concat"
        bidirectional: bool = True,
        use_opcode_embeddings: bool = True,
        use_graph_edges: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conditioning_type = conditioning
        self.bidirectional = bidirectional
        self.use_opcode_embeddings = use_opcode_embeddings
        self.use_graph_edges = use_graph_edges
        self.num_layers = num_layers

        # Node featurization
        self.opcode_dim = opcode_embed_dim if use_opcode_embeddings else 0
        if use_opcode_embeddings:
            self.opcode_embed = nn.Embedding(num_opcodes, opcode_embed_dim)

        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim - self.opcode_dim),
            nn.LayerNorm(hidden_dim - self.opcode_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Configuration encoder
        self.config_encoder = MLP(
            in_dim=config_feat_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

        # Conditioning mechanism
        if self.conditioning_type == "film":
            self.film = FiLM(node_dim=hidden_dim, condition_dim=hidden_dim)
        elif self.conditioning_type == "concat":
            self.concat_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            DirectionAwareMessagePassingLayer(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Global pooling
        self.pooling = GlobalPooling(hidden_dim)

        # Final prediction head
        self.head = MLP(
            in_dim=(hidden_dim * 2) + hidden_dim,
            out_dim=1,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )

    def extract_base_node_repr(
        self, node_feat: torch.Tensor, node_opcode: torch.Tensor
    ) -> torch.Tensor:
        """Compute configuration-independent base node representations."""
        feat_emb = self.node_proj(node_feat)
        if self.use_opcode_embeddings:
            clamped_opcodes = torch.clamp(node_opcode, 0, self.opcode_embed.num_embeddings - 1)
            op_emb = self.opcode_embed(clamped_opcodes)
            h_base = torch.cat([op_emb, feat_emb], dim=-1)
        else:
            h_base = feat_emb
        return h_base

    def forward_graph_candidates(
        self,
        h_base: torch.Tensor,
        fwd_edge_index: torch.Tensor,
        rev_edge_index: torch.Tensor,
        configs: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute predicted log normalized runtimes for candidate configurations of a single graph.

        Args:
            h_base: (N, hidden_dim) configuration-independent node embeddings.
            fwd_edge_index: (2, E_fwd) forward edges for this graph.
            rev_edge_index: (2, E_rev) reverse edges for this graph.
            configs: (C, 24) configuration features.
            chunk_size: Optional integer to chunk candidates and cap peak memory.

        Returns:
            (C,) predicted log normalized runtimes.
        """
        num_candidates = configs.size(0)
        num_nodes = h_base.size(0)

        if chunk_size is not None and chunk_size < num_candidates:
            # Process in chunks to maintain strict O(1) memory per candidate batch
            all_scores = []
            for start_idx in range(0, num_candidates, chunk_size):
                end_idx = min(start_idx + chunk_size, num_candidates)
                chunk_configs = configs[start_idx:end_idx]
                chunk_scores = self.forward_graph_candidates(
                    h_base, fwd_edge_index, rev_edge_index, chunk_configs, chunk_size=None
                )
                all_scores.append(chunk_scores)
            return torch.cat(all_scores, dim=0)

        # 1. Encode configurations
        c_emb = self.config_encoder(configs)  # (C, hidden_dim)
        K = configs.size(0)

        # If configuration conditioning is active:
        # Replicate h_base across K configurations: (K, N, hidden_dim)
        h_k = h_base.unsqueeze(0).expand(K, -1, -1)  # (K, N, D)
        c_k = c_emb.unsqueeze(1).expand(-1, num_nodes, -1)  # (K, N, D)

        if self.conditioning_type == "film":
            h = self.film(h_k, c_k)  # (K, N, D)
        elif self.conditioning_type == "concat":
            h = F.relu(self.concat_proj(torch.cat([h_k, c_k], dim=-1)))
        else:
            h = h_k

        # 2. Reshape for batched message passing: (K * N, D)
        h_flat = h.reshape(K * num_nodes, self.hidden_dim)

        if self.use_graph_edges and (fwd_edge_index.size(1) > 0 or rev_edge_index.size(1) > 0):
            # Tile edges across K candidates with node offsets
            offsets = (torch.arange(K, device=configs.device) * num_nodes).view(-1, 1, 1)

            if fwd_edge_index.size(1) > 0:
                fwd_tiled = (fwd_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            else:
                fwd_tiled = torch.empty((2, 0), dtype=torch.long, device=configs.device)

            if rev_edge_index.size(1) > 0 and self.bidirectional:
                rev_tiled = (rev_edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
            else:
                rev_tiled = torch.empty((2, 0), dtype=torch.long, device=configs.device)

            # Apply GNN message passing layers
            for layer in self.gnn_layers:
                h_flat = layer(h_flat, fwd_tiled, rev_tiled)

        # 3. Graph pooling across each candidate graph
        candidate_node_idx = torch.arange(K, device=configs.device).repeat_interleave(num_nodes)
        pooled = self.pooling(h_flat, candidate_node_idx, K)  # (K, 2 * hidden_dim)

        # 4. Final prediction head
        combined = torch.cat([pooled, c_emb], dim=-1)  # (K, 3 * hidden_dim)
        scores = self.head(combined).squeeze(-1)  # (K,)
        return scores

    def forward_batch(
        self, batch: Dict[str, Any], chunk_size: Optional[int] = 64
    ) -> List[torch.Tensor]:
        """Compute candidate score predictions for all graphs in the collated batch."""
        node_feat = batch["node_feat"]
        node_opcode = batch["node_opcode"]
        graph_node_idx = batch["graph_node_idx"]
        num_graphs = batch["num_graphs"]
        fwd_edge_index = batch["fwd_edge_index"]
        rev_edge_index = batch["rev_edge_index"]
        graph_configs = batch["graph_configs"]

        # Base node representations
        h_all = self.extract_base_node_repr(node_feat, node_opcode)

        graph_predictions = []
        node_offset = 0

        for g_idx in range(num_graphs):
            # Extract nodes for graph g
            node_mask = graph_node_idx == g_idx
            num_g_nodes = node_mask.sum().item()
            h_g = h_all[node_mask]

            # Extract edges for graph g
            if fwd_edge_index.size(1) > 0:
                edge_mask = (fwd_edge_index[0] >= node_offset) & (
                    fwd_edge_index[0] < node_offset + num_g_nodes
                )
                g_fwd_edges = fwd_edge_index[:, edge_mask] - node_offset
                g_rev_edges = rev_edge_index[:, edge_mask] - node_offset
            else:
                g_fwd_edges = torch.empty((2, 0), dtype=torch.long, device=node_feat.device)
                g_rev_edges = torch.empty((2, 0), dtype=torch.long, device=node_feat.device)

            configs_g = graph_configs[g_idx]
            scores_g = self.forward_graph_candidates(
                h_g, g_fwd_edges, g_rev_edges, configs_g, chunk_size=chunk_size
            )
            graph_predictions.append(scores_g)

            node_offset += num_g_nodes

        return graph_predictions
