"""Feature preprocessing, standardization, and graph-summary feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from compilerlens.data import GraphData


class FeaturePreprocessor:
    """Standardizes continuous node and configuration features strictly fit on training data.

    Guarantees no data leakage: target runtimes, normalizers, and graph identifiers are NEVER
    passed or processed.
    """

    def __init__(
        self,
        node_feat_dim: int = 140,
        config_feat_dim: int = 24,
    ) -> None:
        self.node_feat_dim = node_feat_dim
        self.config_feat_dim = config_feat_dim

        self.node_mean: Optional[np.ndarray] = None
        self.node_std: Optional[np.ndarray] = None
        self.config_mean: Optional[np.ndarray] = None
        self.config_std: Optional[np.ndarray] = None
        self.is_fit: bool = False

    def fit(self, training_graphs: Sequence[GraphData]) -> FeaturePreprocessor:
        """Fit empirical mean and standard deviation on training graphs only."""
        if not training_graphs:
            raise ValueError("Cannot fit preprocessor on an empty collection of graphs.")

        # Accumulate node features
        total_nodes = sum(g.num_nodes for g in training_graphs)
        node_sum = np.zeros(self.node_feat_dim, dtype=np.float64)
        node_sq_sum = np.zeros(self.node_feat_dim, dtype=np.float64)

        total_configs = sum(g.num_configs for g in training_graphs)
        config_sum = np.zeros(self.config_feat_dim, dtype=np.float64)
        config_sq_sum = np.zeros(self.config_feat_dim, dtype=np.float64)

        for g in training_graphs:
            nf = g.node_feat.astype(np.float64)
            node_sum += np.sum(nf, axis=0)
            node_sq_sum += np.sum(nf**2, axis=0)

            cf = g.config_feat.astype(np.float64)
            config_sum += np.sum(cf, axis=0)
            config_sq_sum += np.sum(cf**2, axis=0)

        # Compute statistics
        self.node_mean = (node_sum / max(total_nodes, 1)).astype(np.float32)
        node_var = (node_sq_sum / max(total_nodes, 1)) - (self.node_mean.astype(np.float64) ** 2)
        node_var = np.maximum(node_var, 0.0)
        self.node_std = np.sqrt(node_var).astype(np.float32)
        # Avoid division by zero on constant features
        self.node_std = np.where(self.node_std < 1e-6, 1.0, self.node_std)

        self.config_mean = (config_sum / max(total_configs, 1)).astype(np.float32)
        config_var = (config_sq_sum / max(total_configs, 1)) - (self.config_mean.astype(np.float64) ** 2)
        config_var = np.maximum(config_var, 0.0)
        self.config_std = np.sqrt(config_var).astype(np.float32)
        self.config_std = np.where(self.config_std < 1e-6, 1.0, self.config_std)

        self.is_fit = True
        return self

    def transform_node_feat(
        self, node_feat: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply standardization to continuous node features."""
        if not self.is_fit:
            raise RuntimeError("FeaturePreprocessor must be fit before transforming features.")

        if isinstance(node_feat, torch.Tensor):
            device = node_feat.device
            mean_t = torch.from_numpy(self.node_mean).to(device)
            std_t = torch.from_numpy(self.node_std).to(device)
            return (node_feat - mean_t) / std_t
        else:
            return (node_feat - self.node_mean) / self.node_std

    def transform_config_feat(
        self, config_feat: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply standardization to continuous candidate configuration features."""
        if not self.is_fit:
            raise RuntimeError("FeaturePreprocessor must be fit before transforming features.")

        if isinstance(config_feat, torch.Tensor):
            device = config_feat.device
            mean_t = torch.from_numpy(self.config_mean).to(device)
            std_t = torch.from_numpy(self.config_std).to(device)
            return (config_feat - mean_t) / std_t
        else:
            return (config_feat - self.config_mean) / self.config_std

    def to_dict(self) -> Dict[str, Any]:
        """Serialize preprocessor state to dictionary."""
        return {
            "node_feat_dim": self.node_feat_dim,
            "config_feat_dim": self.config_feat_dim,
            "is_fit": self.is_fit,
            "node_mean": self.node_mean.tolist() if self.node_mean is not None else None,
            "node_std": self.node_std.tolist() if self.node_std is not None else None,
            "config_mean": self.config_mean.tolist() if self.config_mean is not None else None,
            "config_std": self.config_std.tolist() if self.config_std is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeaturePreprocessor:
        """Instantiate preprocessor from state dictionary."""
        inst = cls(
            node_feat_dim=data["node_feat_dim"],
            config_feat_dim=data["config_feat_dim"],
        )
        inst.is_fit = data["is_fit"]
        if inst.is_fit:
            inst.node_mean = np.array(data["node_mean"], dtype=np.float32)
            inst.node_std = np.array(data["node_std"], dtype=np.float32)
            inst.config_mean = np.array(data["config_mean"], dtype=np.float32)
            inst.config_std = np.array(data["config_std"], dtype=np.float32)
        return inst

    def save(self, filepath: Union[str, Path]) -> None:
        """Save preprocessor state to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> FeaturePreprocessor:
        """Load preprocessor state from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def extract_graph_summary_features(
    graph: GraphData,
    preprocessor: Optional[FeaturePreprocessor] = None,
    num_opcodes: int = 128,
) -> np.ndarray:
    """Extract a fixed-size graph summary representation for tabular / GBT models.

    Components:
      1. Structural counts: [log1p(nodes), log1p(edges), edges/nodes] (3 dims)
      2. Normalized opcode frequency histogram across 128 discrete opcodes (128 dims)
      3. Pooled node feature moments: mean (140) and std (140) across all nodes (280 dims)
    Total summary dimension = 3 + 128 + 280 = 411 dimensions.
    """
    # 1. Structural counts
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    counts = np.array(
        [
            np.log1p(num_nodes),
            np.log1p(num_edges),
            float(num_edges) / max(num_nodes, 1),
        ],
        dtype=np.float32,
    )

    # 2. Opcode histogram
    opcode_counts = np.bincount(
        np.clip(graph.node_opcode, 0, num_opcodes - 1),
        minlength=num_opcodes,
    ).astype(np.float32)
    opcode_hist = opcode_counts / max(num_nodes, 1)

    # 3. Pooled node features
    nf = graph.node_feat
    if preprocessor is not None and preprocessor.is_fit:
        nf = preprocessor.transform_node_feat(nf)

    nf_mean = np.mean(nf, axis=0)  # (140,)
    nf_std = np.std(nf, axis=0)    # (140,)

    return np.concatenate([counts, opcode_hist, nf_mean, nf_std]).astype(np.float32)
