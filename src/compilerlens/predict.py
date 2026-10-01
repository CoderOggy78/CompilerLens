"""Inference utilities, configuration recommendation, and test prediction exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from compilerlens.data import GraphData
from compilerlens.features import FeaturePreprocessor


def predict_graph_candidates(
    model: nn.Module,
    graph: GraphData,
    preprocessor: FeaturePreprocessor,
    device: torch.device = torch.device("cpu"),
    chunk_size: Optional[int] = 64,
) -> np.ndarray:
    """Predict log normalized runtimes for all candidate configurations of a single graph.

    Args:
        model: Trained PyTorch ranking model.
        graph: Target GraphData instance.
        preprocessor: Fitted FeaturePreprocessor for standardizing inputs.
        device: Target execution device.
        chunk_size: Optional chunk size for candidate chunking.

    Returns:
        np.ndarray of shape (C,) representing predicted log runtimes (lower is predicted faster).
    """
    model.eval()
    with torch.no_grad():
        node_feat_std = preprocessor.transform_node_feat(
            torch.from_numpy(graph.node_feat).float()
        ).to(device)
        node_opcode = torch.from_numpy(graph.node_opcode).long().to(device)
        configs_std = preprocessor.transform_config_feat(
            torch.from_numpy(graph.config_feat).float()
        ).to(device)

        if graph.num_edges > 0:
            edges = torch.from_numpy(graph.edge_index).long().to(device)
            fwd_edges = edges.t().contiguous()
            rev_edges = edges[:, [1, 0]].t().contiguous()
        else:
            fwd_edges = torch.empty((2, 0), dtype=torch.long, device=device)
            rev_edges = torch.empty((2, 0), dtype=torch.long, device=device)

        if hasattr(model, "forward_graph_candidates"):
            h_base = model.extract_base_node_repr(node_feat_std, node_opcode)
            scores = model.forward_graph_candidates(
                h_base, fwd_edges, rev_edges, configs_std, chunk_size=chunk_size
            )
            return scores.cpu().numpy()
        else:
            # Fallback for models with forward_batch
            batch = {
                "num_graphs": 1,
                "graph_ids": [graph.graph_id],
                "node_feat": node_feat_std,
                "node_opcode": node_opcode,
                "graph_node_idx": torch.zeros(graph.num_nodes, dtype=torch.long, device=device),
                "fwd_edge_index": fwd_edges,
                "rev_edge_index": rev_edges,
                "graph_configs": [configs_std],
            }
            scores = model.forward_batch(batch)[0]
            return scores.cpu().numpy()


def rank_compiler_configurations(
    model: nn.Module,
    graph: GraphData,
    preprocessor: FeaturePreprocessor,
    device: torch.device = torch.device("cpu"),
    top_k: int = 5,
    chunk_size: Optional[int] = 64,
) -> Dict[str, Any]:
    """Rank candidate compiler configurations from fastest to slowest and return top-k recommendations.

    Args:
        model: Trained ranking model.
        graph: Target computation graph.
        preprocessor: Fitted preprocessor.
        device: Device to run inference on.
        top_k: Number of shortlisted configurations to recommend.
        chunk_size: Chunk size for candidate processing.

    Returns:
        Dictionary with ranked indices, predicted scores, and top-k recommendations.
    """
    scores = predict_graph_candidates(
        model, graph, preprocessor, device=device, chunk_size=chunk_size
    )
    ranked_indices = np.argsort(scores).tolist()
    top_k_indices = ranked_indices[: min(top_k, len(ranked_indices))]

    recommendations = []
    for rank, idx in enumerate(top_k_indices, start=1):
        rec = {
            "rank": rank,
            "candidate_index": idx,
            "predicted_log_runtime": float(scores[idx]),
            "configuration_features": graph.config_feat[idx].tolist(),
        }
        if graph.normalized_runtimes is not None:
            rec["observed_normalized_runtime"] = float(graph.normalized_runtimes[idx])
        recommendations.append(rec)

    return {
        "graph_id": graph.graph_id,
        "num_candidates": graph.num_configs,
        "ranked_indices": ranked_indices,
        "top_k_recommendations": recommendations,
    }


def export_test_predictions(
    model: nn.Module,
    test_graphs: Sequence[GraphData],
    preprocessor: FeaturePreprocessor,
    output_path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
    chunk_size: Optional[int] = 64,
) -> Path:
    """Generate and export configuration ranking predictions for the official test set.

    Format: CSV file with columns 'ID' (graph_id) and 'TopConfigs' (semicolon-separated candidate indices).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for g in test_graphs:
        scores = predict_graph_candidates(
            model, g, preprocessor, device=device, chunk_size=chunk_size
        )
        ranked_indices = np.argsort(scores).tolist()
        indices_str = ";".join(str(idx) for idx in ranked_indices)
        rows.append({"ID": g.graph_id, "TopConfigs": indices_str})

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "TopConfigs"])
        writer.writeheader()
        writer.writerows(rows)

    return output_path
