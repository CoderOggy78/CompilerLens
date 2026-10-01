"""Tests for data loading, schema validation, split isolation, and edge conversion."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from compilerlens.data import (
    GraphData,
    TpuTileDataset,
    collate_tile_graphs,
    create_graph_disjoint_splits,
    load_npz_graph,
)


@pytest.fixture
def synthetic_graph_fixture():
    """Generates a valid synthetic computation graph fixture for lightweight testing."""
    num_nodes = 10
    num_edges = 12
    num_configs = 8

    rng = np.random.RandomState(42)
    node_feat = rng.randn(num_nodes, 140).astype(np.float32)
    node_opcode = rng.randint(0, 100, size=num_nodes).astype(np.int32)

    # Valid DAG edges
    src = rng.randint(0, num_nodes - 1, size=num_edges)
    dst = rng.randint(src + 1, num_nodes, size=num_edges)
    edge_index = np.stack([src, dst], axis=1).astype(np.int64)

    config_feat = rng.randn(num_configs, 24).astype(np.float32)
    config_runtime = rng.randint(100000, 500000, size=num_configs).astype(np.int64)
    normalizers = np.full(num_configs, 200000, dtype=np.int64)

    return {
        "node_feat": node_feat,
        "node_opcode": node_opcode,
        "edge_index": edge_index,
        "config_feat": config_feat,
        "config_runtime": config_runtime,
        "config_runtime_normalizers": normalizers,
    }


def test_load_npz_graph_valid(synthetic_graph_fixture, tmp_path):
    """Verify loading and schema parsing of a valid NPZ file."""
    fpath = tmp_path / "sample_graph.npz"
    np.savez(fpath, **synthetic_graph_fixture)

    graph = load_npz_graph(fpath, split="train", validate=True)
    assert graph.graph_id == "sample_graph"
    assert graph.num_nodes == 10
    assert graph.num_edges == 12
    assert graph.num_configs == 8
    assert graph.node_feat.shape == (10, 140)
    assert graph.config_feat.shape == (8, 24)
    assert graph.normalized_runtimes is not None
    assert len(graph.normalized_runtimes) == 8
    assert graph.log_normalized_runtimes is not None
    assert len(graph.log_normalized_runtimes) == 8


def test_schema_validation_missing_key(synthetic_graph_fixture, tmp_path):
    """Ensure missing required keys raise ValueError."""
    del synthetic_graph_fixture["node_feat"]
    fpath = tmp_path / "bad_graph.npz"
    np.savez(fpath, **synthetic_graph_fixture)

    with pytest.raises(ValueError, match="missing required fields"):
        load_npz_graph(fpath, validate=True)


def test_schema_validation_invalid_edge_indices(synthetic_graph_fixture, tmp_path):
    """Ensure out-of-bounds edge indices trigger validation error."""
    synthetic_graph_fixture["edge_index"][0] = [0, 999]  # 999 >= num_nodes (10)
    fpath = tmp_path / "bad_edges.npz"
    np.savez(fpath, **synthetic_graph_fixture)

    with pytest.raises(ValueError, match="Invalid edge indices"):
        load_npz_graph(fpath, validate=True)


def test_schema_validation_nonpositive_runtimes(synthetic_graph_fixture, tmp_path):
    """Ensure non-positive runtimes are rejected for labeled graphs."""
    synthetic_graph_fixture["config_runtime"][0] = -50
    fpath = tmp_path / "neg_runtime.npz"
    np.savez(fpath, **synthetic_graph_fixture)

    with pytest.raises(ValueError, match="Non-positive runtimes"):
        load_npz_graph(fpath, validate=True)


def test_graph_level_split_isolation(tmp_path):
    """Verify that graph-level splits are strictly disjoint with zero graph leakage."""
    file_list = [tmp_path / f"model_program_{i:04d}.npz" for i in range(50)]
    for p in file_list:
        p.touch()

    train_files, val_files = create_graph_disjoint_splits(file_list, val_ratio=0.2, seed=42)
    assert len(train_files) == 40
    assert len(val_files) == 10

    train_set = {p.stem for p in train_files}
    val_set = {p.stem for p in val_files}
    intersection = train_set.intersection(val_set)
    assert len(intersection) == 0, f"Graph leakage detected across splits: {intersection}"


def test_collate_tile_graphs_edge_directions(synthetic_graph_fixture, tmp_path):
    """Verify collation of multiple graphs preserves forward and reverse edge relations."""
    fpath1 = tmp_path / "g1.npz"
    fpath2 = tmp_path / "g2.npz"
    np.savez(fpath1, **synthetic_graph_fixture)
    np.savez(fpath2, **synthetic_graph_fixture)

    g1 = load_npz_graph(fpath1)
    g2 = load_npz_graph(fpath2)

    batch = collate_tile_graphs([g1, g2])
    assert batch["num_graphs"] == 2
    assert batch["node_feat"].size(0) == g1.num_nodes + g2.num_nodes
    assert batch["fwd_edge_index"].size(1) == g1.num_edges + g2.num_edges
    assert batch["rev_edge_index"].size(1) == g1.num_edges + g2.num_edges

    # Check reverse edge swap: rev[0] == fwd[1] and rev[1] == fwd[0]
    fwd = batch["fwd_edge_index"]
    rev = batch["rev_edge_index"]
    assert torch.equal(fwd[0], rev[1])
    assert torch.equal(fwd[1], rev[0])
