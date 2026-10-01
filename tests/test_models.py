"""Tests for neural model architectures, forward/backward passes, and candidate chunking."""

import pytest
import torch

from compilerlens.models.baseline_graphfree import GraphFreeModel
from compilerlens.models.baseline_graphsage import GraphSAGEModel
from compilerlens.models.gnn_conditioned import ConfigConditionedGNN


@pytest.fixture
def dummy_batch():
    """Generates a small batched graph dictionary for model testing."""
    num_nodes = 8
    num_edges = 10
    num_configs_g0 = 5
    num_configs_g1 = 4

    rng = torch.Generator().manual_seed(42)

    node_feat = torch.randn(num_nodes, 140, generator=rng)
    node_opcode = torch.randint(0, 100, (num_nodes,), generator=rng)
    graph_node_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    # Forward edges
    src = torch.tensor([0, 1, 1, 2, 4, 4, 5, 5, 6, 7])
    dst = torch.tensor([1, 2, 3, 3, 5, 6, 6, 7, 7, 7])
    fwd_edges = torch.stack([src, dst], dim=0)
    rev_edges = torch.stack([dst, src], dim=0)

    configs_g0 = torch.randn(num_configs_g0, 24, generator=rng)
    configs_g1 = torch.randn(num_configs_g1, 24, generator=rng)

    return {
        "num_graphs": 2,
        "graph_ids": ["g0", "g1"],
        "node_feat": node_feat,
        "node_opcode": node_opcode,
        "graph_node_idx": graph_node_idx,
        "fwd_edge_index": fwd_edges,
        "rev_edge_index": rev_edges,
        "graph_configs": [configs_g0, configs_g1],
        "graph_targets": [torch.randn(num_configs_g0), torch.randn(num_configs_g1)],
    }


def test_main_gnn_forward_backward_pass(dummy_batch):
    """Verify that ConfigConditionedGNN performs forward and backward passes cleanly with valid gradients."""
    model = ConfigConditionedGNN(
        node_feat_dim=140,
        config_feat_dim=24,
        num_opcodes=128,
        opcode_embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )

    preds = model.forward_batch(dummy_batch)
    assert len(preds) == 2
    assert preds[0].shape == (5,)
    assert preds[1].shape == (4,)

    # Compute a scalar loss and run backward pass
    loss = preds[0].mean() + preds[1].mean()
    loss.backward()

    # Verify gradients exist on all key components
    assert model.opcode_embed.weight.grad is not None
    assert model.config_encoder.net[0].weight.grad is not None
    assert model.gnn_layers[0].fwd_proj.weight.grad is not None
    assert model.head.net[-1].weight.grad is not None


def test_candidate_chunking_equivalence():
    """Verify that candidate chunking produces bitwise/numerical equivalent inference to unchunked inference."""
    torch.manual_seed(42)
    model = ConfigConditionedGNN(
        node_feat_dim=140,
        config_feat_dim=24,
        num_opcodes=128,
        opcode_embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
    )
    model.eval()

    num_nodes = 6
    num_configs = 15  # 15 candidates
    node_feat = torch.randn(num_nodes, 140)
    node_opcode = torch.randint(0, 100, (num_nodes,))
    configs = torch.randn(num_configs, 24)

    fwd_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    rev_edges = torch.tensor([[1, 2, 3, 4], [0, 1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        h_base = model.extract_base_node_repr(node_feat, node_opcode)

        # Unchunked forward pass
        scores_unchunked = model.forward_graph_candidates(
            h_base, fwd_edges, rev_edges, configs, chunk_size=None
        )

        # Chunked forward pass with small chunk_size=4
        scores_chunked = model.forward_graph_candidates(
            h_base, fwd_edges, rev_edges, configs, chunk_size=4
        )

    assert torch.allclose(scores_unchunked, scores_chunked, atol=1e-5), (
        f"Chunked scores differ from unchunked: max diff = {(scores_unchunked - scores_chunked).abs().max()}"
    )


def test_baselines_forward_pass(dummy_batch):
    """Verify that GraphFreeModel and GraphSAGEModel execute forward passes properly."""
    gf = GraphFreeModel(hidden_dim=32, opcode_embed_dim=16, dropout=0.0)
    gf_preds = gf.forward_batch(dummy_batch)
    assert len(gf_preds) == 2
    assert gf_preds[0].shape == (5,)

    sage = GraphSAGEModel(hidden_dim=32, opcode_embed_dim=16, num_layers=2, dropout=0.0)
    sage_preds = sage.forward_batch(dummy_batch)
    assert len(sage_preds) == 2
    assert sage_preds[0].shape == (5,)
