"""Tests for feature preprocessing and graph summary extraction."""

import numpy as np
import pytest
import torch

from compilerlens.data import GraphData
from compilerlens.features import FeaturePreprocessor, extract_graph_summary_features


@pytest.fixture
def sample_graph():
    rng = np.random.RandomState(42)
    return GraphData(
        graph_id="test_g",
        split="train",
        num_nodes=5,
        num_edges=4,
        num_configs=6,
        node_feat=rng.randn(5, 140).astype(np.float32) * 5.0 + 10.0,
        node_opcode=np.array([2, 5, 11, 24, 45], dtype=np.int32),
        edge_index=np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64),
        config_feat=rng.randn(6, 24).astype(np.float32) * 2.0 + 3.0,
        config_runtime=np.array([100, 200, 150, 120, 180, 220], dtype=np.int64),
        config_runtime_normalizers=np.full(6, 100, dtype=np.int64),
        normalized_runtimes=np.array([1.0, 2.0, 1.5, 1.2, 1.8, 2.2]),
        log_normalized_runtimes=np.log(np.array([1.0, 2.0, 1.5, 1.2, 1.8, 2.2], dtype=np.float32)),
    )


def test_feature_preprocessor_fit_and_transform(sample_graph):
    """Test fitting preprocessor strictly on training graphs and transforming features."""
    preprocessor = FeaturePreprocessor(node_feat_dim=140, config_feat_dim=24)
    assert not preprocessor.is_fit

    with pytest.raises(RuntimeError, match="must be fit before transforming"):
        preprocessor.transform_node_feat(sample_graph.node_feat)

    preprocessor.fit([sample_graph])
    assert preprocessor.is_fit
    assert preprocessor.node_mean.shape == (140,)
    assert preprocessor.node_std.shape == (140,)

    # Transform numpy
    std_node = preprocessor.transform_node_feat(sample_graph.node_feat)
    assert np.allclose(np.mean(std_node, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(std_node, axis=0), 1.0, atol=1e-5)

    # Transform torch tensor
    t_feat = torch.from_numpy(sample_graph.node_feat)
    std_t = preprocessor.transform_node_feat(t_feat)
    assert torch.is_tensor(std_t)
    assert torch.allclose(torch.mean(std_t, dim=0), torch.zeros(140), atol=1e-5)


def test_extract_graph_summary_features(sample_graph):
    """Verify graph summary feature extraction dimensions and finiteness."""
    preprocessor = FeaturePreprocessor(node_feat_dim=140, config_feat_dim=24)
    preprocessor.fit([sample_graph])

    summary = extract_graph_summary_features(sample_graph, preprocessor=preprocessor, num_opcodes=128)
    # Expected: 3 structural + 128 opcode hist + 140 mean + 140 std = 411 dimensions
    assert summary.shape == (411,)
    assert np.all(np.isfinite(summary))
    # Structural count checks
    assert np.isclose(summary[0], np.log1p(sample_graph.num_nodes))
    assert np.isclose(summary[1], np.log1p(sample_graph.num_edges))
