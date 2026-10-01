"""Tests for checkpoint save/load equivalence, training state restoration, and overfitting sanity check."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from compilerlens.data import GraphData, TpuTileDataset
from compilerlens.features import FeaturePreprocessor
from compilerlens.models.gnn_conditioned import ConfigConditionedGNN
from compilerlens.train import Trainer


@pytest.fixture
def tiny_dataset(tmp_path):
    """Generates two tiny real-format NPZ files for pipeline and checkpointing verification."""
    rng = np.random.RandomState(42)

    files = []
    for g_i in range(2):
        fpath = tmp_path / f"tiny_g{g_i}.npz"
        np.savez(
            fpath,
            node_feat=rng.randn(6, 140).astype(np.float32),
            node_opcode=rng.randint(0, 50, size=6).astype(np.int32),
            edge_index=np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=np.int64),
            config_feat=rng.randn(8, 24).astype(np.float32),
            config_runtime=np.array([120, 150, 180, 210, 240, 270, 300, 330], dtype=np.int64),
            config_runtime_normalizers=np.full(8, 200, dtype=np.int64),
        )
        files.append(fpath)

    return files


def test_checkpoint_save_and_load_equivalence(tiny_dataset, tmp_path):
    """Verify that saving and loading a checkpoint reproduces identical model outputs."""
    torch.manual_seed(42)

    train_ds = TpuTileDataset(tiny_dataset, split_name="train", max_configs=8)
    val_ds = TpuTileDataset(tiny_dataset, split_name="valid", max_configs=8)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit([train_ds[0], train_ds[1]])

    cfg = {
        "device": "cpu",
        "training": {
            "batch_size": 2,
            "num_epochs": 1,
            "learning_rate": 0.001,
            "checkpoint_dir": str(tmp_path / "ckpts"),
        },
    }

    model = ConfigConditionedGNN(
        node_feat_dim=140,
        config_feat_dim=24,
        hidden_dim=32,
        opcode_embed_dim=16,
        num_layers=2,
        dropout=0.0,
    )

    trainer = Trainer(model, train_ds, val_ds, preprocessor, cfg, device=torch.device("cpu"))
    # Save checkpoint
    ckpt_path = trainer.save_checkpoint(epoch=1, val_metric=0.25, is_best=True)

    # Initial evaluation predictions
    preds_before = trainer.evaluate(val_ds)

    # Create a fresh model and trainer and load checkpoint
    model_restored = ConfigConditionedGNN(
        node_feat_dim=140,
        config_feat_dim=24,
        hidden_dim=32,
        opcode_embed_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    trainer_restored = Trainer(
        model_restored, train_ds, val_ds, preprocessor, cfg, device=torch.device("cpu")
    )
    trainer_restored.load_checkpoint(ckpt_path)

    # Restored evaluation predictions
    preds_after = trainer_restored.evaluate(val_ds)

    assert pytest.approx(preds_before["top1_regret_mean"], abs=1e-5) == preds_after["top1_regret_mean"]


def test_model_overfit_tiny_subset_sanity_check(tiny_dataset, tmp_path):
    """Verify that the model can successfully overfit a tiny dataset as an implementation sanity check.

    This ensures gradients, message passing, conditioning, and loss backward flows operate
    without bottlenecks or vanishing gradients.
    """
    torch.manual_seed(42)
    train_ds = TpuTileDataset(tiny_dataset, split_name="train", max_configs=4)
    val_ds = TpuTileDataset(tiny_dataset, split_name="valid", max_configs=4)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit([train_ds[0], train_ds[1]])

    cfg = {
        "device": "cpu",
        "training": {
            "batch_size": 2,
            "num_epochs": 20,
            "learning_rate": 0.01,
            "checkpoint_dir": str(tmp_path / "overfit_ckpts"),
            "loss_mode": "regression_only",
        },
    }

    model = ConfigConditionedGNN(
        node_feat_dim=140,
        config_feat_dim=24,
        hidden_dim=32,
        opcode_embed_dim=16,
        num_layers=2,
        dropout=0.0,
    )

    trainer = Trainer(model, train_ds, val_ds, preprocessor, cfg, device=torch.device("cpu"))
    history = trainer.fit()

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]

    assert final_loss < initial_loss * 0.5, (
        f"Model failed to overfit tiny subset: initial {initial_loss:.4f} -> final {final_loss:.4f}"
    )
