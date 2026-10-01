"""Model training, optimization, pilot throughput profiling, and checkpoint management."""

from __future__ import annotations

import logging
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compilerlens.data import TpuTileDataset, collate_tile_graphs
from compilerlens.evaluate import evaluate_predictions
from compilerlens.features import FeaturePreprocessor
from compilerlens.losses import CombinedLoss

logger = logging.getLogger(__name__)


def get_device(requested: str = "auto") -> torch.device:
    """Select the optimal compute device (CUDA, Apple Silicon MPS, or CPU)."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Seed all pseudo-random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """End-to-end trainer for CompilerLens neural ranking models."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset: TpuTileDataset,
        val_dataset: TpuTileDataset,
        preprocessor: FeaturePreprocessor,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or get_device(config.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.preprocessor = preprocessor

        tr_cfg = config.get("training", {})
        self.batch_size = tr_cfg.get("batch_size", 4)
        self.num_epochs = tr_cfg.get("num_epochs", 10)
        self.learning_rate = tr_cfg.get("learning_rate", 1e-3)
        self.weight_decay = tr_cfg.get("weight_decay", 1e-4)
        self.grad_clip_norm = tr_cfg.get("grad_clip_norm", 1.0)
        self.checkpoint_dir = Path(tr_cfg.get("checkpoint_dir", "checkpoints/run"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = CombinedLoss(
            lambda_rank=tr_cfg.get("lambda_rank", 1.0),
            huber_delta=tr_cfg.get("huber_delta", 1.0),
            tie_threshold=tr_cfg.get("tie_threshold", 1e-4),
            max_pairs_per_graph=tr_cfg.get("max_pairs_per_graph", 256),
            mode=tr_cfg.get("loss_mode", "combined"),
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.num_epochs, 1),
            eta_min=1e-5,
        )

        self.best_val_metric = float("inf")
        self.start_epoch = 0
        self.history: List[Dict[str, Any]] = []

    def save_checkpoint(
        self,
        epoch: int,
        val_metric: float,
        is_best: bool = False,
    ) -> Path:
        """Persist full training state including model, optimizer, scheduler, and RNGs."""
        ckpt = {
            "epoch": epoch,
            "val_metric": val_metric,
            "best_val_metric": self.best_val_metric,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "preprocessor_state": self.preprocessor.to_dict(),
            "config": self.config,
            "history": self.history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "env_versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
        }

        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            logger.info(f"Saved new best model checkpoint to {best_path} (Metric: {val_metric:.4f})")

        return ckpt_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Resume training from a saved checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_metric = ckpt.get("best_val_metric", float("inf"))
        self.history = ckpt.get("history", [])

        # Restore RNG states
        if "rng_state" in ckpt:
            random.setstate(ckpt["rng_state"]["python"])
            np.random.set_state(ckpt["rng_state"]["numpy"])
            torch.set_rng_state(ckpt["rng_state"]["torch"])

        logger.info(f"Resumed from epoch {self.start_epoch}, previous best metric: {self.best_val_metric:.4f}")

    def profile_pilot_epoch(self) -> Dict[str, float]:
        """Measure pilot throughput on 1 epoch to estimate runtime reliably."""
        logger.info("Profiling pilot epoch throughput...")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_tile_graphs,
        )

        start_time = time.time()
        self.model.train()
        graphs_processed = 0

        for batch in train_loader:
            # Transfer and standardize
            batch["node_feat"] = self.preprocessor.transform_node_feat(
                batch["node_feat"]
            ).to(self.device)
            batch["node_opcode"] = batch["node_opcode"].to(self.device)
            batch["graph_node_idx"] = batch["graph_node_idx"].to(self.device)
            batch["fwd_edge_index"] = batch["fwd_edge_index"].to(self.device)
            batch["rev_edge_index"] = batch["rev_edge_index"].to(self.device)
            batch["graph_configs"] = [
                self.preprocessor.transform_config_feat(c).to(self.device)
                for c in batch["graph_configs"]
            ]
            batch["graph_targets"] = [t.to(self.device) for t in batch["graph_targets"]]

            self.optimizer.zero_grad()
            preds = self.model.forward_batch(batch)
            loss, _, _ = self.loss_fn(preds, batch["graph_targets"])
            loss.backward()
            self.optimizer.step()

            graphs_processed += batch["num_graphs"]
            if graphs_processed >= min(12, len(self.train_dataset)):
                break

        elapsed = time.time() - start_time
        gps = graphs_processed / max(elapsed, 1e-4)
        logger.info(f"Pilot throughput: {gps:.2f} graphs/sec ({elapsed:.2f}s for {graphs_processed} graphs)")
        return {
            "graphs_per_sec": gps,
            "pilot_elapsed_sec": elapsed,
            "estimated_epoch_sec": len(self.train_dataset) / max(gps, 1e-4),
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_rank_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch["node_feat"] = self.preprocessor.transform_node_feat(
                batch["node_feat"]
            ).to(self.device)
            batch["node_opcode"] = batch["node_opcode"].to(self.device)
            batch["graph_node_idx"] = batch["graph_node_idx"].to(self.device)
            batch["fwd_edge_index"] = batch["fwd_edge_index"].to(self.device)
            batch["rev_edge_index"] = batch["rev_edge_index"].to(self.device)
            batch["graph_configs"] = [
                self.preprocessor.transform_config_feat(c).to(self.device)
                for c in batch["graph_configs"]
            ]
            batch["graph_targets"] = [t.to(self.device) for t in batch["graph_targets"]]

            self.optimizer.zero_grad()
            preds = self.model.forward_batch(batch)
            loss, reg_loss, rank_loss = self.loss_fn(preds, batch["graph_targets"])

            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_rank_loss += rank_loss.item()
            num_batches += 1

        return {
            "train_loss": total_loss / max(num_batches, 1),
            "train_reg_loss": total_reg_loss / max(num_batches, 1),
            "train_rank_loss": total_rank_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataset: TpuTileDataset) -> Dict[str, Any]:
        """Run candidate ranking evaluation on validation or test dataset."""
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_tile_graphs,
        )

        all_preds: Dict[str, np.ndarray] = {}
        ground_truth_graphs = [dataset[i] for i in range(len(dataset))]

        for batch in loader:
            batch["node_feat"] = self.preprocessor.transform_node_feat(
                batch["node_feat"]
            ).to(self.device)
            batch["node_opcode"] = batch["node_opcode"].to(self.device)
            batch["graph_node_idx"] = batch["graph_node_idx"].to(self.device)
            batch["fwd_edge_index"] = batch["fwd_edge_index"].to(self.device)
            batch["rev_edge_index"] = batch["rev_edge_index"].to(self.device)
            batch["graph_configs"] = [
                self.preprocessor.transform_config_feat(c).to(self.device)
                for c in batch["graph_configs"]
            ]

            batch_scores = self.model.forward_batch(batch)
            for g_id, scores in zip(batch["graph_ids"], batch_scores):
                all_preds[g_id] = scores.cpu().numpy()

        eval_summary = evaluate_predictions(
            all_preds,
            ground_truth_graphs,
            top_k_list=self.config.get("evaluation", {}).get("top_k_list", [1, 5, 10]),
            bootstrap_rounds=self.config.get("evaluation", {}).get("bootstrap_rounds", 200),
        )
        return eval_summary

    def fit(self, resume_from: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Run full training lifecycle with checkpointing and early stopping."""
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_tile_graphs,
        )

        patience = self.config.get("training", {}).get("early_stopping_patience", 5)
        epochs_without_improvement = 0

        logger.info(f"Starting training: {self.start_epoch} -> {self.num_epochs} epochs on {self.device}")

        for epoch in range(self.start_epoch, self.num_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            # Validation
            val_results = self.evaluate(self.val_dataset)
            epoch_time = time.time() - t0

            val_metric = val_results["top1_regret_mean"]
            is_best = val_metric < self.best_val_metric
            if is_best:
                self.best_val_metric = val_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            record = {
                "epoch": epoch,
                "epoch_time_sec": epoch_time,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_metrics,
                "val_top1_regret_mean": val_results["top1_regret_mean"],
                "val_top1_regret_median": val_results["top1_regret_median"],
                "val_spearman_mean": val_results.get("spearman_mean"),
                "val_log_mae": val_results.get("log_mae_mean"),
            }
            self.history.append(record)

            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_metrics['train_loss']:.4f} "
                f"| Val Top-1 Regret: {val_results['top1_regret_mean']:.4f} "
                f"| Val Spearman: {val_results.get('spearman_mean') or 0.0:.4f} "
                f"| Time: {epoch_time:.1f}s"
            )

            self.save_checkpoint(epoch, val_metric, is_best=is_best)

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        return self.history
