"""Loss functions for compiler configuration ranking and runtime regression."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    """Pairwise logistic ranking loss over configurations within a single computation graph.

    Sign Convention:
      Let y_i, y_j be true log normalized runtimes, where lower is better (faster).
      For a candidate pair (i, j) where y_i < y_j (configuration i is faster than configuration j),
      the model should predict y_hat_i < y_hat_j.
      The loss is:
          loss(i, j) = softplus(y_hat_i - y_hat_j) = log(1 + exp(y_hat_i - y_hat_j))
      If y_hat_i < y_hat_j, y_hat_i - y_hat_j is negative, loss -> 0.
      If y_hat_i > y_hat_j (violating ordering), y_hat_i - y_hat_j is positive, loss penalizes linearly.
    """

    def __init__(
        self,
        tie_threshold: float = 1e-4,
        max_pairs_per_graph: int = 256,
    ) -> None:
        super().__init__()
        self.tie_threshold = tie_threshold
        self.max_pairs_per_graph = max_pairs_per_graph

    def sample_pairs(
        self, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct candidate pairs (i, j) where target[i] < target[j] - tie_threshold.

        Returns:
            Tuple of (idx_faster, idx_slower) 1D LongTensors.
        """
        num_candidates = targets.size(0)
        if num_candidates < 2:
            empty = torch.empty(0, dtype=torch.long, device=targets.device)
            return empty, empty

        # Form pairwise difference matrix where entry (i, j) is targets[j] - targets[i]
        # targets.unsqueeze(0) is shape (1, C) -> targets[j]
        # targets.unsqueeze(1) is shape (C, 1) -> targets[i]
        diff_matrix = targets.unsqueeze(0) - targets.unsqueeze(1)
        valid_mask = diff_matrix > self.tie_threshold

        idx_i, idx_j = torch.where(valid_mask)
        num_valid = idx_i.size(0)

        if num_valid == 0:
            empty = torch.empty(0, dtype=torch.long, device=targets.device)
            return empty, empty

        if num_valid > self.max_pairs_per_graph:
            perm = torch.randperm(num_valid, device=targets.device)[: self.max_pairs_per_graph]
            idx_i = idx_i[perm]
            idx_j = idx_j[perm]

        return idx_i, idx_j

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute average pairwise ranking loss for a single graph's candidate predictions."""
        idx_faster, idx_slower = self.sample_pairs(targets)
        if idx_faster.numel() == 0:
            return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

        # preds[idx_faster] should be < preds[idx_slower]
        # penalty when preds[idx_faster] - preds[idx_slower] > 0
        diff = preds[idx_faster] - preds[idx_slower]
        loss = F.softplus(diff)
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    """Combined regression and pairwise ranking objective for learned compiler cost models.

    Loss = L_regression + lambda_rank * L_ranking

    Averages losses per graph first, then averages across graphs in a batch, ensuring programs
    with many candidate configurations do not dominate gradient steps.
    """

    def __init__(
        self,
        lambda_rank: float = 1.0,
        huber_delta: float = 1.0,
        tie_threshold: float = 1e-4,
        max_pairs_per_graph: int = 256,
        mode: str = "combined",  # "combined", "regression_only", or "ranking_only"
    ) -> None:
        super().__init__()
        self.lambda_rank = lambda_rank
        self.huber_delta = huber_delta
        self.mode = mode
        self.ranking_loss_fn = PairwiseRankingLoss(
            tie_threshold=tie_threshold,
            max_pairs_per_graph=max_pairs_per_graph,
        )

    def forward(
        self,
        graph_preds: List[torch.Tensor],
        graph_targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined loss over a list of graph-level prediction and target tensors.

        Args:
            graph_preds: List of 1D tensors, each representing predicted log runtimes for graph g.
            graph_targets: List of 1D tensors, each representing true log runtimes for graph g.

        Returns:
            Tuple of (total_loss, regression_loss_component, ranking_loss_component).
        """
        assert len(graph_preds) == len(graph_targets), "Mismatched predictions and targets lists."
        num_graphs = len(graph_preds)
        if num_graphs == 0:
            zero = torch.tensor(0.0, requires_grad=True)
            return zero, zero, zero

        reg_losses = []
        rank_losses = []

        for p, t in zip(graph_preds, graph_targets):
            # Huber regression loss
            reg_loss_g = F.huber_loss(p, t, delta=self.huber_delta, reduction="mean")
            reg_losses.append(reg_loss_g)

            # Pairwise ranking loss
            rank_loss_g = self.ranking_loss_fn(p, t)
            rank_losses.append(rank_loss_g)

        mean_reg_loss = torch.stack(reg_losses).mean()
        mean_rank_loss = torch.stack(rank_losses).mean()

        if self.mode == "regression_only":
            total_loss = mean_reg_loss
        elif self.mode == "ranking_only":
            total_loss = mean_rank_loss
        else:
            total_loss = mean_reg_loss + (self.lambda_rank * mean_rank_loss)

        return total_loss, mean_reg_loss, mean_rank_loss
