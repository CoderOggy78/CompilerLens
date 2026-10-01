"""Tests for pairwise ranking loss, sign conventions, tie handling, and combined loss."""

import pytest
import torch
import torch.nn.functional as F

from compilerlens.losses import CombinedLoss, PairwiseRankingLoss


def test_pairwise_ranking_loss_sign_convention():
    """CRITICAL TEST: Verify that PairwiseRankingLoss severely penalizes reversed rankings.

    Ground Truth:
      Configuration 0 is faster than Configuration 1:
      y_0 = 1.0, y_1 = 2.0 (lower is faster).

    Scenario A (Correct Ranking):
      Model predicts y_hat_0 = 1.0, y_hat_1 = 2.0 (predicts 0 is faster).

    Scenario B (Reversed / Wrong Ranking):
      Model predicts y_hat_0 = 2.0, y_hat_1 = 1.0 (predicts 1 is faster).

    Assertion:
      Loss(Scenario B) MUST be strictly greater than Loss(Scenario A).
    """
    targets = torch.tensor([1.0, 2.0])  # config 0 is faster than config 1
    loss_fn = PairwiseRankingLoss(tie_threshold=1e-4)

    # Scenario A: Correct ranking (0 predicted faster than 1)
    preds_correct = torch.tensor([1.0, 2.0])
    loss_correct = loss_fn(preds_correct, targets)

    # Scenario B: Reversed ranking (0 predicted slower than 1)
    preds_reversed = torch.tensor([2.0, 1.0])
    loss_reversed = loss_fn(preds_reversed, targets)

    # softplus(-1.0) = log(1 + e^-1) = ~0.313
    # softplus(+1.0) = log(1 + e^1)  = ~1.313
    assert loss_reversed.item() > loss_correct.item()
    assert (loss_reversed.item() - loss_correct.item()) > 0.8, (
        f"Reversed loss {loss_reversed.item()} should strongly exceed correct loss {loss_correct.item()}"
    )


def test_pairwise_loss_catches_inverted_predictions():
    """Catch inverted predictions across multiple candidate configurations."""
    targets = torch.tensor([0.5, 1.2, 2.5, 4.0])  # strictly ascending: 0 < 1 < 2 < 3
    loss_fn = PairwiseRankingLoss(tie_threshold=1e-4, max_pairs_per_graph=100)

    # Perfectly ordered predictions
    preds_perfect = torch.tensor([0.1, 1.0, 2.0, 3.5])
    loss_perfect = loss_fn(preds_perfect, targets)

    # Completely inverted predictions
    preds_inverted = torch.tensor([3.5, 2.0, 1.0, 0.1])
    loss_inverted = loss_fn(preds_inverted, targets)

    assert loss_inverted.item() > loss_perfect.item() * 3.0


def test_pairwise_ranking_loss_tie_handling():
    """Ensure candidate pairs with runtime differences below tie_threshold are excluded."""
    # Difference is 1e-5 < tie_threshold (1e-4)
    targets = torch.tensor([1.0, 1.00001])
    loss_fn = PairwiseRankingLoss(tie_threshold=1e-4)

    preds = torch.tensor([2.0, 1.0])
    loss = loss_fn(preds, targets)
    # No valid pairs should be constructed
    assert loss.item() == 0.0


def test_combined_loss_modes():
    """Verify CombinedLoss behavior in regression_only, ranking_only, and combined modes."""
    targets = [torch.tensor([1.0, 2.0, 3.0])]
    preds = [torch.tensor([1.1, 2.2, 2.9])]

    combined_loss_fn = CombinedLoss(lambda_rank=1.0, mode="combined")
    tot_comb, reg_comb, rank_comb = combined_loss_fn(preds, targets)
    assert tot_comb.item() == pytest.approx((reg_comb + rank_comb).item(), rel=1e-5)

    reg_only_fn = CombinedLoss(mode="regression_only")
    tot_reg, _, _ = reg_only_fn(preds, targets)
    assert tot_reg.item() == pytest.approx(reg_comb.item(), rel=1e-5)

    rank_only_fn = CombinedLoss(mode="ranking_only")
    tot_rank, _, _ = rank_only_fn(preds, targets)
    assert tot_rank.item() == pytest.approx(rank_comb.item(), rel=1e-5)
