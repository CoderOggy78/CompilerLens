"""Tests for evaluation metrics: Top-1 regret, Best-of-top-k regret, Spearman correlation, and CIs."""

import numpy as np
import pytest

from compilerlens.evaluate import (
    compute_bootstrap_ci,
    evaluate_predictions,
    evaluate_single_graph,
)


def test_hand_verifiable_regret_calculation():
    """Verify Top-1 and Top-k regret calculations against hand-calculated analytical ground truth.

    Setup:
      3 Candidate configurations:
        Candidate 0: runtime = 10.0  (Oracle Best c*, runtime = 10.0)
        Candidate 1: runtime = 15.0  (50% slower than c*)
        Candidate 2: runtime = 30.0  (200% slower than c*)

    Case 1: Model predicts scores [2.0, 1.0, 3.0]
      Lowest score is Candidate 1 (score 1.0).
      Predicted best: Candidate 1 with runtime 15.0.
      Expected Top-1 regret: 15.0 / 10.0 - 1.0 = 0.50 (50% regret).

      For k=2: Shortlist candidates are {Candidate 1, Candidate 0}.
      Best in shortlist has runtime min(15.0, 10.0) = 10.0.
      Expected Best-of-top-2 regret: 10.0 / 10.0 - 1.0 = 0.00 (0% regret).
    """
    observed_runtimes = np.array([10.0, 15.0, 30.0])
    pred_scores = np.array([2.0, 1.0, 3.0])

    res = evaluate_single_graph(
        pred_scores=pred_scores,
        observed_normalized_runtimes=observed_runtimes,
        top_k_list=[1, 2],
        graph_id="hand_check_1",
    )

    assert res.oracle_best_runtime == 10.0
    assert res.predicted_best_runtime == 15.0
    assert pytest.approx(res.top1_regret, abs=1e-6) == 0.50
    assert pytest.approx(res.best_of_topk_regret[1], abs=1e-6) == 0.50
    assert pytest.approx(res.best_of_topk_regret[2], abs=1e-6) == 0.00


def test_perfect_prediction_yields_zero_regret():
    """Verify that predicting the true ordering results in 0.0 Top-1 and Top-k regret."""
    observed_runtimes = np.array([1.2, 2.5, 4.0, 5.0])
    # Exact monotonic scores
    pred_scores = np.array([0.1, 0.5, 0.8, 1.2])

    res = evaluate_single_graph(
        pred_scores=pred_scores,
        observed_normalized_runtimes=observed_runtimes,
        top_k_list=[1, 2, 3],
        graph_id="perfect_check",
    )

    assert pytest.approx(res.top1_regret, abs=1e-6) == 0.0
    assert pytest.approx(res.best_of_topk_regret[1], abs=1e-6) == 0.0
    assert pytest.approx(res.best_of_topk_regret[2], abs=1e-6) == 0.0
    assert res.spearman_rho is not None
    assert pytest.approx(res.spearman_rho, abs=1e-6) == 1.0


def test_spearman_undefined_on_constant_predictions():
    """Ensure constant predictions gracefully return None for Spearman rho rather than crashing."""
    observed_runtimes = np.array([1.0, 2.0, 3.0])
    pred_scores = np.array([1.5, 1.5, 1.5])  # Zero variance

    res = evaluate_single_graph(
        pred_scores=pred_scores,
        observed_normalized_runtimes=observed_runtimes,
        top_k_list=[1],
        graph_id="const_check",
    )

    assert res.spearman_rho is None


def test_bootstrap_confidence_interval():
    """Verify that bootstrap confidence intervals produce non-empty, ordered bounds."""
    values = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.05, 0.18, 0.22])
    low, high = compute_bootstrap_ci(values, num_rounds=500, confidence_level=0.95, seed=42)

    assert low < high
    assert low >= np.min(values)
    assert high <= np.max(values)
