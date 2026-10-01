"""Evaluation metrics and protocols for compiler configuration ranking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

from compilerlens.data import GraphData


@dataclass
class GraphEvaluationResult:
    """Evaluation metrics for a single computation graph's candidate rankings."""

    graph_id: str
    num_candidates: int
    oracle_best_runtime: float
    predicted_best_runtime: float
    top1_regret: float
    best_of_topk_regret: Dict[int, float]
    spearman_rho: Optional[float]
    log_mae: Optional[float]
    log_rmse: Optional[float]


def evaluate_single_graph(
    pred_scores: np.ndarray,
    observed_normalized_runtimes: np.ndarray,
    observed_log_runtimes: Optional[np.ndarray] = None,
    top_k_list: Sequence[int] = (1, 5, 10),
    graph_id: str = "unknown",
) -> GraphEvaluationResult:
    """Evaluate candidate ranking quality on a single computation graph.

    Args:
        pred_scores: 1D array of model predicted scores (lower is predicted faster).
        observed_normalized_runtimes: 1D array of ground truth normalized runtimes (lower is faster).
        observed_log_runtimes: Optional 1D array of ground truth log normalized runtimes.
        top_k_list: Shortlist candidate counts for best-of-top-k evaluation.
        graph_id: Identifier string for reporting.

    Returns:
        GraphEvaluationResult.
    """
    assert pred_scores.shape == observed_normalized_runtimes.shape, (
        f"Mismatched shapes: preds {pred_scores.shape} vs targets {observed_normalized_runtimes.shape}"
    )
    num_candidates = len(pred_scores)
    assert num_candidates > 0, "Cannot evaluate empty candidate set."

    # Identify oracle best among evaluated candidates
    oracle_idx = int(np.argmin(observed_normalized_runtimes))
    oracle_runtime = float(observed_normalized_runtimes[oracle_idx])
    assert oracle_runtime > 0, f"Oracle runtime must be positive, got {oracle_runtime}"

    # Sort candidates by predicted score ascending (lowest score = fastest predicted)
    sorted_candidate_indices = np.argsort(pred_scores)

    # Top-1 regret
    pred_best_idx = int(sorted_candidate_indices[0])
    pred_best_runtime = float(observed_normalized_runtimes[pred_best_idx])
    top1_regret = float(pred_best_runtime / oracle_runtime - 1.0)
    top1_regret = max(0.0, top1_regret)  # Guard against precision errors

    # Best-of-top-k regret
    best_of_topk = {}
    for k in top_k_list:
        effective_k = min(k, num_candidates)
        shortlist_indices = sorted_candidate_indices[:effective_k]
        best_shortlist_runtime = float(np.min(observed_normalized_runtimes[shortlist_indices]))
        regret_k = float(best_shortlist_runtime / oracle_runtime - 1.0)
        best_of_topk[k] = max(0.0, regret_k)

    # Spearman rank correlation
    # Note: If either array is constant, Spearman correlation is undefined
    if (
        np.all(pred_scores == pred_scores[0])
        or np.all(observed_normalized_runtimes == observed_normalized_runtimes[0])
        or num_candidates < 2
    ):
        spearman_rho = None
    else:
        res = stats.spearmanr(pred_scores, observed_normalized_runtimes)
        spearman_rho = float(res.statistic) if not np.isnan(res.statistic) else None

    # Regression error metrics if log runtimes available
    log_mae = None
    log_rmse = None
    if observed_log_runtimes is not None:
        diff = pred_scores - observed_log_runtimes
        log_mae = float(np.mean(np.abs(diff)))
        log_rmse = float(np.sqrt(np.mean(diff**2)))

    return GraphEvaluationResult(
        graph_id=graph_id,
        num_candidates=num_candidates,
        oracle_best_runtime=oracle_runtime,
        predicted_best_runtime=pred_best_runtime,
        top1_regret=top1_regret,
        best_of_topk_regret=best_of_topk,
        spearman_rho=spearman_rho,
        log_mae=log_mae,
        log_rmse=log_rmse,
    )


def compute_bootstrap_ci(
    values: np.ndarray,
    num_rounds: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute empirical bootstrap confidence interval over graph-level metrics."""
    if len(values) < 2:
        val = float(values[0]) if len(values) == 1 else 0.0
        return val, val

    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.empty(num_rounds, dtype=np.float64)

    for i in range(num_rounds):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return lower, upper


def evaluate_predictions(
    graph_predictions: Dict[str, np.ndarray],
    ground_truth_graphs: Sequence[GraphData],
    top_k_list: Sequence[int] = (1, 5, 10),
    bootstrap_rounds: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Aggregate evaluation across all graphs in the evaluation set.

    Reports:
      - Mean and median Top-1 regret
      - Mean and median Best-of-top-k regret for k in top_k_list
      - Mean and median Spearman rank correlation (with undefined count reported)
      - Mean MAE and RMSE on log normalized runtime
      - 95% bootstrap confidence intervals across graphs
    """
    gt_map = {g.graph_id: g for g in ground_truth_graphs}
    per_graph_results: List[GraphEvaluationResult] = []

    for g_id, preds in graph_predictions.items():
        if g_id not in gt_map:
            continue
        g = gt_map[g_id]
        if g.normalized_runtimes is None:
            continue

        res = evaluate_single_graph(
            pred_scores=preds,
            observed_normalized_runtimes=g.normalized_runtimes,
            observed_log_runtimes=g.log_normalized_runtimes,
            top_k_list=top_k_list,
            graph_id=g_id,
        )
        per_graph_results.append(res)

    if not per_graph_results:
        raise ValueError("No matching labeled graphs found to evaluate.")

    num_graphs = len(per_graph_results)
    top1_regrets = np.array([r.top1_regret for r in per_graph_results], dtype=np.float64)

    # Top-k regrets
    topk_regrets_dict = {k: [] for k in top_k_list}
    for r in per_graph_results:
        for k in top_k_list:
            if k in r.best_of_topk_regret:
                topk_regrets_dict[k].append(r.best_of_topk_regret[k])

    # Spearman correlations
    spearman_values = [r.spearman_rho for r in per_graph_results if r.spearman_rho is not None]
    undefined_spearman_count = num_graphs - len(spearman_values)

    # Regression errors
    mae_values = [r.log_mae for r in per_graph_results if r.log_mae is not None]
    rmse_values = [r.log_rmse for r in per_graph_results if r.log_rmse is not None]

    # Bootstrap CIs
    ci_top1 = compute_bootstrap_ci(top1_regrets, num_rounds=bootstrap_rounds, seed=seed)
    ci_topk = {
        k: compute_bootstrap_ci(np.array(topk_regrets_dict[k]), num_rounds=bootstrap_rounds, seed=seed)
        for k in top_k_list
    }

    summary: Dict[str, Any] = {
        "num_graphs": num_graphs,
        "undefined_spearman_count": undefined_spearman_count,
        "top1_regret_mean": float(np.mean(top1_regrets)),
        "top1_regret_median": float(np.median(top1_regrets)),
        "top1_regret_ci95": ci_top1,
        "best_of_topk": {},
    }

    for k in top_k_list:
        arr = np.array(topk_regrets_dict[k])
        summary["best_of_topk"][k] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "ci95": ci_topk[k],
        }

    if spearman_values:
        arr_sp = np.array(spearman_values)
        summary["spearman_mean"] = float(np.mean(arr_sp))
        summary["spearman_median"] = float(np.median(arr_sp))
        summary["spearman_ci95"] = compute_bootstrap_ci(arr_sp, num_rounds=bootstrap_rounds, seed=seed)
    else:
        summary["spearman_mean"] = None
        summary["spearman_median"] = None
        summary["spearman_ci95"] = None

    if mae_values:
        summary["log_mae_mean"] = float(np.mean(mae_values))
        summary["log_rmse_mean"] = float(np.mean(rmse_values))
    else:
        summary["log_mae_mean"] = None
        summary["log_rmse_mean"] = None

    summary["per_graph_results"] = [asdict(r) for r in per_graph_results]
    return summary


def save_evaluation_report(
    eval_summary: Dict[str, Any], filepath: Union[str, Path]
) -> None:
    """Save evaluation metrics and per-graph results to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(eval_summary, f, indent=2)
