"""Research-grade visualization suite for CompilerLens architecture and benchmark results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

from compilerlens.data import GraphData

logger = logging.getLogger(__name__)

# Configure publication-quality matplotlib style
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
ACCENT_GREEN = "#2ca02c"
ACCENT_RED = "#d62728"
ACCENT_PURPLE = "#9467bd"


def save_figures(fig: plt.Figure, base_path: Union[str, Path]) -> None:
    """Save figure in both high-resolution PNG (300 DPI) and vector SVG formats."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figures to {png_path} and {svg_path}")


def plot_pipeline_diagram(output_path: Union[str, Path]) -> None:
    """Generate high-resolution end-to-end pipeline diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Define stage boxes
    boxes = [
        {"x": 0.05, "y": 0.55, "w": 0.22, "h": 0.35, "title": "1. Program IR Extraction", "desc": "• XLA HLO computation graph\n• N nodes with 140 features\n• Opcode categories (0-127)\n• Directed dataflow edges", "color": "#E3F2FD", "edge": "#1E88E5"},
        {"x": 0.36, "y": 0.55, "w": 0.24, "h": 0.35, "title": "2. Candidate Search Space", "desc": "• Fused tile configurations\n• 24-dim tiling parameters\n• C candidates per graph\n• Target: log normalized runtime", "color": "#FFF3E0", "edge": "#FB8C00"},
        {"x": 0.69, "y": 0.55, "w": 0.26, "h": 0.35, "title": "3. Direction-Aware GNN", "desc": "• Opcode & feature projections\n• Configuration FiLM modulation\n• Bidirectional message passing\n• Global graph pooling", "color": "#E8F5E9", "edge": "#43A047"},
        {"x": 0.20, "y": 0.08, "w": 0.28, "h": 0.32, "title": "4. Learned Cost Head", "desc": "• Concatenated graph & config\n• Regression Huber loss\n• Pairwise ranking loss\n• Sign-verified ordering", "color": "#F3E5F5", "edge": "#8E24AA"},
        {"x": 0.55, "y": 0.08, "w": 0.30, "h": 0.32, "title": "5. Top-k Autotuning Selection", "desc": "• Ranked candidate shortlist\n• Top-1 zero-shot recommendation\n• Best-of-top-k search regret\n• Deployment candidate export", "color": "#FBE9E7", "edge": "#D84315"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=b["color"], edgecolor=b["edge"], linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] - 0.06, b["title"],
                ha="center", va="center", fontsize=11, fontweight="bold", color="#212121")
        ax.text(b["x"] + 0.02, b["y"] + b["h"] - 0.12, b["desc"],
                ha="left", va="top", fontsize=9, color="#424242", linespacing=1.4)

    # Arrows between boxes
    ax.annotate("", xy=(0.35, 0.72), xytext=(0.28, 0.72),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.68, 0.72), xytext=(0.61, 0.72),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.34, 0.41), xytext=(0.75, 0.54),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2, connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(0.54, 0.24), xytext=(0.49, 0.24),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))

    ax.set_title("CompilerLens: Learning to Rank Compiler Configurations with Graph Neural Networks",
                 fontsize=14, fontweight="bold", pad=20)
    save_figures(fig, output_path)


def plot_model_architecture(output_path: Union[str, Path]) -> None:
    """Generate high-resolution diagram of the GNN model architecture with tensor shapes."""
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    layers = [
        {"x": 0.05, "y": 0.65, "w": 0.18, "h": 0.25, "name": "Node Inputs", "shapes": "Opcode: (N,)\nFeat: (N, 140)\nEdges: (2, E)", "bg": "#E1F5FE", "border": "#0288D1"},
        {"x": 0.05, "y": 0.25, "w": 0.18, "h": 0.25, "name": "Config Input", "shapes": "Config Feat:\n(C, 24)\nTile parameters", "bg": "#FFF8E1", "border": "#FFA000"},
        {"x": 0.28, "y": 0.65, "w": 0.18, "h": 0.25, "name": "Node Projections", "shapes": "Opcode Emb: (N, 32)\nLinear+LN: (N, 96)\nBase Repr: (N, 128)", "bg": "#EDE7F6", "border": "#512DA8"},
        {"x": 0.28, "y": 0.25, "w": 0.18, "h": 0.25, "name": "Config MLP", "shapes": "Linear+ReLU (64)\nLinear+LN (128)\nConfig Emb: (C, 128)", "bg": "#EDE7F6", "border": "#512DA8"},
        {"x": 0.51, "y": 0.45, "w": 0.20, "h": 0.40, "name": "FiLM Modulation\n& Direction GNN", "shapes": "FiLM: gamma*h + beta\nShape: (C, N, 128)\n---\n3x Directional Layers:\nW_fwd * h + W_rev * h\nResidual + LayerNorm", "bg": "#E8F5E9", "border": "#388E3C"},
        {"x": 0.76, "y": 0.45, "w": 0.19, "h": 0.40, "name": "Pooling & Readout", "shapes": "Mean+Max Pooling:\nShape: (C, 256)\n---\nConcat with Config:\nShape: (C, 384)\n---\nMLP Head -> (C, 1)\nlog normalized runtime", "bg": "#FBE9E7", "border": "#D84315"},
    ]

    for lay in layers:
        rect = patches.FancyBboxPatch(
            (lay["x"], lay["y"]), lay["w"], lay["h"],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=lay["bg"], edgecolor=lay["border"], linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(lay["x"] + lay["w"] / 2, lay["y"] + lay["h"] - 0.04, lay["name"],
                ha="center", va="center", fontsize=10, fontweight="bold", color="#212121")
        ax.text(lay["x"] + lay["w"] / 2, lay["y"] + (lay["h"] - 0.08) / 2, lay["shapes"],
                ha="center", va="center", fontsize=8.5, color="#37474F", linespacing=1.3)

    # Connections
    ax.annotate("", xy=(0.27, 0.77), xytext=(0.24, 0.77),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.27, 0.37), xytext=(0.24, 0.37),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.50, 0.70), xytext=(0.47, 0.77),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.50, 0.55), xytext=(0.47, 0.37),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.75, 0.65), xytext=(0.72, 0.65),
                arrowprops=dict(facecolor="#555", edgecolor="#555", arrowstyle="->", lw=2))

    ax.set_title("CompilerLens: Configuration-Conditioned Direction-Aware GNN Architecture",
                 fontsize=14, fontweight="bold", pad=20)
    save_figures(fig, output_path)


def plot_computation_graph_sample(graph: GraphData, output_path: Union[str, Path], max_nodes: int = 40) -> None:
    """Plot an actual computation graph or representative subgraph."""
    fig, ax = plt.subplots(figsize=(10, 8))

    num_nodes = min(graph.num_nodes, max_nodes)
    sub_opcodes = graph.node_opcode[:num_nodes]

    # Filter edges within subgraph
    edges = graph.edge_index
    mask = (edges[:, 0] < num_nodes) & (edges[:, 1] < num_nodes)
    sub_edges = edges[mask]

    # Layout nodes in pseudo-topological layers
    rng = np.random.RandomState(42)
    pos = {}
    for i in range(num_nodes):
        # Layer by index ratio
        layer = int((i / num_nodes) * 6)
        x = layer + (rng.rand() - 0.5) * 0.4
        y = (i % 6) + (rng.rand() - 0.5) * 0.4
        pos[i] = (x, y)

    # Draw edges
    for e in sub_edges:
        src, dst = int(e[0]), int(e[1])
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#90A4AE", lw=1.2, shrinkA=8, shrinkB=8, alpha=0.7),
        )

    # Draw nodes colored by opcode
    unique_opcodes = np.unique(sub_opcodes)
    cmap = plt.get_cmap("tab20")
    opcode_to_color = {op: cmap(idx % 20) for idx, op in enumerate(unique_opcodes)}

    for i in range(num_nodes):
        x, y = pos[i]
        op = sub_opcodes[i]
        color = opcode_to_color[op]
        circle = plt.Circle((x, y), 0.16, color=color, ec="#37474F", lw=1.2, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, f"v{i}\n({op})", ha="center", va="center", fontsize=6, fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.8, 6.2)
    ax.axis("off")
    ax.set_title(
        f"Real Computation Subgraph: {graph.graph_id[:24]}...\n"
        f"({num_nodes}/{graph.num_nodes} nodes, {len(sub_edges)}/{graph.num_edges} edges, {len(unique_opcodes)} opcodes)",
        fontsize=12, fontweight="bold", pad=15
    )
    save_figures(fig, output_path)


def plot_dataset_distributions(
    train_graphs: Sequence[GraphData],
    val_graphs: Sequence[GraphData],
    output_path: Union[str, Path],
) -> None:
    """Plot dataset structural distributions, opcode frequencies, and normalized runtimes."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Graph node counts
    train_nodes = [g.num_nodes for g in train_graphs]
    val_nodes = [g.num_nodes for g in val_graphs]
    ax = axes[0, 0]
    ax.hist(train_nodes, bins=30, alpha=0.6, label=f"Train ({len(train_nodes)})", color=PRIMARY_COLOR, density=True)
    ax.hist(val_nodes, bins=30, alpha=0.6, label=f"Valid ({len(val_nodes)})", color=SECONDARY_COLOR, density=True)
    ax.set_xlabel("Number of Nodes per Graph")
    ax.set_ylabel("Density")
    ax.set_title("Node Count Distribution by Split")
    ax.legend()

    # 2. Candidate configuration counts
    train_configs = [g.num_configs for g in train_graphs]
    val_configs = [g.num_configs for g in val_graphs]
    ax = axes[0, 1]
    ax.hist(train_configs, bins=25, alpha=0.6, label="Train", color=PRIMARY_COLOR)
    ax.hist(val_configs, bins=25, alpha=0.6, label="Valid", color=SECONDARY_COLOR)
    ax.set_xlabel("Configurations per Graph")
    ax.set_ylabel("Count of Graphs")
    ax.set_title("Candidate Configuration Set Size Distribution")
    ax.legend()

    # 3. Top Opcode frequencies
    all_opcodes = []
    for g in train_graphs:
        all_opcodes.extend(g.node_opcode.tolist())
    vals, counts = np.unique(all_opcodes, return_counts=True)
    sorted_idx = np.argsort(counts)[::-1][:15]
    top_ops = vals[sorted_idx]
    top_counts = counts[sorted_idx]

    ax = axes[1, 0]
    ax.bar([str(op) for op in top_ops], top_counts, color="#5C6BC0", edgecolor="#283593")
    ax.set_xlabel("XLA Opcode ID")
    ax.set_ylabel("Frequency Across Nodes")
    ax.set_title("Top 15 Most Frequent XLA Opcodes")
    ax.tick_params(axis="x", rotation=45)

    # 4. Normalized runtime distribution
    all_log_runtimes = []
    for g in train_graphs:
        if g.log_normalized_runtimes is not None:
            all_log_runtimes.extend(g.log_normalized_runtimes.tolist())

    ax = axes[1, 1]
    ax.hist(all_log_runtimes, bins=40, color="#26A69A", edgecolor="#004D40", alpha=0.8)
    ax.set_xlabel("Log Normalized Runtime: log(runtime / baseline)")
    ax.set_ylabel("Frequency")
    ax.set_title("Observed Performance Distribution (Train)")
    ax.axvline(0.0, color="red", linestyle="--", label="Default Baseline (0.0)")
    ax.legend()

    plt.tight_layout()
    save_figures(fig, output_path)


def plot_training_curves(history: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Plot multi-epoch training loss, regression loss, ranking loss, and validation regret."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    reg_loss = [h.get("train_reg_loss", 0.0) for h in history]
    rank_loss = [h.get("train_rank_loss", 0.0) for h in history]
    val_regret = [h["val_top1_regret_mean"] for h in history]
    val_spearman = [h.get("val_spearman_mean", 0.0) or 0.0 for h in history]

    # Losses
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Total Loss", color="#1565C0", lw=2, marker="o")
    ax.plot(epochs, reg_loss, label="Huber Regression Loss", color="#00897B", lw=1.5, linestyle="--")
    ax.plot(epochs, rank_loss, label="Pairwise Ranking Loss", color="#FB8C00", lw=1.5, linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Convergence")
    ax.legend()

    # Validation metrics
    ax = axes[1]
    ax.plot(epochs, val_regret, label="Val Top-1 Regret (lower is better)", color=ACCENT_RED, lw=2, marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Regret")
    ax.set_title("Validation Selection Quality")
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_figures(fig, output_path)


def plot_predicted_vs_observed(
    eval_summary: Dict[str, Any],
    ground_truth_graphs: Sequence[GraphData],
    predictions: Dict[str, np.ndarray],
    output_path: Union[str, Path],
    max_points: int = 1500,
) -> None:
    """Scatter plot of predicted log runtime versus observed log runtime."""
    fig, ax = plt.subplots(figsize=(7, 7))

    y_true_list = []
    y_pred_list = []

    for g in ground_truth_graphs:
        if g.graph_id in predictions and g.log_normalized_runtimes is not None:
            y_true_list.extend(g.log_normalized_runtimes.tolist())
            y_pred_list.extend(predictions[g.graph_id].tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    if len(y_true) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    ax.scatter(y_true, y_pred, alpha=0.3, color=PRIMARY_COLOR, edgecolors="none", s=20)

    # Reference line
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    ax.plot(lims, lims, color=ACCENT_RED, linestyle="--", lw=1.5, label="Ideal (y = x)")

    mae = eval_summary.get("log_mae_mean", float(np.mean(np.abs(y_pred - y_true))))
    spearman = eval_summary.get("spearman_mean", 0.0)

    ax.set_xlabel("Observed Log Normalized Runtime")
    ax.set_ylabel("Predicted Log Normalized Runtime")
    ax.set_title(f"Predicted vs. Observed Log Runtime\n(MAE = {mae:.3f}, Spearman rho = {spearman:.3f})", fontsize=11)
    ax.legend()
    save_figures(fig, output_path)


def plot_regret_distribution(eval_summary: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Plot violin and box plots of per-graph Top-1 and Top-k regrets."""
    fig, ax = plt.subplots(figsize=(8, 5))

    per_graph = eval_summary["per_graph_results"]
    data = []
    labels = []

    # Top-1
    top1 = [r["top1_regret"] for r in per_graph]
    data.append(top1)
    labels.append("Top-1 Regret")

    # Top-k
    for k in [5, 10]:
        vals = [r["best_of_topk_regret"][k] for r in per_graph if k in r["best_of_topk_regret"]]
        if vals:
            data.append(vals)
            labels.append(f"Best-of-{k} Regret")

    sns.boxplot(data=data, ax=ax, palette="Blues", width=0.4, fliersize=3)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Regret: r(selected) / r(oracle) - 1")
    ax.set_title("Per-Graph Regret Distribution Across Candidate Shortlists", fontsize=12, fontweight="bold")
    save_figures(fig, output_path)


def plot_baseline_comparison(
    results: Dict[str, Dict[str, Any]], output_path: Union[str, Path]
) -> None:
    """Bar chart comparing all baselines with 95% bootstrap confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results.keys())
    top1_means = [results[m]["top1_regret_mean"] for m in models]
    top1_cis = [
        (results[m]["top1_regret_mean"] - results[m]["top1_regret_ci95"][0],
         results[m]["top1_regret_ci95"][1] - results[m]["top1_regret_mean"])
        for m in models
    ]
    yerr_top1 = np.array(top1_cis).T

    spearman_means = [results[m].get("spearman_mean", 0.0) or 0.0 for m in models]
    spearman_cis = [
        (results[m].get("spearman_mean", 0.0) - results[m].get("spearman_ci95", [0.0, 0.0])[0],
         results[m].get("spearman_ci95", [0.0, 0.0])[1] - results[m].get("spearman_mean", 0.0))
        if results[m].get("spearman_mean") is not None else (0.0, 0.0)
        for m in models
    ]
    yerr_sp = np.array(spearman_cis).T

    colors = ["#78909C", "#FFA726", "#42A5F5", "#AB47BC", "#26A69A"]

    # Top-1 Regret
    ax = axes[0]
    bars = ax.bar(models, top1_means, yerr=yerr_top1, capsize=4, color=colors[:len(models)], edgecolor="#37474F")
    ax.set_ylabel("Top-1 Regret (lower is better)")
    ax.set_title("Top-1 Selection Regret (with 95% Bootstrap CI)")
    ax.tick_params(axis="x", rotation=30)

    # Spearman Correlation
    ax = axes[1]
    ax.bar(models, spearman_means, yerr=yerr_sp, capsize=4, color=colors[:len(models)], edgecolor="#37474F")
    ax.set_ylabel("Spearman Rank Correlation rho (higher is better)")
    ax.set_title("Ranking Correlation (with 95% Bootstrap CI)")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save_figures(fig, output_path)


def plot_topk_regret_curve(
    results: Dict[str, Dict[str, Any]], output_path: Union[str, Path]
) -> None:
    """Plot regret decay curve as shortlist candidate budget k increases."""
    fig, ax = plt.subplots(figsize=(8, 5))

    k_vals = [1, 5, 10]
    colors = {"Random": "#78909C", "GBT": "#FFA726", "GraphFree": "#42A5F5", "GraphSAGE": "#AB47BC", "CompilerLens GNN": "#26A69A"}

    for model_name, res in results.items():
        if "best_of_topk" in res:
            regrets = [res["best_of_topk"][k]["mean"] for k in k_vals if k in res["best_of_topk"]]
            ax.plot(k_vals[:len(regrets)], regrets, marker="o", lw=2, label=model_name, color=colors.get(model_name, None))

    ax.set_xlabel("Shortlist Candidate Budget (k)")
    ax.set_ylabel("Mean Regret: min_{c in top-k} r(c) / r(c*) - 1")
    ax.set_title("Best-of-Top-k Regret Decay Curve", fontsize=12, fontweight="bold")
    ax.set_xticks(k_vals)
    ax.legend()
    save_figures(fig, output_path)


def plot_ablation_comparison(
    ablation_results: Dict[str, float], output_path: Union[str, Path]
) -> None:
    """Plot ablation study results isolating architecture components."""
    fig, ax = plt.subplots(figsize=(9, 5))

    names = list(ablation_results.keys())
    regrets = list(ablation_results.values())

    colors = ["#26A69A" if "Full" in n else "#EF5350" for n in names]
    bars = ax.barh(names, regrets, color=colors, edgecolor="#263238")
    ax.set_xlabel("Top-1 Selection Regret (lower is better)")
    ax.set_title("Ablation Study: Architecture & Objective Components", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, regrets):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)

    save_figures(fig, output_path)


def plot_case_studies(
    eval_summary: Dict[str, Any],
    ground_truth_graphs: Sequence[GraphData],
    predictions: Dict[str, np.ndarray],
    output_path: Union[str, Path],
) -> None:
    """Plot 2 successful and 2 failed ranking examples selected by stated regret thresholds."""
    per_graph = eval_summary["per_graph_results"]
    # Sort by top-1 regret
    sorted_by_regret = sorted(per_graph, key=lambda r: r["top1_regret"])

    # 2 Successes: lowest regret (regret ~ 0)
    successes = sorted_by_regret[:2]
    # 2 Failures: highest regret
    failures = sorted_by_regret[-2:]

    case_studies = [
        ("Success 1 (Near-optimal)", successes[0]),
        ("Success 2 (Near-optimal)", successes[1]),
        ("Failure 1 (Suboptimal)", failures[0]),
        ("Failure 2 (Suboptimal)", failures[1]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (title, case) in enumerate(case_studies):
        ax = axes[idx // 2, idx % 2]
        g_id = case["graph_id"]
        g_preds = predictions[g_id]
        gt_g = next(g for g in ground_truth_graphs if g.graph_id == g_id)
        gt_runtimes = gt_g.normalized_runtimes

        # Candidate scatter
        ax.scatter(g_preds, gt_runtimes, alpha=0.6, color=PRIMARY_COLOR, label="Candidates")

        # Highlight predicted best and oracle best
        pred_best_idx = np.argmin(g_preds)
        oracle_best_idx = np.argmin(gt_runtimes)

        ax.scatter(
            [g_preds[pred_best_idx]], [gt_runtimes[pred_best_idx]],
            color=ACCENT_RED, s=100, marker="*", label=f"Predicted Best (r={gt_runtimes[pred_best_idx]:.2f})", zorder=5
        )
        ax.scatter(
            [g_preds[oracle_best_idx]], [gt_runtimes[oracle_best_idx]],
            color=ACCENT_GREEN, s=80, marker="o", facecolors="none", lw=2, label=f"Oracle Best (r={gt_runtimes[oracle_best_idx]:.2f})", zorder=6
        )

        ax.set_xlabel("Predicted Score (lower is predicted faster)")
        ax.set_ylabel("Observed Normalized Runtime")
        ax.set_title(f"{title}: {g_id[:18]}...\n(Top-1 Regret: {case['top1_regret']:.3f}, Spearman: {case['spearman_rho'] or 0.0:.2f})", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_figures(fig, output_path)
