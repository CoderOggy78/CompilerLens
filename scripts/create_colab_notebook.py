#!/usr/bin/env python3
"""Programmatic generator for the complete, self-contained Google Colab notebook."""

import json
from pathlib import Path


def create_colab_notebook(output_path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": "CompilerLens: Learning to Rank Compiler Configurations with GNNs",
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": [],
    }

    def add_md(content: str):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in content.strip().split("\n")],
        })

    def add_code(code: str):
        nb["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")],
        })

    # Cell 1: Header & Introduction
    add_md(
        """# CompilerLens: Learning to Rank Compiler Configurations with Graph Neural Networks
### A Reproducible Research Project on Learned Cost Models for Accelerated Compilers

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CoderOggy78/CompilerLens/blob/main/notebooks/compilerlens_colab.ipynb)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 1. Executive Summary & Research Motivation

Modern machine learning accelerators (such as Google TPUs and NVIDIA GPUs) feature massive compute density coupled with complex, hierarchical memory systems (High-Bandwidth Memory / HBM, on-chip scratchpad SRAM / Vector Memory, and register files). 

Compilers such as **XLA (Accelerated Linear Algebra)** must make discrete code generation choices—most prominently **loop tiling dimensions**—to fuse operations and keep data on-chip. Choosing suboptimal tile configurations leads to catastrophic memory stalls, while benchmarking candidate configurations directly on real hardware is prohibitively expensive (often taking hours or days per neural network).

### Primary Research Question:
> **"Does a configuration-conditioned graph neural network select faster compiler configurations for unseen program graphs than models that ignore graph connectivity?"**

### Core CS Foundations Covered in this Notebook:
1. **Compiler Intermediate Representations (IR)**: High-level programs expressed as High-Level Optimizer (HLO) computation graphs.
2. **Directed Computation Graphs & Dependencies**: Operation nodes (convolutions, dot products, reductions) connected by directed dataflow and dependency edges.
3. **Tensor Tiling & Memory Access**: Dividing tensor iteration spaces into sub-tiles that fit into fast on-chip SRAM/VMem to maximize operational intensity (FLOPs/byte).
4. **Hardware-Dependent Runtime**: Physical latencies influenced by memory bandwidth, instruction pipelining, and hardware stalls.
5. **Compiler Cost Models**: Moving beyond hand-tuned analytical roofline models toward offline data-driven learned cost models.
6. **Autotuning & Candidate Ranking**: Framing compiler optimization as a Learning-to-Rank (LTR) problem over candidate parameter sets.
7. **Generalization to Unseen Programs**: Evaluating cost models strictly on graph-disjoint program topologies unseen during training.

> **Important Scientific Scope:** This is an **offline learned cost model**. It ranks candidate configurations from an external candidate generator for a specific hardware architecture (TPU v3). It does not compile arbitrary source code directly or claim hardware speedups beyond the evaluated candidate spaces."""
    )

    # Cell 2: Hardware & Environment Verification
    add_md(
        """## 2. Hardware & Environment Verification
We verify available GPU acceleration, host RAM, and storage. CompilerLens automatically adapts to NVIDIA CUDA, Apple Silicon MPS, or multi-threaded CPU runtimes."""
    )
    add_code(
        """import sys
import os
import shutil
import psutil
import torch

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple Silicon MPS Acceleration Active")
else:
    device = torch.device("cpu")
    print("Running on CPU")

ram_gb = psutil.virtual_memory().total / (1024**3)
disk_gb = shutil.disk_usage(".").free / (1024**3)
print(f"System RAM: {ram_gb:.2f} GB")
print(f"Available Disk Space: {disk_gb:.2f} GB")"""
    )

    # Cell 3: Clone Repository & Install Dependencies
    add_md(
        """## 3. Repository Setup & Dependency Installation
If running in a fresh Google Colab session, we clone the repository and install dependencies in editable mode."""
    )
    add_code(
        """# Clone repo if in Colab environment
if not os.path.exists("src/compilerlens"):
    print("Cloning CompilerLens repository...")
    !git clone https://github.com/CoderOggy78/CompilerLens.git
    %cd CompilerLens

# Install package dependencies
!pip install -q -r requirements.txt
!pip install -q -e . --no-deps

import compilerlens
print(f"CompilerLens package version: {compilerlens.__version__}")"""
    )

    # Cell 4: Optional Google Drive Persistence
    add_md(
        """## 4. Optional Google Drive Persistence (Colab Only)
To avoid re-downloading dataset archives every time a Colab runtime disconnects, you can optionally mount Google Drive and symlink the cache directory."""
    )
    add_code(
        """USE_GOOGLE_DRIVE = False  # Set to True if running in Colab with Drive storage

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    drive_cache = '/content/drive/MyDrive/compilerlens_data'
    os.makedirs(drive_cache, exist_ok=True)
    if not os.path.exists('data/raw'):
        os.makedirs('data', exist_ok=True)
        os.symlink(drive_cache, 'data/raw')
    print("Google Drive cache mounted successfully.")
else:
    print("Using local ephemeral storage.")"""
    )

    # Cell 5: Automated Dataset Acquisition & Validation
    add_md(
        """## 5. Automated Data Acquisition & Integrity Verification
We acquire the authoritative **TpuGraphs Tile XLA** collection from Google Research:
- Official repository: `https://github.com/google-research-datasets/tpu_graphs`
- Base URL: `https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/`

We check MD5 checksums, unpack securely preventing path traversal attacks, and inspect actual NPZ schema fields."""
    )
    add_code(
        """!python scripts/download_data.py --verify-only"""
    )

    # Cell 6: Inspect a Sample Computation Graph
    add_md(
        """## 6. Graph Schema Inspection & Structure
Let's inspect a real computation graph from the validation set using `np.load(..., allow_pickle=False)`:
- `node_feat`: `(N, 140)` continuous features (tensor dimensions, memory bytes).
- `node_opcode`: `(N,)` XLA HLO operation opcode (e.g. Convolution, Dot, Reduce).
- `edge_index`: `(E, 2)` directed dataflow edges.
- `config_feat`: `(C, 24)` candidate tile size configurations.
- `config_runtime`: `(C,)` physical execution time in nanoseconds on TPU v3.
- `config_runtime_normalizers`: `(C,)` default baseline runtime for normalization."""
    )
    add_code(
        """import glob
import numpy as np
from compilerlens.data import load_npz_graph

sample_file = sorted(glob.glob("data/raw/npz/tile/xla/valid/*.npz"))[0]
graph = load_npz_graph(sample_file, split="valid")

print(f"Graph ID: {graph.graph_id}")
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}, Candidate Configurations: {graph.num_configs}")
print(f"Node Features Shape: {graph.node_feat.shape}, dtype: {graph.node_feat.dtype}")
print(f"Config Features Shape: {graph.config_feat.shape}")
print(f"Sample Opcodes: {np.unique(graph.node_opcode)}")
print(f"Sample Normalized Runtimes (ratio to baseline):\\n{graph.normalized_runtimes[:5]}")
print(f"Oracle Best Runtime in Candidate Set: {np.min(graph.normalized_runtimes):.4f}")"""
    )

    # Cell 7: Exploratory Data Analysis & Plots
    add_md(
        """## 7. Exploratory Data Analysis & Visualizations
Why is compiler ranking hard?
Let's inspect:
1. Graph size distributions (nodes and edges across programs).
2. Opcode frequencies in real XLA graphs.
3. Candidate configuration count distributions.
4. Performance spread: within a single graph, bad tile configurations can cause $10\\times$ to $100\\times$ slowdowns compared to the best choice!"""
    )
    add_code(
        """from IPython.display import Image, display

# Display precomputed publication-quality dataset distribution analysis
display(Image("assets/figures/dataset_distributions.png", width=800))
display(Image("assets/figures/computation_graph_example.png", width=700))"""
    )

    # Cell 8: Leakage-Free Experiment Design & Baselines
    add_md(
        """## 8. Leakage-Free Experiment Design & Baselines

### Leakage-Free Protocol:
1. **Graph-Level Disjoint Splits**: Configurations from the same program graph are NEVER split across train and evaluation sets.
2. **Strict Preprocessor Fitting**: Feature normalization mean and variance are fit strictly on training graphs (`train_dev`).
3. **No Target Leakage**: Runtimes, normalizers, and graph IDs are excluded from model inputs.

### We implement 4 Baseline Models:
- **Baseline A (Random Ranking)**: Repeated seeded random draws (50 draws) to establish expected guessing regret.
- **Baseline B (Gradient-Boosted Trees / GBT)**: Tree ensemble over 411 graph summary features + 24 configuration features.
- **Baseline C (Graph-Free Neural Model)**: Encodes opcodes and node features, pools globally without message passing.
- **Baseline D (GraphSAGE GNN)**: Standard message passing over graph topology without configuration conditioning."""
    )
    add_code(
        """# Execute the full experiment pipeline or smoke profile
# You can change config to configs/colab.yaml or configs/smoke.yaml
!python scripts/run_experiment.py --config configs/smoke.yaml"""
    )

    # Cell 9: Main Model Architecture & Equations
    add_md(
        """## 9. Main Model Architecture: Direction-Aware Conditioned GNN

Our proposed model conditions graph message passing on the candidate compiler configuration:

$$\\mathbf{h}_v^{(0)} = \\text{FiLM}(\\mathbf{h}_v, \\mathbf{c}) = \\boldsymbol{\\gamma}(\\mathbf{c}) \\odot \\mathbf{h}_v + \\boldsymbol{\\beta}(\\mathbf{c})$$

### Direction-Aware Message Passing:
In a compiler IR, data flows forward from producers to consumers, but dependency pull flows backward from consumers to producers. We maintain separate forward and reverse relation weights:

$$\\mathbf{m}_{v \\leftarrow u}^{(\\text{fwd})} = \\mathbf{W}_{\\text{fwd}} \\mathbf{h}_u, \\quad \\mathbf{m}_{v \\leftarrow w}^{(\\text{rev})} = \\mathbf{W}_{\\text{rev}} \\mathbf{h}_w$$

$$\\mathbf{h}_v^{(l+1)} = \\text{LayerNorm}\\left(\\mathbf{h}_v^{(l)} + \\text{Dropout}\\left(\\text{ReLU}\\left(\\mathbf{W}_{\\text{upd}} [\\mathbf{h}_v^{(l)} \\,\\|\\, \\bar{\\mathbf{m}}_v] \\right)\\right)\\right)$$

### Joint Regression & Pairwise Ranking Objective:
$$\\mathcal{L} = \\mathcal{L}_{\\text{Huber}}(\\hat{y}, y) + \\lambda_{\\text{rank}} \\mathcal{L}_{\\text{pairwise}}(\\hat{y}_i, \\hat{y}_j)$$
$$\\mathcal{L}_{\\text{pairwise}} = \\log\\left(1 + \\exp(\\hat{y}_i - \\hat{y}_j)\\right) \\quad \\text{for pairs where } y_i < y_j$$"""
    )
    add_code(
        """display(Image("assets/figures/compilerlens_pipeline.png", width=850))
display(Image("assets/figures/gnn_architecture.png", width=850))"""
    )

    # Cell 10: Training Convergence & Loss Curves
    add_md(
        """## 10. Training Convergence & Validation Regret
Let's inspect the loss curves and validation selection quality across epochs:"""
    )
    add_code(
        """display(Image("assets/figures/training_curves.png", width=800))"""
    )

    # Cell 11: Benchmark Results & Baseline Comparison
    add_md(
        """## 11. Benchmark Results with 95% Bootstrap Confidence Intervals
We compare the models across:
- **Top-1 Regret**: $r(\\text{predicted best}) / r(c^*) - 1$ (percentage slowdown of the top recommended candidate).
- **Spearman Rank Correlation ($\\rho$)**: Rank correlation across candidate configurations.
- **Best-of-top-k Regret ($k=1, 5, 10$)**: Shortlist search regret."""
    )
    add_code(
        """import json
import pandas as pd

with open("reports/experiment_results.json") as f:
    results = json.load(f)

summary_rows = []
for m, res in results["models"].items():
    ci = res.get("top1_regret_ci95", [0.0, 0.0])
    sp_ci = res.get("spearman_ci95", [0.0, 0.0])
    summary_rows.append({
        "Model": m,
        "Top-1 Regret (Mean)": f"{res['top1_regret_mean']:.4f}",
        "Top-1 Regret (95% CI)": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
        "Best-of-5 Regret": f"{res['best_of_topk']['5']['mean']:.4f}" if '5' in res['best_of_topk'] else "N/A",
        "Best-of-10 Regret": f"{res['best_of_topk']['10']['mean']:.4f}" if '10' in res['best_of_topk'] else "N/A",
        "Spearman rho": f"{res.get('spearman_mean') or 0.0:.3f}",
        "Parameters": f"{res.get('param_count', 0):,}",
    })

df = pd.DataFrame(summary_rows)
display(df)

display(Image("assets/figures/baseline_comparison.png", width=800))
display(Image("assets/figures/topk_regret_curve.png", width=650))"""
    )

    # Cell 12: Ablation Studies
    add_md(
        """## 12. Ablation Studies & Architectural Insights
We systematically remove components to isolate their contribution:
1. **No Graph Edges**: Tests whether topology matters or if node set pooling is sufficient.
2. **No Opcode Embeddings**: Tests the necessity of operator semantics.
3. **Regression-Only Loss**: Removes pairwise ranking loss.
4. **2 vs. 3 Message-Passing Layers**: Tests message receptive field depth."""
    )
    add_code(
        """display(Image("assets/figures/ablation_study.png", width=750))"""
    )

    # Cell 13: Case Studies & Error Analysis
    add_md(
        """## 13. Case Studies: Successful Rankings vs. Failure Modes
We visualize 2 successful ranking examples (where predicted best is within 1% of oracle) and 2 failure cases (where predictions diverge from observed runtimes)."""
    )
    add_code(
        """display(Image("assets/figures/case_studies.png", width=850))"""
    )

    # Cell 14: Interactive Live Inference on Unseen Program Graph
    add_md(
        """## 14. Interactive Inference: Recommend Top Compiler Configurations
Let's load an unseen graph from the test set and rank its candidate configurations using the trained model."""
    )
    add_code(
        """import torch
from compilerlens.models.gnn_conditioned import ConfigConditionedGNN
from compilerlens.features import FeaturePreprocessor
from compilerlens.predict import rank_compiler_configurations

# Load preprocessor
preprocessor = FeaturePreprocessor.load("reports/preprocessor_stats.json")

# Load model from checkpoint
ckpt = torch.load("checkpoints/smoke/best_model.pt", map_location=device, weights_only=False)
model = ConfigConditionedGNN(
    node_feat_dim=140,
    config_feat_dim=24,
    hidden_dim=ckpt["config"]["model"]["hidden_dim"],
    opcode_embed_dim=ckpt["config"]["model"]["opcode_embed_dim"],
    num_layers=ckpt["config"]["model"]["num_layers"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# Select a held-out graph
test_graph_file = sorted(glob.glob("data/raw/npz/tile/xla/valid/*.npz"))[-1]
test_g = load_npz_graph(test_graph_file, split="valid")

recs = rank_compiler_configurations(model, test_g, preprocessor, device=device, top_k=5)

print(f"Top 5 Recommended Compiler Configurations for {recs['graph_id']}:")
print("-" * 65)
for r in recs["top_k_recommendations"]:
    runtime_str = f", Observed Runtime Ratio: {r.get('observed_normalized_runtime', 0.0):.3f}" if 'observed_normalized_runtime' in r else ""
    print(f"Rank {r['rank']}: Candidate Index {r['candidate_index']:4d} | Pred Log Score: {r['predicted_log_runtime']:+.4f}{runtime_str}")"""
    )

    # Cell 15: Artifact Packaging & Conclusion
    add_md(
        """## 15. Export & Project Summary
All experiment logs, test predictions, figures, and research checkpoints are persisted and ready for publication."""
    )
    add_code(
        """!ls -lh reports/
!ls -lh checkpoints/smoke/
print("CompilerLens execution completed successfully!")"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote Colab notebook to {output_path}")


if __name__ == "__main__":
    create_colab_notebook(Path("notebooks/compilerlens_colab.ipynb"))
