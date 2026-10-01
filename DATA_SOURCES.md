# Data Sources and Attribution

This document records the provenance, licensing, checksums, retrieval dates, and schema definitions for all datasets used in the **CompilerLens** project.

---

## 1. Upstream Dataset: TpuGraphs (Tile Collection)

- **Dataset Name**: TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs
- **Official Repository**: [https://github.com/google-research-datasets/tpu_graphs](https://github.com/google-research-datasets/tpu_graphs)
- **Primary Research Paper**: 
  > Phothilimthana, P. M., Sabne, A., Chiu, N., et al. "TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs." *NeurIPS 2023 Track on Datasets and Benchmarks*, 2023. arXiv: [2308.13490](https://arxiv.org/abs/2308.13490).
- **Google Research Blog**: [Advancements in Machine Learning for Machine Learning](https://research.google/blog/advancements-in-machine-learning-for-machine-learning/)
- **Upstream License**: Creative Commons Attribution 4.0 International (CC-BY 4.0).
- **Retrieval Date**: 2026-10-01 / 2026-10-02

---

## 2. Archive Retrieval URLs & Checksums

| Archive File | Canonical HTTP URL | Recommended HTTPS Mirror (GCS) | Size (Bytes) | MD5 Checksum | CRC32C |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `npz_tile_xla_train.tar` | `http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_train.tar` | `https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_train.tar` | 158,965,760 | `528f9bcfd86581662f238a9cfb8613e7` | `6+hT0Q==` |
| `npz_tile_xla_valid.tar` | `http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_valid.tar` | `https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_valid.tar` | 15,923,200 | `eae09512dd0da8d6a6dddff4b922b278` | `+FEPHA==` |
| `npz_tile_xla_test.tar` | `http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_test.tar` | `https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_test.tar` | 15,912,960 | `3ad9d6c51fbd51937e55dffbe12d9856` | `VEug8A==` |

*Note: The upstream HTTP domain (`download.tensorflow.org`) serves identical artifacts from Google Cloud Storage (`storage.googleapis.com/download.tensorflow.org/...`), which supports standard TLS/HTTPS verification.*

---

## 3. Schema & Field Definitions

Each graph is serialized as an unpickled `.npz` file containing:

| Field Name | Type | Shape | Semantics & Units |
| :--- | :--- | :--- | :--- |
| `node_feat` | `float32` | `(N, 140)` | Continuous node attributes (tensor shapes, buffer sizes, memory footprints). Fit scaling solely on training graphs. |
| `node_opcode` | `uint8` / `int32` | `(N,)` | XLA HLO operation opcode (e.g., convolution, dot, reduce, broadcast). Values range from 0 to 127+. |
| `edge_index` | `int64` | `(E, 2)` | Directed edges `[u, v]` indicating dataflow (node `u` produces tensor consumed by node `v`, or vice versa). Both forward and reverse edges are maintained with distinct relation types. |
| `config_feat` | `float32` | `(C, 24)` | Compiler tile configuration parameters (loop tiling sizes and loop ordering for fused subgraphs). |
| `config_runtime` | `int64` | `(C,)` | Observed execution latency on Google TPU v3 in nanoseconds. (In `test`, set to 0 as official ground truth is withheld). |
| `config_runtime_normalizers` | `int64` | `(C,)` | Execution latency of the default unoptimized compiler baseline on the same TPU hardware. |

### Normalization and Target
The primary target metric is log normalized runtime:
$$\text{normalized\_runtime} = \frac{\text{config\_runtime}}{\text{config\_runtime\_normalizers}}$$
$$y = \log(\text{normalized\_runtime})$$
Lower values indicate faster execution relative to the default compiler configuration.

---

## 4. Graph Counts and Splits

- **Train Split (`train`)**: 5,709 unique computation graphs.
- **Validation Split (`valid`)**: 676 unique computation graphs.
- **Test Split (`test`)**: 844 unique computation graphs (unlabeled).

### Graph-Disjoint Development Split
Because test labels are withheld by the upstream benchmark hosts, we partition the official training set into:
- **`train_dev`** (80% of official training graphs, deterministic seed 42)
- **`val_dev`** (20% of official training graphs, deterministic seed 42)

The official `valid` split is reserved for final held-out validation of frozen models, preventing hyperparameter tuning leakage.
