"""Data loading, validation, dataset acquisition, and preprocessing for CompilerLens."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Checksums and metadata for authoritative upstream TpuGraphs tile archives
ARCHIVE_METADATA = {
    "npz_tile_xla_train.tar": {
        "primary_url": "https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_train.tar",
        "fallback_url": "http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_train.tar",
        "expected_bytes": 158965760,
        "md5": "528f9bcfd86581662f238a9cfb8613e7",
    },
    "npz_tile_xla_valid.tar": {
        "primary_url": "https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_valid.tar",
        "fallback_url": "http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_valid.tar",
        "expected_bytes": 15923200,
        "md5": "eae09512dd0da8d6a6dddff4b922b278",
    },
    "npz_tile_xla_test.tar": {
        "primary_url": "https://storage.googleapis.com/download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_test.tar",
        "fallback_url": "http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_test.tar",
        "expected_bytes": 15912960,
        "md5": "3ad9d6c51fbd51937e55dffbe12d9856",
    },
}


@dataclass
class GraphData:
    """Represents a single computation graph and its compiler candidate configurations."""

    graph_id: str
    split: str
    num_nodes: int
    num_edges: int
    num_configs: int
    node_feat: np.ndarray  # Shape: (N, 140), float32
    node_opcode: np.ndarray  # Shape: (N,), int32
    edge_index: np.ndarray  # Shape: (E, 2), int64
    config_feat: np.ndarray  # Shape: (C, 24), float32
    config_runtime: Optional[np.ndarray] = None  # Shape: (C,), int64
    config_runtime_normalizers: Optional[np.ndarray] = None  # Shape: (C,), int64
    normalized_runtimes: Optional[np.ndarray] = None  # Shape: (C,), float64
    log_normalized_runtimes: Optional[np.ndarray] = None  # Shape: (C,), float32

    def to_torch(
        self,
        node_feat_tensor: Optional[torch.Tensor] = None,
        config_feat_tensor: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Convert graph structures and candidate features to PyTorch tensors."""
        nf = (
            node_feat_tensor
            if node_feat_tensor is not None
            else torch.from_numpy(self.node_feat).float()
        )
        cf = (
            config_feat_tensor
            if config_feat_tensor is not None
            else torch.from_numpy(self.config_feat).float()
        )
        return {
            "graph_id": self.graph_id,
            "split": self.split,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_configs": self.num_configs,
            "node_feat": nf.to(device),
            "node_opcode": torch.from_numpy(self.node_opcode).long().to(device),
            "edge_index": torch.from_numpy(self.edge_index).long().to(device),
            "config_feat": cf.to(device),
            "log_normalized_runtimes": (
                torch.from_numpy(self.log_normalized_runtimes).float().to(device)
                if self.log_normalized_runtimes is not None
                else None
            ),
        }


def compute_file_md5(filepath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_disk_space(target_dir: Union[str, Path], required_bytes: int) -> bool:
    """Verify sufficient disk space is available."""
    stat = shutil.disk_usage(target_dir)
    return stat.free >= required_bytes


def download_file(
    primary_url: str,
    fallback_url: Optional[str],
    dest_path: Union[str, Path],
    expected_bytes: Optional[int] = None,
    expected_md5: Optional[str] = None,
    max_retries: int = 3,
) -> Path:
    """Download a remote file with retries, HTTPS fallback, and checksum verification."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if existing file is already valid
    if dest_path.exists():
        if expected_bytes is not None and dest_path.stat().st_size == expected_bytes:
            if expected_md5 is not None:
                actual_md5 = compute_file_md5(dest_path)
                if actual_md5 == expected_md5:
                    logger.info(f"Using cached file: {dest_path}")
                    return dest_path
            else:
                return dest_path

    # Check disk space
    if expected_bytes is not None:
        if not check_disk_space(dest_path.parent, expected_bytes * 2):
            raise IOError(
                f"Insufficient disk space in {dest_path.parent} for {expected_bytes} bytes."
            )

    urls_to_try = [primary_url]
    if fallback_url and fallback_url != primary_url:
        urls_to_try.append(fallback_url)

    last_error = None
    for url in urls_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Downloading {url} (attempt {attempt}/{max_retries})...")
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "CompilerLens/0.1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as response, open(
                    dest_path, "wb"
                ) as out_file:
                    shutil.copyfileobj(response, out_file)

                # Verify size
                if expected_bytes is not None:
                    actual_size = dest_path.stat().st_size
                    if actual_size != expected_bytes:
                        raise ValueError(
                            f"Downloaded file size {actual_size} != expected {expected_bytes}"
                        )

                # Verify checksum
                if expected_md5 is not None:
                    actual_md5 = compute_file_md5(dest_path)
                    if actual_md5 != expected_md5:
                        raise ValueError(
                            f"MD5 mismatch: {actual_md5} != {expected_md5}"
                        )

                logger.info(f"Successfully downloaded {dest_path}")
                return dest_path
            except Exception as e:
                last_error = e
                logger.warning(f"Download attempt {attempt} for {url} failed: {e}")
                if dest_path.exists():
                    dest_path.unlink()

    raise RuntimeError(
        f"Failed to download from all candidate URLs: {urls_to_try}. Error: {last_error}"
    )


def safe_extract_tar(tar_path: Union[str, Path], extract_dir: Union[str, Path]) -> None:
    """Safely extract a tar archive preventing directory traversal vulnerabilities."""
    tar_path = Path(tar_path)
    extract_dir = Path(extract_dir).resolve()

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            member_path = (extract_dir / member.name).resolve()
            if not str(member_path).startswith(str(extract_dir)):
                raise SecurityError(
                    f"Attempted Path Traversal in Tar File: {member.name}"
                )
        tar.extractall(path=extract_dir)


def load_npz_graph(
    filepath: Union[str, Path],
    split: str = "unknown",
    allow_pickle: bool = False,
    validate: bool = True,
) -> GraphData:
    """Load and validate an NPZ file representing a TPU computation graph and candidate configurations.

    Args:
        filepath: Path to the .npz file.
        split: The dataset partition name ('train', 'valid', 'test', 'train_dev', 'val_dev').
        allow_pickle: Strict security policy: disallows pickle loading.
        validate: If True, performs structural and numerical sanity checks.

    Returns:
        GraphData instance.
    """
    filepath = Path(filepath)
    graph_id = filepath.stem

    with np.load(filepath, allow_pickle=allow_pickle) as data:
        required_keys = {"node_feat", "node_opcode", "edge_index", "config_feat"}
        missing = required_keys - set(data.files)
        if missing:
            raise ValueError(f"File {filepath} is missing required fields: {missing}")

        node_feat = data["node_feat"].astype(np.float32)
        node_opcode = data["node_opcode"].astype(np.int32)
        edge_index = data["edge_index"].astype(np.int64)
        config_feat = data["config_feat"].astype(np.float32)

        config_runtime = (
            data["config_runtime"].astype(np.int64)
            if "config_runtime" in data.files
            else None
        )
        config_runtime_normalizers = (
            data["config_runtime_normalizers"].astype(np.int64)
            if "config_runtime_normalizers" in data.files
            else None
        )

    num_nodes = node_feat.shape[0]
    num_edges = edge_index.shape[0]
    num_configs = config_feat.shape[0]

    # Structural Validation
    if validate:
        if node_opcode.shape[0] != num_nodes:
            raise ValueError(
                f"Opcode length {node_opcode.shape[0]} != node count {num_nodes} in {filepath}"
            )

        if node_feat.ndim != 2 or node_feat.shape[1] != 140:
            raise ValueError(
                f"node_feat shape {node_feat.shape} must be (N, 140) in {filepath}"
            )

        if config_feat.ndim != 2 or config_feat.shape[1] != 24:
            raise ValueError(
                f"config_feat shape {config_feat.shape} must be (C, 24) in {filepath}"
            )

        if num_edges > 0:
            if edge_index.ndim != 2 or edge_index.shape[1] != 2:
                raise ValueError(
                    f"edge_index shape {edge_index.shape} must be (E, 2) in {filepath}"
                )
            min_idx = np.min(edge_index)
            max_idx = np.max(edge_index)
            if min_idx < 0 or max_idx >= num_nodes:
                raise ValueError(
                    f"Invalid edge indices in {filepath}: range [{min_idx}, {max_idx}], num_nodes={num_nodes}"
                )

        if not np.all(np.isfinite(node_feat)):
            raise ValueError(f"Non-finite values detected in node_feat in {filepath}")

        if not np.all(np.isfinite(config_feat)):
            raise ValueError(f"Non-finite values detected in config_feat in {filepath}")

    # Compute normalized and log-normalized runtimes if available
    normalized_runtimes = None
    log_normalized_runtimes = None

    if config_runtime is not None and config_runtime_normalizers is not None:
        if config_runtime.shape[0] != num_configs:
            raise ValueError(
                f"config_runtime len {config_runtime.shape[0]} != config_feat count {num_configs}"
            )
        # Check if labeled (train/valid) or unlabeled (test with all zeros)
        has_labels = (
            np.max(config_runtime) > 0 and np.max(config_runtime_normalizers) > 0
        )
        if has_labels:
            if validate:
                if np.any(config_runtime <= 0):
                    raise ValueError(
                        f"Non-positive runtimes found in labeled graph {filepath}"
                    )
                if np.any(config_runtime_normalizers <= 0):
                    raise ValueError(
                        f"Non-positive normalizers found in labeled graph {filepath}"
                    )

            runtimes_f64 = config_runtime.astype(np.float64)
            normalizers_f64 = np.maximum(
                config_runtime_normalizers.astype(np.float64), 1.0
            )
            normalized_runtimes = runtimes_f64 / normalizers_f64
            # Avoid log(0) or negative values
            log_normalized_runtimes = np.log(
                np.maximum(normalized_runtimes, 1e-6)
            ).astype(np.float32)

    return GraphData(
        graph_id=graph_id,
        split=split,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_configs=num_configs,
        node_feat=node_feat,
        node_opcode=node_opcode,
        edge_index=edge_index,
        config_feat=config_feat,
        config_runtime=config_runtime,
        config_runtime_normalizers=config_runtime_normalizers,
        normalized_runtimes=normalized_runtimes,
        log_normalized_runtimes=log_normalized_runtimes,
    )


def create_graph_disjoint_splits(
    file_list: Sequence[Union[str, Path]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Partition a list of graph files into deterministic, strictly graph-disjoint subsets.

    Args:
        file_list: Collection of file paths to partition.
        val_ratio: Fraction reserved for validation.
        seed: Random seed for deterministic reproducibility.

    Returns:
        (train_files, val_files) lists of Paths.
    """
    paths = sorted([Path(p) for p in file_list])
    rng = np.random.RandomState(seed)
    indices = np.arange(len(paths))
    rng.shuffle(indices)

    num_val = int(len(paths) * val_ratio)
    val_idx = set(indices[:num_val])

    train_files = [paths[i] for i in range(len(paths)) if i not in val_idx]
    val_files = [paths[i] for i in range(len(paths)) if i in val_idx]

    # Verify disjointness
    train_stems = {p.stem for p in train_files}
    val_stems = {p.stem for p in val_files}
    intersection = train_stems.intersection(val_stems)
    assert not intersection, f"Graph leakage detected across splits: {intersection}"

    return train_files, val_files


class TpuTileDataset(Dataset):
    """PyTorch Dataset for TpuGraphs Tile XLA computation graphs.

    Supports lazy loading from disk, in-memory caching, candidate subsampling during training,
    and full candidate candidate retention during evaluation.
    """

    def __init__(
        self,
        file_paths: Sequence[Union[str, Path]],
        split_name: str,
        preprocessor: Optional[Any] = None,
        max_configs: Optional[int] = None,
        cache_in_memory: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.file_paths = [Path(p) for p in sorted(file_paths)]
        self.split_name = split_name
        self.preprocessor = preprocessor
        self.max_configs = max_configs
        self.cache_in_memory = cache_in_memory
        self.seed = seed
        self._cache: Dict[int, GraphData] = {}

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> GraphData:
        if self.cache_in_memory and idx in self._cache:
            graph_data = self._cache[idx]
        else:
            filepath = self.file_paths[idx]
            graph_data = load_npz_graph(filepath, split=self.split_name)
            if self.cache_in_memory:
                self._cache[idx] = graph_data

        # If training and candidate count exceeds max_configs, sample a subset deterministically
        if (
            self.max_configs is not None
            and graph_data.num_configs > self.max_configs
            and self.split_name.startswith("train")
        ):
            # Sample configs
            rng = np.random.RandomState(self.seed + idx)
            sel_idx = rng.choice(graph_data.num_configs, size=self.max_configs, replace=False)
            sel_idx.sort()

            sub_config_feat = graph_data.config_feat[sel_idx]
            sub_config_runtime = (
                graph_data.config_runtime[sel_idx]
                if graph_data.config_runtime is not None
                else None
            )
            sub_normalizers = (
                graph_data.config_runtime_normalizers[sel_idx]
                if graph_data.config_runtime_normalizers is not None
                else None
            )
            sub_norm_runtimes = (
                graph_data.normalized_runtimes[sel_idx]
                if graph_data.normalized_runtimes is not None
                else None
            )
            sub_log_runtimes = (
                graph_data.log_normalized_runtimes[sel_idx]
                if graph_data.log_normalized_runtimes is not None
                else None
            )

            graph_data = GraphData(
                graph_id=graph_data.graph_id,
                split=graph_data.split,
                num_nodes=graph_data.num_nodes,
                num_edges=graph_data.num_edges,
                num_configs=len(sel_idx),
                node_feat=graph_data.node_feat,
                node_opcode=graph_data.node_opcode,
                edge_index=graph_data.edge_index,
                config_feat=sub_config_feat,
                config_runtime=sub_config_runtime,
                config_runtime_normalizers=sub_normalizers,
                normalized_runtimes=sub_norm_runtimes,
                log_normalized_runtimes=sub_log_runtimes,
            )

        return graph_data


def collate_tile_graphs(batch: Sequence[GraphData]) -> Dict[str, Any]:
    """Collates a list of GraphData objects into a batched representation for message passing.

    Nodes from separate graphs are concatenated into a contiguous block, with edge indices
    shifted by the running node offset. Forward edges and reverse edges are explicitly cataloged.
    """
    total_nodes = sum(g.num_nodes for g in batch)
    total_edges = sum(g.num_edges for g in batch)

    batch_node_feat = []
    batch_node_opcode = []
    batch_graph_indices = []
    batch_fwd_edges = []
    batch_rev_edges = []
    node_offset = 0

    graph_configs = []
    graph_targets = []
    graph_normalized_runtimes = []
    graph_ids = []

    for g_idx, g in enumerate(batch):
        batch_node_feat.append(torch.from_numpy(g.node_feat).float())
        batch_node_opcode.append(torch.from_numpy(g.node_opcode).long())
        batch_graph_indices.append(
            torch.full((g.num_nodes,), g_idx, dtype=torch.long)
        )

        if g.num_edges > 0:
            edges = torch.from_numpy(g.edge_index).long() + node_offset
            batch_fwd_edges.append(edges)
            # Reverse edge: swap src and dst
            batch_rev_edges.append(edges[:, [1, 0]])

        graph_configs.append(torch.from_numpy(g.config_feat).float())
        graph_ids.append(g.graph_id)

        if g.log_normalized_runtimes is not None:
            graph_targets.append(
                torch.from_numpy(g.log_normalized_runtimes).float()
            )
        if g.normalized_runtimes is not None:
            graph_normalized_runtimes.append(
                torch.from_numpy(g.normalized_runtimes).double()
            )

        node_offset += g.num_nodes

    node_feat = torch.cat(batch_node_feat, dim=0)
    node_opcode = torch.cat(batch_node_opcode, dim=0)
    graph_node_idx = torch.cat(batch_graph_indices, dim=0)

    if batch_fwd_edges:
        fwd_edge_index = torch.cat(batch_fwd_edges, dim=0).t().contiguous()  # (2, E)
        rev_edge_index = torch.cat(batch_rev_edges, dim=0).t().contiguous()  # (2, E)
    else:
        fwd_edge_index = torch.empty((2, 0), dtype=torch.long)
        rev_edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "num_graphs": len(batch),
        "graph_ids": graph_ids,
        "node_feat": node_feat,
        "node_opcode": node_opcode,
        "graph_node_idx": graph_node_idx,
        "fwd_edge_index": fwd_edge_index,
        "rev_edge_index": rev_edge_index,
        "graph_configs": graph_configs,  # List of tensors of shape (C_g, 24)
        "graph_targets": graph_targets,  # List of tensors of shape (C_g,)
        "graph_normalized_runtimes": graph_normalized_runtimes,
    }
