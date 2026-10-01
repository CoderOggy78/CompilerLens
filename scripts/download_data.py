#!/usr/bin/env python3
"""Automated data acquisition script for CompilerLens (TpuGraphs Tile XLA Collection)."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from compilerlens.data import (
    ARCHIVE_METADATA,
    check_disk_space,
    download_file,
    safe_extract_tar,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract TpuGraphs archives.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to store downloaded tar archives and extracted files.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip downloading large training archive.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing archives without re-downloading.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_name": "TpuGraphs (Tile Collection)",
        "source": "https://github.com/google-research-datasets/tpu_graphs",
        "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "archives": {},
        "file_counts": {},
    }

    archives_to_process = ["npz_tile_xla_valid.tar", "npz_tile_xla_test.tar"]
    if not args.skip_train:
        archives_to_process.insert(0, "npz_tile_xla_train.tar")

    for archive_name in archives_to_process:
        meta = ARCHIVE_METADATA[archive_name]
        tar_dest = raw_dir / archive_name

        if not args.verify_only:
            logger.info(f"Downloading {archive_name}...")
            download_file(
                primary_url=meta["primary_url"],
                fallback_url=meta["fallback_url"],
                dest_path=tar_dest,
                expected_bytes=meta["expected_bytes"],
                expected_md5=meta["md5"],
            )

            logger.info(f"Extracting {archive_name}...")
            safe_extract_tar(tar_dest, raw_dir)

        manifest["archives"][archive_name] = {
            "status": "verified" if tar_dest.exists() else "missing",
            "bytes": tar_dest.stat().st_size if tar_dest.exists() else 0,
            "md5": meta["md5"],
        }

    # Count extracted files
    for split in ["train", "valid", "test"]:
        split_dir = raw_dir / "npz" / "tile" / "xla" / split
        count = len(list(split_dir.glob("*.npz"))) if split_dir.exists() else 0
        manifest["file_counts"][split] = count
        logger.info(f"Split '{split}': {count} .npz computation graphs found.")

    manifest_path = raw_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote dataset manifest to {manifest_path}")


if __name__ == "__main__":
    main()
