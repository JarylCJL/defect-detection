#!/usr/bin/env python3
# =============================================================================
# Script: Cleanup Checkpoints
# What it does:
# - Removes RUN_* directories under a base directory that are NOT listed in a
#   checkpoint map file (first token per line is the run ID to keep).
#
# How to use:
# - Dry run (no deletes):
#   python scripts/cleanup_checkpoints.py --dry-run
# - Execute deletions:
#   python scripts/cleanup_checkpoints.py
# - Optional overrides:
#   python scripts/cleanup_checkpoints.py --base-dir PATH --map-file PATH
# =============================================================================
"""Remove checkpoint run folders not listed in a checkpoint map."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Iterable


def load_keep_set(map_path: str) -> set[str]:
    keep: set[str] = set()
    with open(map_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            run_id = line.split()[0]
            keep.add(run_id)
    return keep


def iter_run_dirs(base_dir: str) -> Iterable[str]:
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("RUN_"):
            yield name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove RUN_* directories not listed in the checkpoint map."
    )
    parser.add_argument(
        "--base-dir",
        default="../models/building_defect_yolo_nas_m",
        help="Directory containing RUN_* folders. Resolved relative to CWD (defaults to the shared top-level models/ dir).",
    )
    parser.add_argument(
        "--map-file",
        default="../models/checkpoint-map.txt",
        help="Text file listing runs to keep (first token per line).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    map_file = args.map_file

    if not os.path.isdir(base_dir):
        print(f"Base dir not found: {base_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(map_file):
        print(f"Map file not found: {map_file}", file=sys.stderr)
        return 2

    keep = load_keep_set(map_file)
    to_remove: list[str] = []
    to_keep: list[str] = []

    for name in sorted(iter_run_dirs(base_dir)):
        if name in keep:
            to_keep.append(name)
        else:
            to_remove.append(name)

    if to_remove:
        for name in to_remove:
            path = os.path.join(base_dir, name)
            if args.dry_run:
                print(f"DRY-RUN remove {path}")
            else:
                print(f"Removing {path}")
                shutil.rmtree(path)
    else:
        print("Nothing to remove.")

    print(f"Kept {len(to_keep)} run(s). Removed {len(to_remove)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
