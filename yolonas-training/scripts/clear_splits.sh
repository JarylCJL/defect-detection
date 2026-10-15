#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: Clear the train/val/test split
#
# What it does:
# - Deletes yolonas-training/training-data/{train,val,test} and recreates them
#   empty, along with the split_manifest.csv and split_report.txt that
#   described them. classes.txt is kept.
# - Note: it also deletes the .gitkeep files tracked in git, so git shows them
#   as deleted; restore them with `git checkout -- training-data` or rebuild.
# - Usually not needed: build_training_data.py and
#   build_training_data_class_balanced.py clear the old split themselves (and
#   keep the .gitkeep files) before writing a new one.
#
# How to use (from anywhere; paths are anchored to this script's location):
#   bash scripts/clear_splits.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/training-data"

rm -rf "${DATA_DIR}/train" "${DATA_DIR}/val" "${DATA_DIR}/test"
mkdir -p "${DATA_DIR}/train" "${DATA_DIR}/val" "${DATA_DIR}/test"
# The manifest and report describe the split just deleted, so they go too.
# classes.txt is kept: it is tracked in git and the next build rewrites it if needed.
rm -f "${DATA_DIR}/split_manifest.csv" "${DATA_DIR}/split_report.txt"

echo "Cleared and recreated: ${DATA_DIR}/train ${DATA_DIR}/val ${DATA_DIR}/test"
