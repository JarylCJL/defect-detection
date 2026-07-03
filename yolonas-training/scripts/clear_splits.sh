#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

rm -rf "${DATA_DIR}/train" "${DATA_DIR}/val" "${DATA_DIR}/test"
mkdir -p "${DATA_DIR}/train" "${DATA_DIR}/val" "${DATA_DIR}/test"

echo "Cleared and recreated: ${DATA_DIR}/train ${DATA_DIR}/val ${DATA_DIR}/test"
