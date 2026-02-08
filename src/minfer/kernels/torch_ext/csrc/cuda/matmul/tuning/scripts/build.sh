#!/bin/bash

# finds repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done

MATMUL="$REPO_ROOT/src/minfer/kernels/torch_ext/csrc/cuda/matmul"

nvcc -O3 \
     -std=c++17 \
     --use_fast_math \
     -gencode=arch=compute_89,code=sm_89 \
     --ptxas-options=-v \
     -I"$MATMUL" \
     tune.cu \
     -o tune