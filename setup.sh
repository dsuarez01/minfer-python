#!/bin/bash

## This is intended for an offline compute node scheduled in an interactive bash session
## Hence no sbatch here

## Notes about (preliminary) uv setup on this cluster:
## ensure ~/.local/bin is in PATH env (check ~/.bashrc), 
## otherwise uv isn't found

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

module load cuda/12.9
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

# local disk for compile cache
LOCAL_DISK=/state/partition1/user/$USER
mkdir -p $LOCAL_DISK/torch_extensions

export TORCH_EXTENSIONS_DIR=$LOCAL_DISK/torch_extensions
export TORCH_CUDA_ARCH_LIST="7.0" # NVIDIA V100

# this compiles the kernels as specified in setup.py
# (compiling may take a few minutes)
export MAX_JOBS=8
# (run uv sync --group dev --no-install-project on the login node first)
uv sync --group dev --no-build-isolation --verbose