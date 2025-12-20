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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# local disk for compile cache
LOCAL_DISK=/state/partition1/user/$USER
mkdir -p $LOCAL_DISK/torch_extensions

export TORCH_EXTENSIONS_DIR=$LOCAL_DISK/torch_extensions