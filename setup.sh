#!/bin/bash

## This is intended for an offline compute node scheduled in an interactive bash session
## Hence no sbatch here

## Notes about (preliminary) uv setup on this cluster:
## ensure ~/.local/bin is in PATH env (check ~/.bashrc), 
## otherwise uv isn't found

export CC=$(which gcc)
export CXX=$(which g++)

# in case system-wide python deafult version too old
# (remove as needed)
if [ ! -d ".venv" ]; then
    uv venv --python 3.12
fi
source .venv/bin/activate

module load cuda/12.4.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export OMP_NUM_THREADS=$(($(nproc)/2))

# local disk for compile cache
LOCAL_DISK=/tmp/$USER
mkdir -p $LOCAL_DISK/torch_extensions

export TORCH_EXTENSIONS_DIR=$LOCAL_DISK/torch_extensions