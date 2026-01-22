# minfer-python

## Table of Contents

- [**Overview**](#overview)
    - [*Introduction*](#overview---introduction)
    - [*Requirements*](#overview---requirements)
    - [*Quick Start*](#overview---quick-start)
- [**Results**](#results)
- [**Remarks**](#remarks)
- [**Acknowledgments**](#acknowledgments)
- [**References**](#references)


## Overview:

### Introduction:

Benchmarking inference on (decoder-only) LLM models. Kernels are written using CUDA as the underlying backend.

For some operations, their compute intensity scales more rapidly than the amount of data moved. Since inference is typically a memory-bound task, this justifies a careful approach to the optimization of their kernel implementations, and some of the techniques used are specific to the GPU device used. Hence, I optimized these kernels specifically for use on the `NVIDIA L40S GPU` (as the Hopper GPUs were constantly in high demand). All of the work here was done on a university SLURM cluster: thanks to MIT ORCD for making this possible.

NVIDIA GPUs differ in various capabilities, both at the software and hardware level, that make it quite difficult to judge whether you will observe the same performance benefits on a different GPU. It is currently beyond the scope of this project to write hand-optimized kernels for other NVIDIA GPUs... However, I believe it is likely that these kernels will work well on one of the Ampere architecture GPUs (due to similar compute capabilities and opt-in limits on shared memory). *If you are using a device with a different compute capability, change `setup.py` to reflect this.*

Note especially if you are testing this out / forking this project for development on a cluster setup:

- The environment variable `TORCH_EXTENSIONS_DIR` determines where the compiled kernels are stored. You will want to set this directory to a local disk.
- You may observe slower build times depending on the fileblock size configured on your cluster system. Smaller fileblock sizes are better for this project: take note and switch away from e.g. a cluster configured with the Lustre filesystem, and instead to NFS/NFS4 or Scratch if possible.

For more information regarding the installation, refer to `setup.py` and `setup.sh`. The setup script will have to be tweaked depending on your system configuration.

### Requirements:

- Python >= 3.12
- >= 1 NVIDIA GPU (see [introduction](#overview---introduction))
- CUDA (used 12.4 on my system)
- uv for env management

### Quick Start:

To install the package in editable mode:

```bash
[DEBUG=1] uv sync --group dev # use testing suite
```

To verify that the setup works on your system, run the unit tests located at `tests/kernel.py` with Pytest:

```bash
pytest test/kernels.py
```

To benchmark against the corresponding Pytorch implementation:

```bash
python apps/benchmark.py
```

To profile kernels with NCU:

```bash
ncu --set full python apps/profile.py
```


## Results

The timing benchmarks measure performance relative to the Pytorch implementations, where we are essentially competing against highly optimized vendor libraries such as cuBLAS and/or CUTLASS. For these benchmarks, I do not have sudo access privileges on the cluster to be able to lock the GPU or memory clocks. This is only possible with NSight Compute in my case, and we use it to profile the kernels.

### Matrix Multiplication (HGEMM)

| Kernel       | Throughput (GFLOPS) | % PyTorch HGEMM Throughput |  Speedup vs. Baseline        |
|--------------|---------------------|----------------------------|------------------------------|
| Basic Tiling |                     |                            |                              |
| **PyTorch**  |                     | **100%**                   |                              |


## Remarks

We start with the same baseline implementation as [^1], but some of the subsequent improvements will differ due to e.g. the addition of an explicitly supported
asynchronous memcpy from global to shared memory, differences in shared memory and supported matrix fragment sizes, register capacity on each streaming multiprocessor, etc.


## Acknowledgments

Thanks to MIT ORCD for access to cluster resources.


## References

[^1]: [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html#the-memory-wall)
[^2]: [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/gemm-gpu)
[^3]: [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples)
[^4]: [Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks)
[^5]: [C++/CUDA Extensions in PyTorch](https://github.com/pytorch/extension-cpp)