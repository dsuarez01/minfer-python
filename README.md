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

The compute intensity of certain operations common to deep learning — e.g. the matrix multiplication — scales more rapidly than the amount of data moved. Since inference is typically a memory-bound task, this justifies a careful approach to the optimization of their kernel implementations, and thus the techniques used here to optimize their kernels are specific to the GPU device used. Hence, I optimized these kernels specifically for use on the `NVIDIA L40S GPU` (as the Hopper GPUs were in frequent use).

NVIDIA GPUs differ in supported instructions and hardware capabilities, which makes it quite difficult to judge whether you will observe the same performance on a different device. And writing hand-optimized kernels for other NVIDIA GPUs is currently beyond the scope of this project. However, I believe it is likely that these kernels will work well on one of the Ampere architecture GPUs, due to e.g. similar compute capabilities and opt-in limits on shared memory. *Regardless, if you are using a device with a different compute capability, change `setup.py` to reflect this.*

Note especially if you are testing this out / forking this project for local development on a cluster setup:

- The environment variable `TORCH_EXTENSIONS_DIR` determines where the compiled kernels are stored. You will want to set this directory to a local disk.
- In general, it is better not to build the project on compute nodes isolated from internet access, e.g. on a SLURM cluster setup. You would have to build the project without build isolation, meaning many slow transactions over the network filesystem if the cluster is configured to use Lustre (large fileblock sizes). In addition, since the project would be too slow to copy over to a local disk (env managers make many small file transactions), dependencies have to be fetched at runtime from the login node as well, so not just at compile-time. If your cluster uses this setup, consider moving over to an NFS/NFS4 cluster with fast scratch storage and smaller fileblock sizes, and additionally ensure that the compute nodes are connected to the internet. The project will then be fast to compile, and fast at runtime.

For more information regarding the installation, refer to `setup.py` and `setup.sh`. The setup script will have to be adjusted depending on your system configuration.

### Requirements:

- Python >= 3.12
- uv for env management
- At least 1 NVIDIA GPU with driver supporting CUDA 12.4+ (see [intro](#introduction))

### Quick Start:

To install the package in editable mode:

```bash
[DEBUG=1] uv sync --group dev # use testing suite
```

To verify that the setup works on your system, run the unit tests located at `tests/kernel.py` with Pytest:

```bash
pytest test/kernels.py
```

To benchmark the kernels against the corresponding Pytorch implementations:

```bash
python apps/benchmark.py
```

To profile kernels with NCU:

```bash
ncu --set full python apps/profile.py
```


## Results

The timing benchmarks measure performance relative to the Pytorch implementations, where we are essentially competing against highly optimized vendor libraries such as cuBLAS and/or CUTLASS where applicable. For these benchmarks, I do not have sudo access privileges on the cluster to be able to lock the GPU or memory clocks[^2]. This is only possible with NSight Compute in my case, and so we use it to profile the kernels.

### Matrix Multiplication (HGEMM)

All of the measurements reported are the median of trials via repeated testing.

| Kernel                      | Throughput (TFLOPS) | Pytorch HGEMM Throughput   | % PyTorch HGEMM Throughput   |  Speedup vs. Baseline        |
|-----------------------------|---------------------|----------------------------|------------------------------|------------------------------|
| Baseline (Basic Tiling)     | 19.97               | 165.05                     | **12.1%**                    |  1x                          |
| Unroll/vectorize shmem load | 120.77              | 162.97                     | **74.1%**                    |  6x                          |
| Swizzling                   | 155.74              | 167.13                     | **93.1%**                    |  7.8x                        |


## Remarks

The most substantial progress resulted from referencing NVIDIA's cuda-samples repo[^3] for their WMMA HGEMM implementation. Due to eventually wanting to understand the underlying functionality of the WMMA API, I eventually searched for an HGEMM optimization blogpost that made use of the lower-level MMA API[^1], and a related blog diving into SGEMM optimization[^2].

We start with the baseline implementation from the former blogpost[^1]. Some of the subsequent improvements will differ due to e.g. the introduction of an explicitly supported asynchronous memcpy from global to shared memory (starting with the Ampere architecture), differences in opt-in shared memory and supported matrix fragment sizes, register capacity on each streaming multiprocessor, etc. Of course, optimizations related to e.g. avoiding shared memory bank conflicts[^4] such as swizzling[^1], or vectorized memory transactions, etc. will look quite similar in terms of approach.




## Acknowledgments

All of the work here was developed on a university SLURM cluster. Thanks to MIT ORCD for access to substantial compute resources.

Thanks to Pytorch for the C++/CUDA extension[^5], a tool without which it would have been difficult to incorporate work in kernel optimization into an inference engine without writing an entirely separate tensor backend from scratch.


## References

[^1]: [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)
[^2]: [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/gemm-gpu)
[^3]: [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples)
[^4]: [Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks)
[^5]: [C++/CUDA Extensions in PyTorch](https://github.com/pytorch/extension-cpp)