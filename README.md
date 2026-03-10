# minfer-python

## Table of Contents

- [**Overview**](#overview)
    - [*Introduction*](#overview---introduction)
    - [*Requirements*](#overview---requirements)
    - [*Quick Start*](#overview---quick-start)
- [**Results**](#results)
    - [*Benchmark Notes*](#results---benchmark-notes)
    - [*HGEMM*](#results---hgemm)
- [**Remarks**](#remarks)
- [**Acknowledgments**](#acknowledgments)
- [**References**](#references)


## Overview

### Introduction:

The goal of this project is to write an efficient inference engine for (decoder-only) LLM models using custom-written CUDA C++ kernels. Exceptions are currently made for e.g. top-k, reduce-all kernels which are commonly used in MoE layers, though they may be implemented in the near future. All kernels are written using CUDA as the underlying backend, with the current focus being on migrating to use of CUTLASS for the remainder of the project.

Which kernels are worth optimizing? For certain operations in deep learning — for example, matrix multiplication — arithmetic computation asymptotically scales more rapidly than the amount of data moved. Since inference is typically a memory-bound task, this justifies a careful approach to their optimization. Some optimizations turn out to be rather specific to the hardware (the GPU) used. Only the L40S and A100 GPUs are available to me on a university SLURM cluster. Currently, an inference-optimized GPU like the L40S makes sense for this task as of now, though I may switch over to the A100 if implementing backward passes for any of the kernels seems worthwhile.

NVIDIA GPUs differ in supported instructions and hardware capabilities. I would recommended at least selecting a GPU within the same architecture family (e.g. RTX 4000 series) to see similar results. Note that you will need to change `setup.py` to reflect e.g. use of an Ampere GPU.

Note especially if you are testing this out / forking this project for local development on a cluster setup:

- The environment variable `TORCH_EXTENSIONS_DIR` determines where the compiled kernels are stored. You will want to set this directory to a local disk.
- In general, it is better not to build the project on compute nodes isolated from internet access, e.g. on some SLURM cluster setups. You would have to build the project without build isolation, meaning many slow transactions over the network filesystem if the cluster is configured to use Lustre (large fileblock sizes). In addition, since the project would be too slow to copy over to a local disk (env managers make many small file transactions), dependencies have to be fetched at runtime from the login node as well. 
- If you are on a cluster using this setup, consider moving over to an NFS/NFS4 cluster with fast scratch storage and smaller fileblock sizes, and additionally ensure that the compute nodes are connected to the internet. The project should then be fast to compile, and fast at runtime.

For more information regarding the installation, refer to `setup.py` and `setup.sh`. The setup script will have to be adjusted depending on your system configuration.

### Requirements:

- Python >= 3.12
- uv for env management
- At least 1 NVIDIA GPU with driver supporting CUDA 12.4+

### Quick Start:

To install the package in editable mode:

```bash
[DEBUG=1] uv sync --group dev # use testing suite
```

NOTE: Only the `rmsnorm`, `rope`, `embed`, and `gemm` kernels are currently supported. To verify that the setup works on your system, run the unit tests located at `tests/kernels.py` with Pytest:

```bash
pytest test/kernels.py -k "rmsnorm or rope or embed or gemm"
```

To benchmark the kernels against the corresponding Pytorch implementations:

```bash
python apps/benchmark.py --kernel {rmsnorm,rope,embed,gemm} --which {ref,minfer}
```

To profile kernels with NCU:

```bash
ncu --set full python apps/profile.py --kernel {rmsnorm,rope,embed,gemm} --which {ref,minfer}
```


## Results

### Benchmark Notes:

Note here that we only report results for kernels that are "worth" heavily optimizing as described in the [introduction](#introduction). As of now, these kernels include the GEMM, Flash Attention, MoE Scoring and FFN kernels.

The timing benchmarks measure performance of the custom kernels relative to kernels that Pytorch calls. If you profile the Pytorch implementations, you will notice that these kernels come from highly optimized and proprietary vendor libraries (cuBLAS, CUTLASS). We try and reverse engineer their implementation techniques via reading the SASS (assembly) instructions in e.g. Nsight Compute.

For these benchmarks, I do not have sudo access privileges on the cluster to be able to lock the SM or memory clocks on the GPU.[^2] This is only possible with NSight Compute in my case, and so we use it to profile the kernels.

### HGEMM:

All of the measurements reported are the median of trials via repeated testing. Refer to the [PyTorch benchmarking utility](https://github.com/pytorch/pytorch/tree/v2.10.0/torch/utils/benchmark/utils) implementation for more details.

| Size (M=K=N) | alpha/beta | minfer (µs) | cuBLAS (µs) | Winner | Speedup vs. Reference |
|------|-----------|------------|------------|--------|-----------------------|
| 512 | 1.0, 0.0 | 9.3 | 14.4 | minfer | 55% |
| 512 | 1.0, 1.0 | 10.1 | 10.5 | minfer | 4% |
| 512 | 2.0, 3.0 | 10.1 | 10.1 | tie | 0% |
| 512 | 0.5, 2.0 | 10.1 | 10.1 | tie | 0% |
| 1024 | 1.0, 0.0 | 16.4 | 21.8 | minfer | 33% |
| 1024 | 1.0, 1.0 | 17.1 | 24.7 | minfer | 44% |
| 1024 | 2.0, 3.0 | 17.2 | 24.6 | minfer | 43% |
| 1024 | 0.5, 2.0 | 17.2 | 24.7 | minfer | 44% |
| 2048 | 1.0, 0.0 | 72.0 | 70.2 | cuBLAS | -3% |
| 2048 | 1.0, 1.0 | 73.2 | 75.9 | minfer | 4% |
| 2048 | 2.0, 3.0 | 74.2 | 75.8 | minfer | 2% |
| 2048 | 0.5, 2.0 | 73.8 | 75.9 | minfer | 3% |
| 4096 | 1.0, 0.0 | 605.3 | 733.7 | minfer | 21% |
| 4096 | 1.0, 1.0 | 658.2 | 821.3 | minfer | 20% |
| 4096 | 2.0, 3.0 | 662.5 | 809.6 | minfer | 18% |
| 4096 | 0.5, 2.0 | 656.0 | 838.6 | minfer | 22% |
| 8192 | 1.0, 0.0 | 5693.0 | 6260.4 | minfer | 10% |
| 8192 | 1.0, 1.0 | 5744.4 | 6529.0 | minfer | 12% |
| 8192 | 2.0, 3.0 | 5787.5 | 6530.9 | minfer | 11% |
| 8192 | 0.5, 2.0 | 5778.8 | 6506.9 | minfer | 11% |


## Remarks

Initial progress resulted from referencing NVIDIA's cuda-samples repo[^3] for their WMMA GEMM implementation, and learning from the pattern implemented there. Due to eventually wanting to understand the underlying functionality of the WMMA API, I eventually searched for an HGEMM optimization blogpost[^1] that made use of the MMA API (register fragment level), and a related blog covering SGEMM optimization.[^2]

I found a blog on HGEMM optimization to be helpful as a first introduction to (Turing-era) optimization techniques.[^1] Subsequent improvements I made differ due to the introduction of asynchronous global memory to shared memory feature starting with the Ampere architecture (this allows for more extensive software pipelining in a way that synchronous loading does not allow for), differences in opt-in shared memory and supported matrix fragment size instructions, register capacity on each SM, etc. The article also suggested optimizations mitigating shared memory bank conflicts[^4], e.g. swizzling. We also vectorize the shared memory loads requests.

Persistent kernels / Stream-K optimization will be implemented to handle wave quantization (load imbalance in terms of how work units are assigned to SMs).[^6] CUTLASS provides nice abstractions to deal with this and allows for us to e.g. to be able to try different swizzling patterns or add support for more kinds of dtypes, mixed precision, etc. And so remaining kernels will be migrated over to CUTLASS for the remainder of the project, though implementing the HGEMM from scratch proved to be a good introduction to kernel optimization on NVIDIA GPUs.


## Acknowledgments

All of the work here was developed on a university SLURM cluster. Thanks to MIT ORCD for access to substantial compute resources.

Thanks to Pytorch for the C++/CUDA extension[^5], a tool without which it would have been difficult to incorporate work in kernel optimization into an inference engine without writing an entirely separate tensor backend from scratch.


## References

[^1]: [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html)
[^2]: [Advanced Matrix Multiplication Optimization on NVIDIA GPUs](https://salykova.github.io/gemm-gpu)
[^3]: [NVIDIA cuda-samples](https://github.com/NVIDIA/cuda-samples)
[^4]: [Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks)
[^5]: [C++/CUDA Extensions in PyTorch](https://github.com/pytorch/extension-cpp)
[^6]: [CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)