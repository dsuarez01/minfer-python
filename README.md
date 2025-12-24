# minfer-python

Benchmarking inference on (decoder-only) LLM models.

Kernels written using CUDA as the underlying backend.

Currently the kernels are stubs, and they successfully compile as tested on my university cluster setup. 

Information you may find useful:

The environment variable `TORCH_EXTENSIONS_DIR` determines where the compiled kernels are stored, and `TORCH_CUDA_ARCH_LIST` limits which compute capabilities the kernels are compiled for.