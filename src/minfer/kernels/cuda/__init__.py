from typing import Any
from torch.utils.cpp_extension import load
import os

_cuda_kernels = None

def _get_cuda_kernels():
    global _cuda_kernels
    if _cuda_kernels is None:
        source_dir = os.path.dirname(__file__)
        _cuda_kernels = load(
            name="kernels",
            sources=[os.path.join(source_dir, "kernels.cu")],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    return _cuda_kernels

cuda_kernels: Any = _get_cuda_kernels()
embed = cuda_kernels.embed
rmsnorm = cuda_kernels.rmsnorm
matmul = cuda_kernels.matmul
qkv = cuda_kernels.qkv
flash_attn = cuda_kernels.flash_attn
moe_scores = cuda_kernels.moe_scores
moe_experts = cuda_kernels.moe_experts

__all__ = ["embed", "rmsnorm", "matmul", "qkv", "flash_attn", "moe_scores", "moe_experts"]