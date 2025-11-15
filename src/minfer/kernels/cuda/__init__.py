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
_dequant = cuda_kernels._dequant
rmsnorm = cuda_kernels.rmsnorm
il_rope = cuda_kernels.il_rope
neox_rope = cuda_kernels.neox_rope
matmul = cuda_kernels.matmul
embed = cuda_kernels.embed
qkv = cuda_kernels.qkv
flash_attn = cuda_kernels.flash_attn
moe_scoring = cuda_kernels.moe_scoring
ffn = cuda_kernels.ffn

__all__ = ["_dequant", "rmsnorm", "il_rope", "neox_rope", "matmul", "embed", "qkv", "flash_attn", "moe_scoring", "ffn"]