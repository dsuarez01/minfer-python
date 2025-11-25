from typing import Any, Callable
from torch.utils.cpp_extension import load
import os

_cuda_kernels = None

def _get_cuda_kernels():
    global _cuda_kernels
    if _cuda_kernels is None:
        source_dir = os.path.dirname(__file__)
        _cuda_kernels = load(
            name="cuda_kernels",
            sources=[
                os.path.join(source_dir, "kernels.cu"),
                # os.path.join(source_dir, "quants.cu"),
            ],
            extra_cuda_cflags=["-O3"],
            verbose=True,
        )
    return _cuda_kernels

cuda_kernels: Any = _get_cuda_kernels()
_dequant_row: Callable = cuda_kernels._dequant_row
rmsnorm: Callable = cuda_kernels.rmsnorm
il_rope: Callable = cuda_kernels.il_rope
neox_rope: Callable = cuda_kernels.neox_rope
matmul: Callable = cuda_kernels.matmul
embed: Callable = cuda_kernels.embed
qkv: Callable = cuda_kernels.qkv
flash_attn: Callable = cuda_kernels.flash_attn
moe_scoring: Callable = cuda_kernels.moe_scoring
ffn: Callable = cuda_kernels.ffn