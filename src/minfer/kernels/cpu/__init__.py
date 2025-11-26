from typing import Any, Callable
from torch.utils.cpp_extension import load
import os

_cpu_kernels = None

def _get_cpu_kernels():
    global _cpu_kernels
    if _cpu_kernels is None:
        source_dir = os.path.dirname(__file__)
        _cpu_kernels = load(
            name="cpu_kernels",
            sources=[
                # os.path.join(source_dir, "kernels.cu"),
                os.path.join(source_dir, "quants.cpp"),
                os.path.join(source_dir, "quants_impl.cpp"),
            ],
            extra_cflags=["-std=c++17", "-O3"],
            verbose=True,
        )
    return _cpu_kernels

cpu_kernels: Any = _get_cpu_kernels()
_dequant_row: Callable = cpu_kernels.dequant_row
_quant_row: Callable = cpu_kernels.quant_row
# rmsnorm: Callable = cpu_kernels.rmsnorm
# il_rope: Callable = cpu_kernels.il_rope
# neox_rope: Callable = cpu_kernels.neox_rope
# matmul: Callable = cpu_kernels.matmul
# embed: Callable = cpu_kernels.embed
# qkv: Callable = cpu_kernels.qkv
# flash_attn: Callable = cpu_kernels.flash_attn
# moe_scoring: Callable = cpu_kernels.moe_scoring
# ffn: Callable = cpu_kernels.ffn