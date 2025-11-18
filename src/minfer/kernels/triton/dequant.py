# TODO: complete me!
import triton
import triton.language as tl

from gguf import GGMLQuantizationType

@triton.jit
def _dequant_row(qtype) -> None:
    if qtype == GGMLQuantizationType.Q4_0:
        return _dequant_row_q4_0()
    # define other cases below

# example
@triton.jit
def _dequant_row_q4_0() -> None:
    pass