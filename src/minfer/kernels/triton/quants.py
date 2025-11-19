# TODO: complete me!
import triton
import triton.language as tl

from .const import *
from gguf import GGMLQuantizationType

# NOTE: make sure to use newest version of GGUF from Github repo for MXFP4 support
@triton.jit
def _dequant_row(qtype : GGMLQuantizationType, x_ptr, y_ptr, b, k):
    
    row_idx = tl.program_id(0)
    
    x_row_ptr = x_ptr + row_idx * b # NOTE: this should be byte offset?
    y_row_ptr = y_ptr + row_idx * k # NOTE: this should be element offset?

    if qtype == GGMLQuantizationType.Q4_0: __dequant_row_q4_0(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q4_1: __dequant_row_q4_1(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q5_0: __dequant_row_q5_0(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q5_1: __dequant_row_q5_1(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q8_0: __dequant_row_q8_0(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.MXFP4: __dequant_row_mxfp4(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q2_K: __dequant_row_q2_K(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q3_K: __dequant_row_q3_K(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q4_K: __dequant_row_q4_K(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q5_K: __dequant_row_q5_K(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q6_K: __dequant_row_q6_K(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.TQ1_0: __dequant_row_tq1_0(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.TQ2_0: __dequant_row_tq2_0(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ2_XXS: __dequant_row_iq2_xxs(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ2_XS: __dequant_row_iq2_xs(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ2_S: __dequant_row_iq2_s(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ3_XXS: __dequant_row_iq3_xxs(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ3_S: __dequant_row_iq3_s(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ1_S: __dequant_row_iq1_s(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ1_M: __dequant_row_iq1_m(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ4_NL: __dequant_row_iq4_nl(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.IQ4_XS: __dequant_row_iq4_xs(x_row_ptr, y_row_ptr, k)
    elif qtype == GGMLQuantizationType.Q8_K: __dequant_row_q8_K(x_row_ptr, y_row_ptr, k)
    else: raise ValueError(f"Unsupported GGMLQuantizationType {qtype.name}")


# legacy / simple quants

@triton.jit
def __dequant_row_q4_0(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q4_1(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q5_0(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q5_1(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q8_0(x_ptr, y_ptr, k) -> None:
    pass

# "microscaling" quant

@triton.jit
def __dequant_row_mxfp4(x_ptr, y_ptr, k) -> None:
    pass


# 2-6 bit quantization in super-blocks

@triton.jit
def __dequant_row_q2_K(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q3_K(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q4_K(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q5_K(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_q6_K(x_ptr, y_ptr, k) -> None:
    pass

# ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

@triton.jit
def __dequant_row_tq1_0(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_tq2_0(x_ptr, y_ptr, k) -> None:
    pass

# ====================== "True" 2-bit (de)-quantization

@triton.jit
def __dequant_row_iq2_xxs(x_ptr, y_ptr, k) -> None:
    pass

# ====================== 2.3125 bpw (de)-quantization

@triton.jit
def __dequant_row_iq2_xs(x_ptr, y_ptr, k) -> None:
    pass

# ====================== 2.5625 bpw (de)-quantization

@triton.jit
def __dequant_row_iq2_s(x_ptr, y_ptr, k) -> None:
    pass

# ====================== 3.0625 bpw (de)-quantization

@triton.jit
def __dequant_row_iq3_xxs(x_ptr, y_ptr, k) -> None:
    pass

# ====================== 3.3125 bpw (de)-quantization

@triton.jit
def __dequant_row_iq3_s(x_ptr, y_ptr, k) -> None:
    pass

# ====================== 1.5625 bpw (de)-quantization

@triton.jit
def __dequant_row_iq1_s(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_iq1_m(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_iq4_nl(x_ptr, y_ptr, k) -> None:
    pass

@triton.jit
def __dequant_row_iq4_xs(x_ptr, y_ptr, k) -> None:
    pass

# ==================== Q8_K ======================

@triton.jit
def __dequant_row_q8_K(x_ptr, y_ptr, k) -> None:
    pass