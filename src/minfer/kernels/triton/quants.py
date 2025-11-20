# TODO: complete me!
import triton
import triton.language as tl

from .const import *
from gguf import GGMLQuantizationType, GGML_QUANT_SIZES, QK_K

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
    qtype = GGMLQuantizationType.Q4_0
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, ds = bl["d"]
    qo, qs = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = (tl.arange(0,nb)*bsz)[:, None]
    d_ptr = tl.load(x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    j = (tl.arange(0,qs))[None, :]
    p = tl.load(x_ptr+bo+qo+j)
    x0 = (p&0x0F)-8
    x1 = (p>>4)-8

    ybo = (tl.arange(0,nb)*qk)[:, None]
    tl.store(y_ptr+ybo+j, x0*d)
    tl.store(y_ptr+ybo+j+qs, x1*d)

@triton.jit
def __dequant_row_q4_1(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q4_1
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do,ds = bl["d"]
    mo,ms = bl["m"]
    qo,qs = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = (tl.arange(0,nb)*bsz)[:, None]
    d_ptr = tl.load(x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    m_ptr = tl.load(x_ptr+bo+mo).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)
    m = tl.load(m_ptr).to(tl.float32)

    j = (tl.arange(0,qs))[None, :]
    ybo = (tl.arange(0,nb)*qk)[:, None]

    p = tl.load(x_ptr+bo+qo+j)
    x0 = (p&0x0F)
    x1 = (p>>4)

    tl.store(y_ptr+ybo+j, x0*d+m)
    tl.store(y_ptr+ybo+j+qs, x1*d+m)

@triton.jit
def __dequant_row_q5_0(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q5_0
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, ds = bl["d"]
    qho, qhs = bl["qh"]
    qlo, qls = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = (tl.arange(0,nb)*bsz)[:, None]
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    qh_ptr = (x_ptr+bo+qho).to(tl.pointer_type(tl.uint32))

    d = tl.load(d_ptr).to(tl.float32)
    qh = tl.load(qh_ptr)

    j = (tl.arange(0,qls))[None, :]
    ybo = (tl.arange(0,nb)*qk)[:, None]
    
    xh_0 = ((qh >> j) << 4) & 0x10
    xh_1 = (qh >> (j+12)) & 0x10
    
    p = tl.load(x_ptr+bo+qlo+j) 
    x0 = ((p & 0x0F) | xh_0) - 16
    x1 = ((p >> 4) | xh_1) - 16

    tl.store(y_ptr+ybo+j, x0*d)
    tl.store(y_ptr+ybo+j+qls, x1*d)


@triton.jit
def __dequant_row_q5_1(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q5_1
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, ds = bl["d"]
    mo, ms = bl["m"]
    qho, qhs = bl["qh"]
    qlo, qls = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = (tl.arange(0,nb)*bsz)[:, None]
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    m_ptr = (x_ptr+bo+mo).to(tl.pointer_type(tl.float16))
    qh_ptr = (x_ptr+bo+qho).to(tl.pointer_type(tl.uint32))

    d = tl.load(d_ptr).to(tl.float32)
    m = tl.load(m_ptr).to(tl.float32)
    qh = tl.load(qh_ptr)

    j = (tl.arange(0,qls))[None, :]
    ybo = (tl.arange(0,nb)*qk)[:, None]
    
    xh_0 = ((qh >> j) << 4) & 0x10
    xh_1 = (qh >> (j+12)) & 0x10
    
    p = tl.load(x_ptr+bo+qlo+j)
    x0 = ((p & 0x0F) | xh_0)
    x1 = ((p >> 4) | xh_1)

    tl.store(y_ptr+ybo+j, x0*d+m)
    tl.store(y_ptr+ybo+j+qls, x1*d+m)

@triton.jit
def __dequant_row_q8_0(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q8_0
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, ds = bl["d"]
    qo, qs = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = (tl.arange(0,nb)*bsz)[:,None]
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    j = tl.arange(0,qs)[None, :]
    ybo = (tl.arange(0,nb)*qk)[:,None]

    x = tl.load((x_ptr+bo+qo+j).to(tl.pointer_type(tl.int8)))

    tl.store(y_ptr+ybo+j, x*d)


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