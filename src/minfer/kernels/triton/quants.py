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
    do, dsz = bl["d"]
    qo, qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    d_ptr = tl.load(x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    j = tl.expand_dims(tl.arange(0,qsz), axis=0)
    p = tl.load(x_ptr+bo+qo+j)
    x0 = (p&0x0F)-8
    x1 = (p>>4)-8

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+j, x0*d)
    tl.store(y_ptr+ybo+j+qsz, x1*d)

@triton.jit
def __dequant_row_q4_1(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q4_1
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do,dsz = bl["d"]
    mo,msz = bl["m"]
    qo,qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    d_ptr = tl.load(x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    m_ptr = tl.load(x_ptr+bo+mo).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)
    m = tl.load(m_ptr).to(tl.float32)

    j = tl.expand_dims(tl.arange(0,qsz), axis=0)
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)

    p = tl.load(x_ptr+bo+qo+j)
    x0 = (p&0x0F)
    x1 = (p>>4)

    tl.store(y_ptr+ybo+j, x0*d+m)
    tl.store(y_ptr+ybo+j+qsz, x1*d+m)

@triton.jit
def __dequant_row_q5_0(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q5_0
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, dsz = bl["d"]
    qho, qhsz = bl["qh"]
    qlo, qlsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    qh_ptr = (x_ptr+bo+qho).to(tl.pointer_type(tl.uint32))

    d = tl.load(d_ptr).to(tl.float32)
    qh = tl.load(qh_ptr)

    j = tl.expand_dims(tl.arange(0,qlsz), axis=0)
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    
    xh_0 = ((qh >> j) << 4) & 0x10
    xh_1 = (qh >> (j+12)) & 0x10
    
    p = tl.load(x_ptr+bo+qlo+j) 
    x0 = ((p & 0x0F) | xh_0) - 16
    x1 = ((p >> 4) | xh_1) - 16

    tl.store(y_ptr+ybo+j, x0*d)
    tl.store(y_ptr+ybo+j+qlsz, x1*d)


@triton.jit
def __dequant_row_q5_1(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q5_1
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, dsz = bl["d"]
    mo, msz = bl["m"]
    qho, qhsz = bl["qh"]
    qlo, qlsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    m_ptr = (x_ptr+bo+mo).to(tl.pointer_type(tl.float16))
    qh_ptr = (x_ptr+bo+qho).to(tl.pointer_type(tl.uint32))

    d = tl.load(d_ptr).to(tl.float32)
    m = tl.load(m_ptr).to(tl.float32)
    qh = tl.load(qh_ptr)

    j = tl.expand_dims(tl.arange(0,qlsz), axis=0)
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    
    xh_0 = ((qh >> j) << 4) & 0x10
    xh_1 = (qh >> (j+12)) & 0x10
    
    p = tl.load(x_ptr+bo+qlo+j)
    x0 = ((p & 0x0F) | xh_0)
    x1 = ((p >> 4) | xh_1)

    tl.store(y_ptr+ybo+j, x0*d+m)
    tl.store(y_ptr+ybo+j+qlsz, x1*d+m)

@triton.jit
def __dequant_row_q8_0(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q8_0
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, dsz = bl["d"]
    qo, qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    j = tl.expand_dims(tl.arange(0,qsz), axis=0)
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)

    x = tl.load((x_ptr+bo+qo+j).to(tl.pointer_type(tl.int8)))

    tl.store(y_ptr+ybo+j, x*d)


# "microscaling" quant

@triton.jit
def __dequant_row_mxfp4(x_ptr, y_ptr, k) -> None:
    pass


# 2-6 bit quantization in super-blocks

@triton.jit
def __dequant_row_q2_K(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q2_K
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    sco, scsz = bl["scales"]
    qso, qsz = bl["qs"]
    do, dsz = bl["d"]
    dmino, dminsz = bl["dmin"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)
    sci = oi//scsz

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    min_ptr = (x_ptr+bo+dmino).to(tl.pointer_type(tl.float16))
    sc_ptr = (x_ptr+bo+sco+sci)

    d = tl.load(d_ptr).to(tl.float32)
    min = tl.load(min_ptr).to(tl.float32)
    sc = tl.load(sc_ptr)

    dl = d*(sc&0xF)
    ml = min*(sc>>4)

    ne = 4            # 4 vals of 2 bits each
    qi = oi//ne
    shift = (oi%ne)*2

    q = tl.load(x_ptr+bo+qso+qi)
    q = ((q>>shift)&0x03).to(tl.int8)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*q-ml)

@triton.jit
def __dequant_row_q3_K(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q3_K
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    hmasko, hmasksz = bl["hmask"]
    qso, qsz = bl["qs"]
    sco, scsz = bl["scales"]
    do, dsz = bl["d"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    ## (very odd logic to get the scales)
    # load first 12 bytes of each block
    sb = tl.load(x_ptr+bo+sco+tl.expand_dims(tl.arange(0,scsz),axis=0))
    
    # bytes packed into 3 uint32s
    aux0 = sb[:,0].to(tl.uint32) | (sb[:,1].to(tl.uint32)<<8) | (sb[:,2].to(tl.uint32)<<16) | (sb[:,3].to(tl.uint32)<<24)
    aux1 = sb[:,4].to(tl.uint32) | (sb[:,5].to(tl.uint32)<<8) | (sb[:,6].to(tl.uint32)<<16) | (sb[:,7].to(tl.uint32)<<24)
    aux2 = sb[:,8].to(tl.uint32) | (sb[:,9].to(tl.uint32)<<8) | (sb[:,10].to(tl.uint32)<<16) | (sb[:,11].to(tl.uint32)<<24)

    kmask1, kmask2 = 0x03030303, 0x0f0f0f0f
    tmp = aux2

    # bit shuffling
    aux2_new = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    aux3_new = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux0_new = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    aux1_new = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4)

    scales_list = []
    for aux in [aux0_new, aux1_new, aux2_new, aux3_new]:
        for s in [0,8,16,24]:
            scales_list.append(tl.expand_dims(((aux>>s) & 0xFF).to(tl.int8), axis=1))
    
    scales = scales_list[0]
    for s in scales_list[1:]:
        scales = tl.cat(scales,s)

    sc = tl.gather(scales, oi//scsz, axis=1)
    ## (end of very odd logic to get the scales)

    dl = d*(sc-32)

    ne = 4                          # 4 vals of 2 bits each
    qi = oi//ne
    hi = qi%(qsz//2)
    biti = oi//32
    m = 1 << biti
    shift = (oi%ne)*2

    q = tl.load(x_ptr+bo+qso+qi)
    h = tl.load(x_ptr+bo+hmasko+hi)


    q = ((q>>shift)&0x03-tl.where((h & m)!=0,0,4)).to(tl.int8)
    
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*q)

@triton.jit
def __dequant_row_q4_K(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q4_K
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, dsz = bl["d"]
    dmino, dminsz = bl["dmin"]
    sco, scsz = bl["scales"]
    qso, qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    # scale and min logic

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d_min_ptr = (x_ptr+bo+dmino).to(tl.pointer_type(tl.float16))

    d = tl.load(d_ptr).to(tl.float32)
    dmin = tl.load(d_min_ptr).to(tl.float32)

    sb = tl.load(x_ptr+bo+sco+tl.expand_dims(tl.arange(0,scsz),axis=0))
    gpi = oi//(qk//8)
    low = gpi < 4

    d1 = tl.gather(sb, tl.where(low, gpi, gpi+4), axis=1)
    d2 = tl.gather(sb, tl.where(low, gpi+4, gpi-4), axis=1) # the True case is a dummy entry
    m1 = tl.gather(sb, gpi+4, axis=1)
    m2 = tl.gather(sb, tl.where(low, gpi+4, gpi), axis=1) # the True case is a dummy entry

    sc = tl.where(low, d1 & 63, (d1 & 0xF) | ((d2 >> 6) << 4))
    m = tl.where(low, m1 & 63, (m1 >> 4) | ((m2 >> 6) << 4))

    dl = d * sc
    ml = dmin * m

    # q

    qi = (oi//2)
    shift = (oi%2)*4

    q = tl.load(x_ptr+bo+qso+qi)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*((q>>shift)&0xF)-ml)

@triton.jit
def __dequant_row_q5_K(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q5_K
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    do, dsz = bl["d"]
    dmino, dminsz = bl["dmin"]
    sco, scsz = bl["scales"]
    qho, qhsz = bl["qh"]
    qso, qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    # scale and min logic

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d_min_ptr = (x_ptr+bo+dmino).to(tl.pointer_type(tl.float16))

    d = tl.load(d_ptr).to(tl.float32)
    dmin = tl.load(d_min_ptr).to(tl.float32)

    sb = tl.load(x_ptr+bo+sco+tl.expand_dims(tl.arange(0,scsz),axis=0))
    gpi = oi//(qk//8)
    low = gpi < 4

    d1 = tl.gather(sb, tl.where(low, gpi, gpi+4), axis=1)
    d2 = tl.gather(sb, tl.where(low, gpi+4, gpi-4), axis=1) # the True case is a dummy entry
    m1 = tl.gather(sb, gpi+4, axis=1)
    m2 = tl.gather(sb, tl.where(low, gpi+4, gpi), axis=1) # the True case is a dummy entry

    sc = tl.where(low, d1 & 63, (d1 & 0xF) | ((d2 >> 6) << 4))
    m = tl.where(low, m1 & 63, (m1 >> 4) | ((m2 >> 6) << 4))

    dl = d * sc
    ml = dmin * m

    ## ql, qh, u1 and u2 merged into u
    ui = oi//(qk//8)
    qli = oi//2
    qhi = oi%(qk//8)
    shift = (oi%2)*4

    ql = tl.load(x_ptr+bo+qso+qli)
    qh = tl.load(x_ptr+bo+qho+qhi)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*((ql>>shift)&0xF)+(tl.where((qh&(1<<ui))!=0,16,0))-ml)


@triton.jit
def __dequant_row_q6_K(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q6_K
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]
    
    qlo, qlsz = bl["ql"]
    qho, qhsz = bl["qh"]
    sco, scsz = bl["scales"]
    do, dsz = bl["d"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    sci = oi//(qk//16)
    sc_ptr = (x_ptr+bo+sco+sci).to(tl.pointer_type(tl.int8))
    sc = tl.load(sc_ptr)

    qhi = oi%(qk//8)
    qhs = oi//(qk//8)

    qli = oi // 2
    qls = (oi%2)*4

    ql = tl.load(x_ptr+bo+qlo+qli)
    qh = tl.load(x_ptr+bo+qho+qhi)

    q6 = (((ql >> qls) & 0x0F) | (((qh >> qhs) & 3) << 4)).to(tl.int8) - 32

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*sc*q6)

# ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

@triton.jit
def __dequant_row_tq1_0(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.TQ1_0
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]

    qso, qsz = bl["qs"]
    qho, qhsz = bl["qh"]
    do, dsz = bl["d"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    POW3 = tl.constexpr([1,3,9,27,81])

    qi = oi//5
    q = tl.load(x_ptr+bo+qso+qi)
    q = q * POW3[oi%5]
    q = ((q.to(tl.uint16)*3)>>8).to(tl.int16)

    qhi = tl.where(oi>=252, (oi-252)//4, 0)
    qh = tl.load(x_ptr+bo+qho+qhi)
    
    qh = (qh>>qhs)&0x3

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*(tl.where(oi<252, q, qh)-1))




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
    qtype = GGMLQuantizationType.IQ4_NL
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]

    do, dsz = bl["d"]
    qso, qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    qi = oi%(qk//2)
    shift = (oi//(qk//2))*4

    qs = tl.load(x_ptr+bo+qso+qi)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*KVALUES_IQ4_NL[(qs>>shift) & 0xF])

@triton.jit
def __dequant_row_iq4_xs(x_ptr, y_ptr, k) -> None:
    pass

# ==================== Q8_K ======================

@triton.jit
def __dequant_row_q8_K(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q8_K
    qk, bsz = GGML_QUANT_SIZES[qtype]
    bl = BLOCK_LAYOUTS[qtype]

    do, dsz = bl["d"]
    qso, qsz = bl["qs"]

    assert k % qk == 0, qtype.name
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)
    
    d = tl.load((x_ptr+bo+do).to(tl.pointer_type(tl.float32)))
    q = tl.load((x_ptr+bo+qso+oi))

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*q)
