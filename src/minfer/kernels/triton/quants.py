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
    qk, bsz = BL_Q4_0.qk, BL_Q4_0.bsz
    do, dsz = BL_Q4_0.do,  BL_Q4_0.dsz
    qo, qsz = BL_Q4_0.qo, BL_Q4_0.qsz

    assert k % qk == 0, "Q4_0"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = tl.load(x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    qi = oi%(qk//2)
    qs = (oi//(qk//2))*4

    q = tl.load(x_ptr+bo+qo+qi)

    x = ((q>>qs)&0x0F).to(tl.int32)-8

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*x)

@triton.jit
def __dequant_row_q4_1(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q4_1.qk, BL_Q4_1.bsz
    do,dsz = BL_Q4_1.do, BL_Q4_1.dsz
    mo,msz = BL_Q4_1.mo, BL_Q4_1.msz
    qo,qsz = BL_Q4_1.qo, BL_Q4_1.qsz

    assert k % qk == 0, "Q4_1"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = tl.load(x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    m_ptr = tl.load(x_ptr+bo+mo).to(tl.pointer_type(tl.float16))
    
    d = tl.load(d_ptr).to(tl.float32)
    m = tl.load(m_ptr).to(tl.float32)

    qi = oi%(qk//2)
    qs = (oi//(qk//2))*4

    q = tl.load(x_ptr+bo+qo+qi)

    x = ((q>>qs)&0x0F).to(tl.int32)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*x+m)

@triton.jit
def __dequant_row_q5_0(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q5_0.qk, BL_Q5_0.bsz
    do, dsz = BL_Q5_0.do, BL_Q5_0.dsz
    qho, qhsz = BL_Q5_0.qho, BL_Q5_0.qhsz
    qo, qsz = BL_Q5_0.qo, BL_Q5_0.qsz

    assert k % qk == 0, "Q5_0"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)
    
    qi = qhi = (oi%(qk//2))
    qs = (oi//(qk//2))*4
    qhs = qi+(oi//(qk//2))*16

    q = tl.load(x_ptr+bo+qo+qi)
    qh = tl.load(x_ptr+bo+qho+qhi)

    qh = (((qh>>qhs)<<4)&0x10)
    x = (((q>>qs)&0x0F)|qh).to(tl.int32)-16

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*x)

@triton.jit
def __dequant_row_q5_1(x_ptr, y_ptr, k) -> None:
    qtype = GGMLQuantizationType.Q5_1
    qk, bsz = BL_Q5_1.qk, BL_Q5_1.bsz
    do, dsz = BL_Q5_1.do, BL_Q5_1.dsz
    mo, msz = BL_Q5_1.mo, BL_Q5_1.msz
    qho, qhsz = BL_Q5_1.qho, BL_Q5_1.qhsz
    qo, qsz = BL_Q5_1.qo, BL_Q5_1.qsz

    assert k % qk == 0, "Q5_1"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    m_ptr = (x_ptr+bo+mo).to(tl.pointer_type(tl.float16))

    d = tl.load(d_ptr).to(tl.float32)
    m = tl.load(m_ptr).to(tl.float32)

    qi = qhi = (oi%(qk//2))
    qs = (oi//(qk//2))*4
    qhs = qi+(oi//(qk//2))*16

    q = tl.load(x_ptr+bo+qo+qi)
    qh = tl.load(x_ptr+bo+qho+qhi)

    qh = (((qh>>qhs)<<4)&0x10)
    x = (((q>>qs)&0x0F)|qh).to(tl.int32)
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)

    tl.store(y_ptr+ybo+oi, d*x+m)

@triton.jit
def __dequant_row_q8_0(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q8_0.qk, BL_Q8_0.bsz
    do, dsz = BL_Q8_0.do, BL_Q8_0.dsz
    qo, qsz = BL_Q8_0.qo, BL_Q8_0.qsz

    assert k % qk == 0, "Q8_0"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)

    x = tl.load((x_ptr+bo+qo+oi).to(tl.pointer_type(tl.int8)))

    tl.store(y_ptr+ybo+oi, x*d)

# "microscaling" quant

@triton.jit
def __dequant_row_mxfp4(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_MXFP4.qk, BL_MXFP4.bsz
    eo, esz = BL_MXFP4.eo, BL_MXFP4.esz
    qo, qsz = BL_MXFP4.qo, BL_MXFP4.qsz

    assert k % qk == 0, "MXFP4"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d = tl.load(x_ptr+bo+eo)
    d = tl.where(d<2,0x00200000<<d,((d.to(tl.uint32)-1)<<23))
    d = d.to(tl.float32, bitcast=True)

    qi = oi%(qk//2)
    q = tl.load(x_ptr+bo+qo+qi)
    qs = (oi//(qk//2))*4
    q = KVALUES_MXFP4[(q>>qs)&0x0F].to(tl.int8)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*q)

# 2-6 bit quantization in super-blocks

@triton.jit
def __dequant_row_q2_K(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q2_K.qk, BL_Q2_K.bsz
    sco, scsz = BL_Q2_K.sco, BL_Q2_K.scsz
    qo, qsz = BL_Q2_K.qo, BL_Q2_K.qsz
    do, dsz = BL_Q2_K.do, BL_Q2_K.dsz
    dmo, dmsz = BL_Q2_K.dmo, BL_Q2_K.dmsz

    assert k % qk == 0, "Q2_K"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)
    sci = oi//scsz

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    dmin_ptr = (x_ptr+bo+dmo).to(tl.pointer_type(tl.float16))
    sc_ptr = (x_ptr+bo+sco+sci)

    d = tl.load(d_ptr).to(tl.float32)
    dmin = tl.load(dmin_ptr).to(tl.float32)
    sc = tl.load(sc_ptr)

    dl = d*(sc&0xF)
    ml = dmin*(sc>>4)

    qi = oi//4
    qs = (oi%4)*2
    q = tl.load(x_ptr+bo+qo+qi)
    q = ((q>>qs)&0x03).to(tl.int8)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*q-ml)

@triton.jit
def __dequant_row_q3_K(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q3_K.qk, BL_Q3_K.bsz
    hmo, hmsz = BL_Q3_K.hmo, BL_Q3_K.hmsz
    qo, qsz = BL_Q3_K.qo, BL_Q3_K.qsz
    sco, scsz = BL_Q3_K.sco, BL_Q3_K.scsz
    do, dsz = BL_Q3_K.do, BL_Q3_K.dsz

    assert k % qk == 0, "Q3_K"
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

    qi = oi//4
    hmi = qi%(qsz//2)
    biti = oi//32
    m = 1 << biti
    qs = (oi%4)*2

    q = tl.load(x_ptr+bo+qo+qi)
    hm = tl.load(x_ptr+bo+hmo+hmi)

    q = (((q>>qs)&0x03)-tl.where((hm&m)!=0,0,4)).to(tl.int8)
    
    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*q)

@triton.jit
def __dequant_row_q4_K(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q4_K.qk, BL_Q4_K.bsz
    do, dsz = BL_Q4_K.do, BL_Q4_K.dsz
    dmo, dmsz = BL_Q4_K.dmo, BL_Q4_K.dmsz
    sco, scsz = BL_Q4_K.sco, BL_Q4_K.scsz
    qo, qsz = BL_Q4_K.qo, BL_Q4_K.qsz

    assert k % qk == 0, "Q4_K"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    # scale and min logic
    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    dmin_ptr = (x_ptr+bo+dmo).to(tl.pointer_type(tl.float16))

    d = tl.load(d_ptr).to(tl.float32)
    dmin = tl.load(dmin_ptr).to(tl.float32)

    sb = tl.load(x_ptr+bo+sco+tl.expand_dims(tl.arange(0,scsz),axis=0))
    gpi = oi//(qk//8)
    low = gpi < 4

    d1 = tl.gather(sb, tl.where(low, gpi, gpi+4), axis=1)
    d2 = tl.gather(sb, tl.where(low, gpi+4, gpi-4), axis=1) # the True case is a dummy entry
    m1 = tl.gather(sb, gpi+4, axis=1)
    m2 = tl.gather(sb, tl.where(low, gpi+4, gpi), axis=1) # the True case is a dummy entry

    sc = tl.where(low, d1 & 63, (d1 & 0xF) | ((d2 >> 6) << 4))
    m = tl.where(low, m1 & 63, (m1 >> 4) | ((m2 >> 6) << 4))

    dl = d*sc
    ml = dmin*m

    qi = (oi//2)
    qs = (oi%2)*4
    q = tl.load(x_ptr+bo+qo+qi)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*((q>>qs)&0xF)-ml)

@triton.jit
def __dequant_row_q5_K(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q5_K.qk, BL_Q5_K.bsz
    do, dsz = BL_Q5_K.do, BL_Q5_K.dsz
    dmo, dmsz = BL_Q5_K.dmo, BL_Q5_K.dmsz
    sco, scsz = BL_Q5_K.sco, BL_Q5_K.scsz
    qho, qhsz = BL_Q5_K.qho, BL_Q5_K.qhsz
    qo, qsz = BL_Q5_K.qo, BL_Q5_K.qsz

    assert k % qk == 0, "Q5_K"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    # scale and min logic

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    dmin_ptr = (x_ptr+bo+dmo).to(tl.pointer_type(tl.float16))

    d = tl.load(d_ptr).to(tl.float32)
    dmin = tl.load(dmin_ptr).to(tl.float32)

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
    qi = oi//2
    qhi = oi%(qk//8)
    qs = (oi%2)*4

    q = tl.load(x_ptr+bo+qo+qi)
    qh = tl.load(x_ptr+bo+qho+qhi)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, dl*((q>>qs)&0xF)+(tl.where((qh&(1<<ui))!=0,16,0))-ml)

@triton.jit
def __dequant_row_q6_K(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q6_K.qk, BL_Q6_K.bsz
    qo, qsz = BL_Q6_K.qo, BL_Q6_K.qsz
    qho, qhsz = BL_Q6_K.qho, BL_Q6_K.qhsz
    sco, scsz = BL_Q6_K.sco, BL_Q6_K.scsz
    do, dsz = BL_Q6_K.do, BL_Q6_K.dsz

    assert k % qk == 0, "Q6_K"
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

    qi = oi//2
    qs = (oi%2)*4

    q = tl.load(x_ptr+bo+qo+qi)
    qh = tl.load(x_ptr+bo+qho+qhi)

    q6 = (((q>>qs)&0x0F)|(((qh>>qhs)&3)<<4)).to(tl.int8)-32

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*sc*q6)

# ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

@triton.jit
def __dequant_row_tq1_0(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_TQ1_0.qk, BL_TQ1_0.bsz
    qo, qsz = BL_TQ1_0.qo, BL_TQ1_0.qsz
    qho, qhsz = BL_TQ1_0.qho, BL_TQ1_0.qhsz
    do, dsz = BL_TQ1_0.do, BL_TQ1_0.dsz

    assert k % qk == 0, "TQ1_0"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    POW3 = tl.constexpr([1,3,9,27,81])

    qi = oi//5
    q = tl.load(x_ptr+bo+qo+qi)
    q = q*POW3[oi%5]
    q = ((q.to(tl.uint16)*3)>>8).to(tl.int16)

    qhi = tl.where(oi>=252, (oi-252)//4, 0)
    qh = tl.load(x_ptr+bo+qho+qhi)
    qh = qh * POW3[tl.where(oi>=252,(oi-252)%4,0)]
    qh = ((qh.to(tl.uint16)*3)>>8).to(tl.int16)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*(tl.where(oi<252, q, qh)-1))

@triton.jit
def __dequant_row_tq2_0(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_TQ2_0.qk, BL_TQ2_0.bsz
    qo, qsz = BL_TQ2_0.qo, BL_TQ2_0.qsz
    do, dsz = BL_TQ2_0.do, BL_TQ2_0.dsz

    assert k % qk == 0, "TQ2_0"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    qi = oi//4
    qs = (oi%4)*2

    q = tl.load(x_ptr+bo+qo+qi)
    q = ((q>>qs)&3).to(tl.int8)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*(q-1))

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
    qk, bsz = BL_IQ4_NL.qk, BL_IQ4_NL.bsz
    do, dsz = BL_IQ4_NL.do, BL_IQ4_NL.dsz
    qo, qsz = BL_IQ4_NL.qo, BL_IQ4_NL.qsz

    assert k % qk == 0, "IQ4_NL"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)

    d_ptr = (x_ptr+bo+do).to(tl.pointer_type(tl.float16))
    d = tl.load(d_ptr).to(tl.float32)

    qi = oi%(qk//2)
    qs = (oi//(qk//2))*4

    q = tl.load(x_ptr+bo+qo+qi)

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*KVALUES_IQ4_NL[(q>>qs)&0xF])

@triton.jit
def __dequant_row_iq4_xs(x_ptr, y_ptr, k) -> None:
    pass

# ==================== Q8_K ======================

@triton.jit
def __dequant_row_q8_K(x_ptr, y_ptr, k) -> None:
    qk, bsz = BL_Q8_K.qk, BL_Q8_K.bsz
    do, dsz = BL_Q8_K.do, BL_Q8_K.dsz
    qo, qsz = BL_Q8_K.qo, BL_Q8_K.qsz

    assert k % qk == 0, "Q8_K"
    nb = k // qk

    bo = tl.expand_dims(tl.arange(0,nb)*bsz, axis=1)
    oi = tl.expand_dims(tl.arange(0,qk), axis=0)
    
    d = tl.load((x_ptr+bo+do).to(tl.pointer_type(tl.float32)))
    q = tl.load((x_ptr+bo+qo+oi))

    ybo = tl.expand_dims(tl.arange(0,nb)*qk, axis=1)
    tl.store(y_ptr+ybo+oi, d*q)