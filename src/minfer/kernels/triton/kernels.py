import triton
import triton.language as tl
from gguf import GGMLQuantizationType

@triton.jit
def _dequant(
    quant_ptr, out_ptr, lut_ptr,
    M: tl.constexpr, N: tl.constexpr,
    block_size: tl.constexpr, n_bits: tl.constexpr,
    packed_size: tl.constexpr, scale_size: tl.constexpr,
    has_zp: tl.constexpr, out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    
    offs_m = pid_m*BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n*BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m<M)[:, None] & (offs_n<N)[None, :]
    
    blk_row = offs_m // block_size
    blk_col = offs_n // block_size
    n_blocks_row = (N+block_size-1) // block_size
    blk_idx = blk_row*n_blocks_row + blk_col
    
    elem_row = offs_m%block_size
    elem_col = offs_n%block_size
    elem_idx = elem_row*block_size + elem_col
    
    byte_offs = elem_idx*n_bits // 8
    bit_offs = (elem_idx*n_bits) % 8
    block_stride = packed_size+scale_size
    packed_offs = blk_idx*block_stride + byte_offs
    
    packed = tl.load(quant_ptr+packed_offs, mask=mask, other=0)
    code = (packed>>bit_offs) & ((1<<n_bits) - 1) # type:ignore
    
    scale_offs = blk_idx*block_stride + packed_size
    scale = tl.load(quant_ptr+scale_offs, mask=mask, other=1.0).to(out_dtype)
    zp = tl.load(quant_ptr+scale_offs+2, mask=mask, other=0.0).to(out_dtype) if has_zp else 0.0
    
    val = tl.load(lut_ptr+code, mask=mask)*scale + zp
    
    out_offs = offs_m[:, None]*N + offs_n[None, :]
    tl.store(out_ptr+out_offs, val, mask=mask)

# attn norm weights always stored in F32
@triton.jit
def rmsnorm(
    in_ptr,
    out_ptr,
    w_ptr,
    dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    sum_sq = 0.0
    for i in range(0, dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        x = tl.load(in_ptr + pid*dim + offs, mask=offs<dim) # essentially: load head?
        sum_sq += tl.sum(x*x)
    
    scale = tl.rsqrt(eps+sum_sq/dim)

    for i in range(0, dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < dim
        x = tl.load(in_ptr + pid*dim + offs, mask=mask)
        w = tl.load(w_ptr + pid*dim + offs, mask=mask)
        tl.store(out_ptr + pid*dim + offs, x*w*scale, mask=mask)

@triton.jit
def il_rope():
    pass

@triton.jit
def neox_rope():
    pass

@triton.jit
def matmul(
    act_ptr, quant_ptr, out_ptr, lut_ptr,
    batch_sz: tl.constexpr, seq_len: tl.constexpr,
    d_in: tl.constexpr, d_out: tl.constexpr,
    block_size: tl.constexpr, n_bits: tl.constexpr,
    packed_sz: tl.constexpr, scale_sz: tl.constexpr,
    has_zp: tl.constexpr, out_dtype: tl.constexpr,
    TILE_SZ: tl.constexpr,
):
    """ (ceil(batch_sz*seq_len/TILE_SZ), ceil(d_out/TILE_SZ), ceil(d_in/TILE_SZ)) """
    tile_batch, tile_out, tile_in = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    weight_tile = tl.zeros((TILE_SZ, TILE_SZ), dtype=out_dtype)
    
    weight_row = tile_out*TILE_SZ
    # weight_col = tile_in*TILE_SZ
    n_blocks_row = (d_in+block_size-1) // block_size
    block_stride = packed_sz+scale_sz
    weight_offs = weight_row*n_blocks_row*block_stride
    
    _dequant(
        quant_ptr+weight_offs, weight_tile, lut_ptr,
        M=TILE_SZ, N=TILE_SZ,
        block_size=block_size, n_bits=n_bits,
        packed_sz=packed_sz, scale_sz=scale_sz,
        has_zp=has_zp, out_dtype=out_dtype,
        BLOCK_M=TILE_SZ, BLOCK_N=TILE_SZ
    )
    
    offs_batch = tile_batch*TILE_SZ + tl.arange(0, TILE_SZ)
    offs_in = tile_in*TILE_SZ + tl.arange(0, TILE_SZ)
    act_mask = (offs_batch < batch_sz*seq_len)[:, None] & (offs_in<d_in)[None, :]
    act_tile = tl.load(act_ptr + offs_batch[:, None]*d_in + offs_in[None, :], mask=act_mask, other=0.0)
    
    result = tl.dot(act_tile, tl.trans(weight_tile))
    
    offs_out = tile_out*TILE_SZ + tl.arange(0, TILE_SZ)
    out_mask = (offs_batch < batch_sz*seq_len)[:, None] & (offs_out<d_out)[None, :]
    tl.atomic_add(out_ptr + offs_batch[:, None]*d_out + offs_out[None, :], result, mask=out_mask)

# TODO: complete
@triton.jit
def embed(
    token_ids_ptr, quant_ptr, out_ptr, lut_ptr,
    batch_sz: tl.constexpr, seq_len: tl.constexpr, hidden_dim: tl.constexpr,
    block_size: tl.constexpr, n_bits: tl.constexpr,
    packed_sz: tl.constexpr, scale_sz: tl.constexpr,
    has_zp: tl.constexpr, out_dtype: tl.constexpr,
):
    """ (batch_sz*seq_len, ceil(hidden_dim/64)) """
    pid_token = tl.program_id(0)
    
    batch_idx = pid_token // seq_len
    seq_idx = pid_token % seq_len
    token_id = tl.load(token_ids_ptr + batch_idx*seq_len + seq_idx)
    
    n_blocks_row = (hidden_dim+block_size-1) // block_size
    block_stride = packed_sz+scale_sz
    row_offs = token_id*n_blocks_row*block_stride
    out_offs = (batch_idx*seq_len + seq_idx)*hidden_dim
    
    _dequant(
        quant_ptr+row_offs, out_ptr+out_offs, lut_ptr,
        M=1, N=hidden_dim,
        block_size=block_size, n_bits=n_bits,
        packed_sz=packed_sz, scale_sz=scale_sz,
        has_zp=has_zp, out_dtype=out_dtype,
        BLOCK_M=1, BLOCK_N=64
    )

@triton.jit
def qkv(
    in_ptr, 
    q_ptr, k_ptr, v_ptr,
    k_cache_ptr, v_cache_ptr,
    fused_qk: tl.constexpr,
    q_dim: tl.constexpr, kv_dim: tl.constexpr,
):
    pass

def flash_attn():
    pass

def moe_scoring():
    pass

def ffn():
    pass