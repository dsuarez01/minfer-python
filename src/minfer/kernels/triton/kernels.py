import triton
import triton.language as tl

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

# TODO: fix args since dequant logic has changed
@triton.jit
def matmul(
    act_ptr, quant_ptr, out_ptr, lut_ptr,
    batch_sz: tl.constexpr, seq_len: tl.constexpr,
    d_in: tl.constexpr, d_out: tl.constexpr,
    block_size: tl.constexpr, n_bits: tl.constexpr,
    packed_size: tl.constexpr, scale_size: tl.constexpr,
    has_zp: tl.constexpr, out_dtype: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """ (ceil(batch_sz*seq_len/TILE_SIZE), ceil(d_out/TILE_SIZE), ceil(d_in/TILE_SIZE)) """
    tile_batch, tile_out, tile_in = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    weight_tile = tl.zeros((TILE_SIZE, TILE_SIZE), dtype=out_dtype)
    
    weight_row = tile_out*TILE_SIZE
    # weight_col = tile_in*TILE_SIZE
    n_blocks_row = (d_in+block_size-1) // block_size
    block_stride = packed_size+scale_size
    weight_offs = weight_row*n_blocks_row*block_stride
    
    # TODO: fix
    # _dequant_row(
    #     quant_ptr+weight_offs, weight_tile, lut_ptr,
    #     M=TILE_SIZE, N=TILE_SIZE,
    #     block_size=block_size, n_bits=n_bits,
    #     packed_size=packed_size, scale_size=scale_size,
    #     has_zp=has_zp, out_dtype=out_dtype,
    #     BLOCK_M=TILE_SIZE, BLOCK_N=TILE_SIZE
    # )
    
    offs_batch = tile_batch*TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_in = tile_in*TILE_SIZE + tl.arange(0, TILE_SIZE)
    act_mask = (offs_batch < batch_sz*seq_len)[:, None] & (offs_in<d_in)[None, :]
    act_tile = tl.load(act_ptr + offs_batch[:, None]*d_in + offs_in[None, :], mask=act_mask, other=0.0)
    
    result = tl.dot(act_tile, tl.trans(weight_tile))
    
    offs_out = tile_out*TILE_SIZE + tl.arange(0, TILE_SIZE)
    out_mask = (offs_batch < batch_sz*seq_len)[:, None] & (offs_out<d_out)[None, :]
    tl.atomic_add(out_ptr + offs_batch[:, None]*d_out + offs_out[None, :], result, mask=out_mask)

# TODO: fix args since dequant logic has changed
@triton.jit
def embed(
    token_ids_ptr, quant_ptr, out_ptr, lut_ptr,
    batch_sz: tl.constexpr, seq_len: tl.constexpr, hidden_dim: tl.constexpr,
    block_size: tl.constexpr, n_bits: tl.constexpr,
    packed_size: tl.constexpr, scale_size: tl.constexpr,
    has_zp: tl.constexpr, out_dtype: tl.constexpr,
):
    """ (batch_sz*seq_len, ceil(hidden_dim/64)) """
    pid_token = tl.program_id(0)
    
    batch_idx = pid_token // seq_len
    seq_idx = pid_token % seq_len
    token_id = tl.load(token_ids_ptr + batch_idx*seq_len + seq_idx)
    
    n_blocks_row = (hidden_dim+block_size-1) // block_size
    block_stride = packed_size+scale_size
    row_offs = token_id*n_blocks_row*block_stride
    out_offs = (batch_idx*seq_len + seq_idx)*hidden_dim
    
    # TODO: fix
    # _dequant_row(
    #     quant_ptr+row_offs, out_ptr+out_offs, lut_ptr,
    #     M=1, N=hidden_dim,
    #     block_size=block_size, n_bits=n_bits,
    #     packed_size=packed_size, scale_size=scale_size,
    #     has_zp=has_zp, out_dtype=out_dtype,
    #     BLOCK_M=1, BLOCK_N=64
    # )

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