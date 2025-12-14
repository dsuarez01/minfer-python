import torch
from torch import Tensor
from minfer.const import GGMLQuantizationType

# NOTE: needed to make CUDA/C++ kernels compatible with torch.compile here

# TODO: complete

def dequant(
    qtype: GGMLQuantizationType, 
    x: Tensor, 
    y: Tensor, 
    qblock_size: int, 
    qtype_size: int
):
    """
    Dequantizes the rows of x into y
    
    qblock_size is num dequantized elements per block
    qtype_size is byte size of block
    """
    return torch.ops.minfer.dequant.default(qtype, x, y, qblock_size, qtype_size)

@torch.library.register_fake("minfer::dequant")
def _(
    qtype, x, y, qblock_size, qtype_size
):
    torch._check(x.dtype == torch.uint8)
    torch._check(y.dtype in (torch.float32, torch.float16))
    torch._check(x.device == y.device)
    torch._check(x.size(0) == y.size(0))

def quant(
    qtype: GGMLQuantizationType, 
    x: Tensor, 
    y: Tensor, 
    qblock_size: int, 
    qtype_size: int
):
    """
    Quantizes the rows of x into y
    
    qblock_size is num dequantized elements per block
    qtype_size is byte size of block
    """
    return torch.ops.minfer.quant.default(qtype, x, y, qblock_size, qtype_size)

@torch.library.register_fake("minfer::quant")
def _(
    qtype, x, y, qblock_size, qtype_size
):
    torch._check(x.dtype == torch.float32)
    torch._check(y.dtype == torch.uint8)
    torch._check(x.device == y.device)
    torch._check(x.size(0) == y.size(0))

def rmsnorm(
    eps: float, 
    input: torch.Tensor, 
    out: torch.Tensor, 
    w: torch.Tensor
):
    """
    Applies rmsnorm (per-head or across entire dim)
    """

    return torch.ops.minfer.rmsnorm.default(eps, input, out, w)

@torch.library.register_fake("minfer::rmsnorm")
def _(
    eps: float, 
    input: torch.Tensor, 
    out: torch.Tensor, 
    w: torch.Tensor
):
    torch._check(input.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(w.dtype == torch.float16)
    torch._check(input.device == out.device)
    torch._check(input.device == w.device)
    torch._check(input.sizes() == out.sizes())
    torch._check(input.dim() in (3, 4))
    torch._check(input.size(-1) == w.size(0) or input.size(-1) % w.size(0) == 0)

def il_rope(
    rotary_dim: int, 
    start_pos: int, 
    freq_base: float, 
    x: torch.Tensor
):
    """
    Applies il_rope in-place (per-head)
    """

    return torch.ops.minfer.il_rope.default(rotary_dim, start_pos, freq_base, x)

@torch.library.register_fake("minfer::il_rope")
def _(
    rotary_dim: int, start_pos: int, freq_base: float, x: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(x.dim() == 4)
    torch._check(rotary_dim <= x.size(3))

def neox_rope(
    rotary_dim: int, 
    start_pos: int, 
    freq_base: float, 
    x: torch.Tensor
):
    """
    Applies neox_rope in-place (per-head)
    """

    return torch.ops.minfer.neox_rope.default(rotary_dim, start_pos, freq_base, x)

@torch.library.register_fake("minfer::neox_rope")
def _(
    rotary_dim: int, start_pos: int, freq_base: float, x: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(x.dim() == 4)
    torch._check(rotary_dim <= x.size(3))

def matmul(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int, 
    x: torch.Tensor, 
    out: torch.Tensor, 
    w: torch.Tensor
):
    """
    computes x @ w.T, stores in out
    """
    return torch.ops.minfer.matmul.default(qtype_int, qblock_size, qtype_size, x, out, w)

@torch.library.register_fake("minfer::matmul")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int, x: torch.Tensor, out: torch.Tensor, w: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(w.dtype == torch.uint8)
    torch._check(x.device == out.device)
    torch._check(x.device == w.device)

def embed(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int, 
    x: torch.Tensor, 
    token_ids: torch.Tensor, 
    w: torch.Tensor
):
    """
    dequantizes rows of w according to token_ids, stores in x
    """
    return torch.ops.minfer.embed.default(qtype_int, qblock_size, qtype_size, x, token_ids, w)

@torch.library.register_fake("minfer::embed")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int, x: torch.Tensor, token_ids: torch.Tensor, w: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(token_ids.dtype == torch.int64)
    torch._check(w.dtype == torch.uint8)
    torch._check(x.device == token_ids.device)
    torch._check(x.device == w.device)
    torch._check(token_ids.dim() == 2)
    torch._check(x.dim() == 3)

def qkv(
    q_qtype_int: int, k_qtype_int: int, v_qtype_int: int, 
    q_qblock_size: int, k_qblock_size: int, v_qblock_size: int,
    q_qtype_size: int, k_qtype_size: int, v_qtype_size: int,
    x: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
    wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor
):
    """
    fused qkv projections
    """
    
    return torch.ops.minfer.qkv.default(
        q_qtype_int, k_qtype_int, v_qtype_int,
        q_qblock_size, k_qblock_size, v_qblock_size,
        q_qtype_size, k_qtype_size, v_qtype_size,
        x, q_out, k_out, v_out, wq, wk, wv
    )

@torch.library.register_fake("minfer::qkv")
def _(
    q_qtype_int: int, k_qtype_int: int, v_qtype_int: int,
    q_qblock_size: int, k_qblock_size: int, v_qblock_size: int,
    q_qtype_size: int, k_qtype_size: int, v_qtype_size: int,
    x: torch.Tensor, q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
    wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(q_out.dtype == torch.float16)
    torch._check(k_out.dtype == torch.float16)
    torch._check(v_out.dtype == torch.float16)
    torch._check(wq.dtype == torch.uint8)
    torch._check(wk.dtype == torch.uint8)
    torch._check(wv.dtype == torch.uint8)
    torch._check(x.device == q_out.device)
    torch._check(x.device == k_out.device)
    torch._check(x.device == v_out.device)
    torch._check(x.dim() == 3)
    torch._check(q_out.dim() == 3)
    torch._check(k_out.dim() == 3)
    torch._check(v_out.dim() == 3)

def flash_attn(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int,
    out: torch.Tensor, 
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor
):
    """
    flash attention (v2-style w/ tree reduction)
    """
    return torch.ops.minfer.flash_attn.default(qtype_int, qblock_size, qtype_size, out, q, k, v)

@torch.library.register_fake("minfer::flash_attn")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int, 
    out: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
):
    torch._check(q.dtype == torch.float16)
    torch._check(k.dtype == torch.float16)
    torch._check(v.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(q.device == k.device)
    torch._check(q.device == v.device)
    torch._check(q.device == out.device)
    torch._check(q.dim() == 4)
    torch._check(k.dim() == 4)
    torch._check(v.dim() == 4)
    torch._check(out.dim() == 4)

def moe_scoring(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int,
    x: torch.Tensor, 
    out: torch.Tensor, 
    w: torch.Tensor
):
    """
    computes scores for each expert (fused softmax)
    """
    return torch.ops.minfer.moe_scoring.default(qtype_int, qblock_size, qtype_size, x, out, w)

@torch.library.register_fake("minfer::moe_scoring")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int,
    x: torch.Tensor, out: torch.Tensor, w: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(w.dtype == torch.uint8)
    torch._check(x.device == out.device)
    torch._check(x.device == w.device)
    torch._check(w.size(0) == out.size(-1))

def ffn(
    up_qtype_int: int, gate_qtype_int: int, down_qtype_int: int,
    up_qblock_size: int, gate_qblock_size: int, down_qblock_size: int,
    up_qtype_size: int, gate_qtype_size: int, down_qtype_size: int,
    input: torch.Tensor, out: torch.Tensor, hb: torch.Tensor, hb2: torch.Tensor,
    ws_up: torch.Tensor, ws_gate: torch.Tensor, ws_down: torch.Tensor
):
    """
    computes per-expert ffn opn
    """

    return torch.ops.minfer.ffn.default(
        up_qtype_int, gate_qtype_int, down_qtype_int,
        up_qblock_size, gate_qblock_size, down_qblock_size,
        up_qtype_size, gate_qtype_size, down_qtype_size,
        input, out, hb, hb2, ws_up, ws_gate, ws_down
    )

@torch.library.register_fake("minfer::ffn")
def _(
    up_qtype_int: int, gate_qtype_int: int, down_qtype_int: int,
    up_qblock_size: int, gate_qblock_size: int, down_qblock_size: int,
    up_qtype_size: int, gate_qtype_size: int, down_qtype_size: int,
    input: torch.Tensor, out: torch.Tensor, hb: torch.Tensor, hb2: torch.Tensor,
    ws_up: torch.Tensor, ws_gate: torch.Tensor, ws_down: torch.Tensor
):
    torch._check(input.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(hb.dtype == torch.float16)
    torch._check(hb2.dtype == torch.float16)
    torch._check(ws_up.dtype == torch.uint8)
    torch._check(ws_gate.dtype == torch.uint8)
    torch._check(ws_down.dtype == torch.uint8)
    torch._check(input.device == out.device)
    torch._check(input.sizes() == out.sizes())
    torch._check(hb.sizes() == hb2.sizes())
    torch._check(input.dim() == 3)
    torch._check(out.dim() == 3)
    torch._check(hb.dim() == 4)
    torch._check(hb2.dim() == 4)