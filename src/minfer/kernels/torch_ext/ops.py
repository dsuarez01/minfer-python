import torch
from minfer.const import GGMLQuantizationType

# NOTE: needed to make CUDA/C++ kernels compatible with torch.compile here

# TODO: complete

def dequant(
    qtype: GGMLQuantizationType, 
    qblock_size: int, 
    qtype_size: int,
    x: torch.Tensor, 
    y: torch.Tensor,
):
    """
    Dequantizes the rows of x into y
    
    qblock_size is num dequantized elements per block
    qtype_size is byte size of block
    """
    return torch.ops.minfer.dequant.default(qtype, qblock_size, qtype_size, x, y)

@torch.library.register_fake("minfer::dequant")
def _(
    qtype, qblock_size, qtype_size, x, y, 
):
    torch._check(x.dtype == torch.uint8)
    torch._check(y.dtype in (torch.float32, torch.float16))
    torch._check(x.device == y.device)
    torch._check(x.device.type == "cuda" or x.device.type == "cpu")
    torch._check(x.size(0) == y.size(0))

def quant(
    qtype: GGMLQuantizationType, 
    qblock_size: int, 
    qtype_size: int,
    x: torch.Tensor, 
    y: torch.Tensor,
):
    """
    Quantizes the rows of x into y
    
    qblock_size is num dequantized elements per block
    qtype_size is byte size of block
    """
    return torch.ops.minfer.quant.default(qtype, qblock_size, qtype_size, x, y)

@torch.library.register_fake("minfer::quant")
def _(
    qtype, qblock_size, qtype_size, x, y,
):
    torch._check(x.dtype == torch.float32)
    torch._check(y.dtype == torch.uint8)
    torch._check(x.device == y.device)
    torch._check(x.device.type == "cuda" or x.device.type == "cpu")
    torch._check(x.size(0) == y.size(0))

def rmsnorm(
    eps: float, 
    input: torch.Tensor, 
    w: torch.Tensor,
    out: torch.Tensor,
):
    """
    Applies rmsnorm (per-head or across entire dim)
    """

    return torch.ops.minfer.rmsnorm.default(eps, input, w, out)

@torch.library.register_fake("minfer::rmsnorm")
def _(
    eps: float, 
    input: torch.Tensor, 
    w: torch.Tensor,
    out: torch.Tensor,
):
    torch._check(input.dtype == torch.float16)
    torch._check(w.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(input.device == w.device == out.device)
    torch._check(input.device.type == "cuda")
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
    torch._check(x.device.type == "cuda")
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
    torch._check(x.device.type == "cuda")
    torch._check(rotary_dim <= x.size(3))

def matmul(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int, 
    x: torch.Tensor, 
    w: torch.Tensor,
    out: torch.Tensor,
):
    """
    computes x @ w.T, stores in out
    """
    return torch.ops.minfer.matmul.default(qtype_int, qblock_size, qtype_size, x, w, out)

@torch.library.register_fake("minfer::matmul")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(w.dtype == torch.uint8)
    torch._check(out.dtype == torch.float16)
    torch._check(x.device == w.device == out.device)
    torch._check(x.device.type == "cuda")

def embed(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int, 
    token_ids: torch.Tensor, 
    w: torch.Tensor,
    x: torch.Tensor,
):
    """
    dequantizes rows of w according to token_ids, stores in x
    """
    return torch.ops.minfer.embed.default(qtype_int, qblock_size, qtype_size, token_ids, w, x)

@torch.library.register_fake("minfer::embed")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int, token_ids: torch.Tensor, w: torch.Tensor, x: torch.Tensor,
):
    torch._check(x.dtype == torch.float16)
    torch._check(token_ids.dtype == torch.int64)
    torch._check(w.dtype == torch.uint8 or w.dtype == torch.float16)

    torch._check(x.device == token_ids.device == w.device)
    torch._check(x.device.type == "cuda")
    torch._check(token_ids.dim() == 2)
    torch._check(w.dim() == 2)
    torch._check(x.dim() == 3)

def qkv(
    q_qtype_int: int, k_qtype_int: int, v_qtype_int: int, 
    q_qblock_size: int, k_qblock_size: int, v_qblock_size: int,
    q_qtype_size: int, k_qtype_size: int, v_qtype_size: int,
    x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
    q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
):
    """
    fused qkv projections
    """
    
    return torch.ops.minfer.qkv.default(
        q_qtype_int, k_qtype_int, v_qtype_int,
        q_qblock_size, k_qblock_size, v_qblock_size,
        q_qtype_size, k_qtype_size, v_qtype_size,
        x, wq, wk, wv,
        q_out, k_out, v_out,
    )

@torch.library.register_fake("minfer::qkv")
def _(
    q_qtype_int: int, k_qtype_int: int, v_qtype_int: int,
    q_qblock_size: int, k_qblock_size: int, v_qblock_size: int,
    q_qtype_size: int, k_qtype_size: int, v_qtype_size: int,
    x: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor,
    q_out: torch.Tensor, k_out: torch.Tensor, v_out: torch.Tensor,
):
    torch._check(x.dtype == torch.float16)
    torch._check(q_out.dtype == torch.float16)
    torch._check(k_out.dtype == torch.float16)
    torch._check(v_out.dtype == torch.float16)
    torch._check(wq.dtype == torch.uint8 or wq.dtype == torch.float16)
    torch._check(wk.dtype == torch.uint8 or wk.dtype == torch.float16)
    torch._check(wv.dtype == torch.uint8 or wv.dtype == torch.float16)
    torch._check(x.device == wq.device == wk.device == wv.device == q_out.device == k_out.device == v_out.device)
    torch._check(x.device.type == "cuda")
    torch._check(x.dim() == 3)
    torch._check(q_out.dim() == 3)
    torch._check(k_out.dim() == 3)
    torch._check(v_out.dim() == 3)
    torch._check(wq.dim() == 2)
    torch._check(wk.dim() == 2)
    torch._check(wv.dim() == 2)

def flash_attn(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int,
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    mask: torch.Tensor,
    out: torch.Tensor, 
):
    """
    flash attention (v2-style w/ KV in outer loop)
    """
    return torch.ops.minfer.flash_attn.default(qtype_int, qblock_size, qtype_size, q, k, v, mask, out)

@torch.library.register_fake("minfer::flash_attn")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int, 
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, out: torch.Tensor,
):
    torch._check(q.dtype == torch.float16)
    torch._check(k.dtype == torch.float16)
    torch._check(v.dtype == torch.float16)
    torch._check(mask.dtype == torch.bool)
    torch._check(out.dtype == torch.float16)
    torch._check(q.device == k.device == v.device == out.device)
    torch._check(q.device.type == "cuda")
    torch._check(q.dim() == 4)
    torch._check(k.dim() == 4)
    torch._check(v.dim() == 4)
    torch._check(mask.dim() == 4)
    torch._check(out.dim() == 4)

def moe_scoring(
    qtype_int: int, 
    qblock_size: int, 
    qtype_size: int,
    x: torch.Tensor, 
    w: torch.Tensor,
    act_exps: torch.Tensor,
    act_exps_scores: torch.Tensor,
    scores: torch.Tensor
):
    """
    computes scores for each expert (fused softmax and top-K)
    """
    return torch.ops.minfer.moe_scoring.default(qtype_int, qblock_size, qtype_size, x, w, act_exps, act_exps_scores, scores)

@torch.library.register_fake("minfer::moe_scoring")
def _(
    qtype_int: int, qblock_size: int, qtype_size: int,
    x: torch.Tensor, w: torch.Tensor,
    act_exps: torch.Tensor, act_exps_scores: torch.Tensor,
    scores: torch.Tensor
):
    torch._check(x.dtype == torch.float16)
    torch._check(w.dtype == torch.float16)
    torch._check(act_exps.dtype == torch.uint8)
    torch._check(act_exps_scores.dtype == torch.float16)
    torch._check(scores.dtype == torch.float16)
    torch._check(x.device == w.device == act_exps.device == act_exps_scores.device == scores.device)
    torch._check(x.device.type == "cuda")
    torch._check(w.size(0) == scores.size(-1))

def ffn(
    up_qtype_int: int, gate_qtype_int: int, down_qtype_int: int,
    up_qblock_size: int, gate_qblock_size: int, down_qblock_size: int,
    up_qtype_size: int, gate_qtype_size: int, down_qtype_size: int,
    input: torch.Tensor, ws_up: torch.Tensor, ws_gate: torch.Tensor, ws_down: torch.Tensor,
    hb: torch.Tensor, hb2: torch.Tensor, out: torch.Tensor,
):
    """
    computes per-expert ffn opn
    """

    return torch.ops.minfer.ffn.default(
        up_qtype_int, gate_qtype_int, down_qtype_int,
        up_qblock_size, gate_qblock_size, down_qblock_size,
        up_qtype_size, gate_qtype_size, down_qtype_size,
        input, ws_up, ws_gate, ws_down, hb, hb2, out
    )

@torch.library.register_fake("minfer::ffn")
def _(
    up_qtype_int: int, gate_qtype_int: int, down_qtype_int: int,
    up_qblock_size: int, gate_qblock_size: int, down_qblock_size: int,
    up_qtype_size: int, gate_qtype_size: int, down_qtype_size: int,
    input: torch.Tensor, ws_up: torch.Tensor, ws_gate: torch.Tensor, ws_down: torch.Tensor,
    hb: torch.Tensor, hb2: torch.Tensor, out: torch.Tensor,
):
    torch._check(input.dtype == torch.float16)
    torch._check(out.dtype == torch.float16)
    torch._check(hb.dtype == torch.float16)
    torch._check(hb2.dtype == torch.float16)
    torch._check(ws_up.dtype == torch.uint8 or ws_up.dtype == torch.float16)
    torch._check(ws_gate.dtype == torch.uint8 or ws_gate.dtype == torch.float16)
    torch._check(ws_down.dtype == torch.uint8 or ws_down.dtype == torch.float16)
    torch._check(input.device == out.device == ws_up.device == ws_gate.device 
                 == ws_down.device == hb.device == hb2.device == out.device)
    torch._check(input.device.type == "cuda")


    torch._check(input.numel() == out.numel() // out.size(0))
    torch._check(hb.sizes() == hb2.sizes())
    torch._check(input.dim() == 3)
    torch._check(out.dim() == 4)
    torch._check(hb.dim() == 4)
    torch._check(hb2.dim() == 4)