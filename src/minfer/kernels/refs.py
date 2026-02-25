import torch
import torch.nn.functional as F

def rmsnorm(
    x: torch.Tensor, 
    weight: torch.Tensor, 
    eps: float
):
    """rmsnorm reference. x: (B, L, n_heads, head_dim) (per-head) or (B, L, hidden_dim) (per-vec)"""
    return F.rms_norm(input=x, weight=weight, normalized_shape=(x.shape[-1],), eps=eps)

def il_rope(
    x: torch.Tensor,
    rotary_dim: int,
    start_pos: int,
    base_freq: float,
) -> torch.Tensor:
    """interleaved RoPE reference. x: (B, L, n_heads, head_dim)"""
    B, L, n_heads, head_dim = x.shape
    freqs = torch.arange(rotary_dim // 2, dtype=torch.float32, device=x.device)
    freqs = base_freq ** (-2.0 * freqs / rotary_dim)
    freqs = (torch.arange(L, dtype=torch.float32, device=x.device) + start_pos)[:, None] * freqs[None, :]
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    x_float = x.float()
    x_complex = torch.view_as_complex(x_float[..., :rotary_dim].reshape(*x_float.shape[:-1], rotary_dim // 2, 2))
    x_rotated = torch.view_as_real(x_complex * freqs_complex).flatten(-2)
    x_float[..., :rotary_dim] = x_rotated
    return x_float.half()

def neox_rope(
    x: torch.Tensor,
    rotary_dim: int,
    start_pos: int,
    base_freq: float,
) -> torch.Tensor:
    """neox (two-halves) RoPE reference. x: (B, L, n_heads, head_dim)"""
    B, L, n_heads, head_dim = x.shape
    freqs = torch.arange(rotary_dim // 2, dtype=torch.float32, device=x.device)
    freqs = base_freq ** (-2.0 * freqs / rotary_dim)
    freqs = (torch.arange(L, dtype=torch.float32, device=x.device) + start_pos)[:, None] * freqs[None, :]
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    x_float = x.float()
    x_complex = torch.view_as_complex(torch.stack([x_float[..., :rotary_dim // 2], x_float[..., rotary_dim // 2:rotary_dim]], dim=-1))
    x_rotated = torch.view_as_real(x_complex * freqs_complex)
    x_float[..., :rotary_dim // 2] = x_rotated[..., 0]
    x_float[..., rotary_dim // 2:rotary_dim] = x_rotated[..., 1]
    return x_float.half()

def gemm(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor, alpha: float, beta: float):
    """addmm reference: alpha*input@weight + beta*bias -> out. ensure input/bias/out are also 2D (may need to squeeze tensors)"""
    torch.addmm(bias, input, weight, out_dtype=torch.float16, beta=beta, alpha=alpha, out=out)

def flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """scaled dot prod attn (SDPA) reference. Q/K/V: (B, n_heads, L, head_dim), note that K/V must be pre-expanded for GQA"""
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)