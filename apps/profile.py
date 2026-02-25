# TODO: complete
import argparse

import torch
import torch.nn.functional as F

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def rmsnorm(backend: str, which: str):
    kerns = KernelBackend(backend)
    # just adjust the shapes if you need per-head (4 dims) vs per-vec (3 dims)
    B, L, shape, eps = 8, 4096, (48,128), 1e-6
    
    x = torch.randn((B,L,*shape), dtype=torch.float16, device="cuda")
    out = torch.zeros_like(x)
    weight = (1/shape[-1]**0.5) * torch.randn(shape[-1], dtype=torch.float16, device="cuda")
    
    def my_rmsnorm():
        kerns.rmsnorm(eps, x, weight, out)

    def ref_rmsnorm():
        F.rms_norm(input=x, weight=weight, normalized_shape=(shape[-1],), eps=eps)
    
    rmsnorm_kern = my_rmsnorm if which == "minfer" else ref_rmsnorm

    with torch.no_grad():
        rmsnorm_kern()

def il_rope(backend: str, which: str):
    kerns = KernelBackend(backend)
    
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6
    
    x = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16, device="cuda")
    
    freqs = torch.arange(rotary_dim//2, dtype=torch.float32, device="cuda")
    freqs = base_freq**(-2.0*freqs/rotary_dim)
    freqs = torch.arange(L, dtype=torch.float32, device="cuda")[:, None] * freqs[None,:]
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    def my_il_rope():
        kerns.il_rope(rotary_dim, 0, base_freq, x)

    def ref_il_rope():
        x_float = x.float()
        x_complex = torch.view_as_complex(x_float[..., :rotary_dim].reshape(*x_float.shape[:-1], rotary_dim//2, 2))
        x_rotated = torch.view_as_real(x_complex * freqs_complex).flatten(-2)
        x_float[..., :rotary_dim] = x_rotated
        x_float.half()
    
    il_rope_kern = my_il_rope if which == "minfer" else ref_il_rope

    with torch.no_grad():
        il_rope_kern()

def neox_rope(backend: str, which: str):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6
    
    x = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16, device="cuda")
    
    freqs = torch.arange(rotary_dim//2, dtype=torch.float32, device="cuda")
    freqs = base_freq**(-2.0*freqs/rotary_dim)
    freqs = torch.arange(L, dtype=torch.float32, device="cuda")[:, None] * freqs[None,:]
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    def my_neox_rope():
        kerns.neox_rope(rotary_dim, 0, base_freq, x)

    def ref_neox_rope():
        x_float = x.float()
        x1 = x_float[..., :rotary_dim//2]
        x2 = x_float[..., rotary_dim//2:rotary_dim]
        x_complex = torch.view_as_complex(torch.stack([x1, x2], dim=-1))
        x_rotated = torch.view_as_real(x_complex * freqs_complex)
        x_float[..., :rotary_dim//2] = x_rotated[..., 0]
        x_float[..., rotary_dim//2:rotary_dim] = x_rotated[..., 1]
        x_float.half()

    neox_rope_kern = my_neox_rope if which == "minfer" else ref_neox_rope

    with torch.no_grad():
        neox_rope_kern()

def gemm(backend: str, which: str):
    kerns = KernelBackend("torch_ext")
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    M, K, N = 16384,16384,16384 # set this
    alpha, beta = 2.0, 3.0

    input = torch.randn((1,M,K), dtype=torch.float16, device="cuda")
    weight = (1/K**0.5) * torch.randn((K,N), dtype=torch.float16, device="cuda")
    bias = torch.randn((1,M,N), dtype=torch.float16, device="cuda")
    out = torch.zeros((1,M,N), dtype=torch.float16, device="cuda")

    bias_2d = bias.squeeze(0)
    input_2d = input.squeeze(0)
    out_2d = out.squeeze(0)

    def my_gemm():
        kerns.gemm(qtype, qblock_size, qtype_size, alpha, beta, input, weight, bias, out)

    def ref_gemm():
        torch.addmm(bias_2d, input_2d, weight, out_dtype=torch.float16, beta=beta, alpha=alpha, out=out_2d)

    gemm_kern = my_gemm if which == "minfer" else ref_gemm

    with torch.no_grad():
        gemm_kern()

def flash_attn(backend: str, which: str):
    kerns = KernelBackend(backend)
    B, L, n_heads, n_kv_heads, head_dim = 4,4096,48,8,128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    Q = torch.randn((B,L,n_heads*head_dim), dtype=torch.float16, device="cuda").view(B,L,n_heads,head_dim).transpose(1,2)
    K = torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16, device="cuda").view(B,L,n_kv_heads,head_dim).transpose(1,2)
    V = torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16, device="cuda").view(B,L,n_kv_heads,head_dim).transpose(1,2)
    mask = (torch.arange(L, device="cuda").unsqueeze(0) <= torch.arange(L,device="cuda").unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool)
    out = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16, device="cuda")

    def my_flash_attn():
        kerns.flash_attn(qtype, qblock_size, qtype_size, Q, K, V, mask, out)

    # for pytorch impl, K and V need to be expanded from n_kv_heads -> n_heads for GQA
    n_rep = n_heads // n_kv_heads
    K_exp = K.repeat_interleave(n_rep, dim=1)
    V_exp = V.repeat_interleave(n_rep, dim=1)

    def ref_flash_attn():
        return F.scaled_dot_product_attention(Q, K_exp, V_exp, attn_mask=mask)
    
    flash_attn_kern = my_flash_attn if which == "minfer" else ref_flash_attn

    with torch.no_grad():
        flash_attn_kern()

if __name__ == "__main__":
    KERNELS = {
        "rmsnorm": rmsnorm,
        "il_rope": il_rope,
        "neox_rope": neox_rope,
        "gemm": gemm,
        # "embed": embed, (TODO)
        # "qkv": qkv, (TODO)
        "flash_attn": flash_attn,
        # "moe_scoring": moe_scoring, (TODO)
        # "ffn": ffn, (TODO)
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch_ext"], default="torch_ext")
    args, _ = parser.parse_known_args()

    parser.add_argument("--kernel", choices=[k for k in vars(KernelBackend(args.backend)) if not k.startswith('_')], required=True)
    parser.add_argument("--which", choices=["minfer", "ref"], required=True)
    args = parser.parse_args()
    KERNELS[args.kernel](backend=args.backend, which=args.which)