# TODO: complete
import argparse

import torch

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def rmsnorm(backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    # just adjust the shapes if you need per-head (4 dims) vs per-vec (3 dims)
    B, L, eps, n_heads, head_dim = 8, 4096, 1e-6, 48, 128
    
    x = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16, device="cuda")
    out = torch.zeros_like(x)
    weight = (1/head_dim**0.5) * torch.randn(head_dim, dtype=torch.float16, device="cuda")
    
    kerns.rmsnorm(eps, x, weight, out)
    torch.cuda.synchronize()

def il_rope(backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, n_heads, L, head_dim, rotary_dim, base_freq = 8, 48, 4096, 128, 64, 1e6
    
    x = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16, device="cuda")
    kerns.il_rope(rotary_dim, 0, base_freq, x)
    torch.cuda.synchronize()

def neox_rope(backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, n_heads, L, head_dim, rotary_dim, base_freq = 8, 48, 4096, 128, 64, 1e6
    
    x = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16, device="cuda")
    kerns.neox_rope(rotary_dim, 0, base_freq, x)
    torch.cuda.synchronize()

def gemm(which:str, shape: tuple):
    kerns = KernelBackend("torch_ext")
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    M, K, N = shape
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

    gemm = my_gemm if which == "minfer" else ref_gemm

    with torch.no_grad():
        gemm()

    torch.cuda.synchronize()

def flash_attn(backend:str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, L, n_heads, n_kv_heads, head_dim = 4,4096,48,8,128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    Q = torch.randn((B,L,n_heads*head_dim), dtype=torch.float16, device="cuda").view(B,L,n_heads,head_dim).transpose(1,2)
    K = torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16, device="cuda").view(B,L,n_kv_heads,head_dim).transpose(1,2)
    V = torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16, device="cuda").view(B,L,n_kv_heads,head_dim).transpose(1,2)
    mask = (torch.arange(L, device="cuda").unsqueeze(0) <= torch.arange(L,device="cuda").unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool)
    out = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16, device="cuda")

    kerns.flash_attn(qtype, qblock_size, qtype_size, Q, K, V, mask, out)
    torch.cuda.synchronize()

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
    parser.add_argument("--kernel", choices=[k for k in vars(KernelBackend) if not k.startswith('_')], required=True)
    parser.add_argument("--which", choices=["minfer", "ref"], required=True)
    args = parser.parse_args()

    KERNELS[args.kernel](which=args.which)