# TODO: complete
from itertools import product
import argparse
import gc

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer, Compare
from torch.profiler import profile, ProfilerActivity, record_function

from minfer.kernels import refs
from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

torch.backends.cuda.matmul.allow_fp16_accumulation = True # forces Pytorch to use HGEMM for fair benchmarking
INT_MAX = 2147483647

def rmsnorm(backend: str, which: str):
    """rmsnorm against pytorch F.rms_norm"""
    
    kerns = KernelBackend(backend)
    
    test_cases = [
        (8,1024,(6144,),1e-6),
        (8,4096,(6144,),1e-6),
        (8,8192,(6144,),1e-6),
        (8,16384,(6144,),1e-6),
        (8,1024,(48,128),1e-6),
        (8,4096,(48,128),1e-6),
        (8,8192,(48,128),1e-6),
        (8,16384,(48,128),1e-6),
    ]
    
    results = []
    
    for B, L, shape, eps in test_cases:

        x = torch.randn((B, L, *shape), dtype=torch.float16, device="cuda")
        weight = 1/(shape[-1]**0.5) * torch.randn(shape[-1], dtype=torch.float16, device="cuda")
        out = torch.zeros_like(x)
        
        def my_rmsnorm():
            with torch.no_grad():
                kerns.rmsnorm(eps, x, weight, out)
        
        def ref_rmsnorm():
            with torch.no_grad():
                refs.rmsnorm(x, weight, eps)
        
        fn = my_rmsnorm if which == "minfer" else ref_rmsnorm

        shape_str = f'hidden_dim={shape[-1]}' if len(shape) == 1 else f'n_heads={shape[0]}, head_dim={shape[1]}'
        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label=f'rmsnorm',
            sub_label=f'B={B}, L={L}, {shape_str}',
            description=f'{which}',
        )
        results.append(timer.blocked_autorange())
        
        del x, weight, out
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def il_rope(backend: str, which: str):
    """il_rope against pytorch ref"""
    
    kerns = KernelBackend(backend)
    
    test_cases = [
        (8,1024,48,128,64,1e6),
        (8,4096,48,128,64,1e6),
        (8,8192,48,128,64,1e6),
        (8,16384,48,128,64,1e6),
    ]
    
    results = []
    
    for B, L, n_heads, head_dim, rotary_dim, base_freq in test_cases:
        
        x = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16, device="cuda")
        
        def my_il_rope():
            with torch.no_grad():
                kerns.il_rope(rotary_dim, 0, base_freq, x)
        
        def ref_il_rope():
            with torch.no_grad():
                refs.il_rope(x, rotary_dim, 0, base_freq)

        fn = my_il_rope if which == "minfer" else ref_il_rope

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label='il_rope',
            sub_label=f'B={B}, L={L}, n_heads={n_heads}, head_dim={head_dim}, rotary_dim={rotary_dim}, base_freq={base_freq}',
            description=f'{which}',
        )
        results.append(timer.blocked_autorange())
        
        del x
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def neox_rope(backend: str, which: str):
    """il_rope against pytorch ref"""
    
    kerns = KernelBackend(backend)
    
    test_cases = [
        (8,1024,48,128,64,1e6),
        (8,4096,48,128,64,1e6),
        (8,8192,48,128,64,1e6),
        (8,16384,48,128,64,1e6),
    ]
    
    results = []
    
    for B, L, n_heads, head_dim, rotary_dim, base_freq in test_cases:
        
        x = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16, device="cuda")

        def my_neox_rope():
            with torch.no_grad():
                kerns.neox_rope(rotary_dim, 0, base_freq, x)

        def ref_neox_rope():
            with torch.no_grad():
                refs.neox_rope(x, rotary_dim, 0, base_freq)
        
        fn = my_neox_rope if which == "minfer" else ref_neox_rope

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label=f'neox_rope',
            sub_label=f'B={B}, L={L}, n_heads={n_heads}, head_dim={head_dim}, rotary_dim={rotary_dim}, base_freq={base_freq}',
            description=f'{which}',
        )
        results.append(timer.blocked_autorange())
        
        del x
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def gemm(backend: str, which: str):
    """matmul against PyTorch @ op"""
    
    kerns = KernelBackend(backend)
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    alpha, beta = 2.0, 3.0
    sizes = [512,1024,2048,4096,8192,16384,32768]
    results = []

    for s in sizes:

        M = K = N = s
    
        input = torch.randn((1,M,K), dtype=torch.float16, device="cuda")
        weight = (1/K**0.5) * torch.randn((K,N), dtype=torch.float16, device="cuda")
        bias = torch.randn((1,M,N), dtype=torch.float16, device="cuda")
        out = torch.zeros((1,M,N), dtype=torch.float16, device="cuda")

        bias_2d = bias.squeeze(0)
        input_2d = input.squeeze(0)
        out_2d = out.squeeze(0)

        def my_gemm():
            with torch.no_grad():
                kerns.gemm(qtype, qblock_size, qtype_size, alpha, beta, input, weight, bias, out)

        def ref_gemm():
            with torch.no_grad():
                refs.gemm(input_2d, weight, bias_2d, out_2d, alpha, beta)

        fn = my_gemm if which == "minfer" else ref_gemm

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label=f'gemm',
            sub_label=f'alpha={alpha}, beta={beta}, M={M}, K={K}, N={N}',
            description=f'{which}',
        )

        result = timer.blocked_autorange()

        results.append(result)

        del input, weight, bias, out, input_2d, bias_2d, out_2d
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def flash_attn(backend: str, which: str):
    """flash_attn against pytorch SDPA"""

    kerns = KernelBackend(backend)
    
    B, n_heads, n_kv_heads, head_dim = 4, 48, 8, 128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]
    
    seq_lens = [4096, 8192, 16384]
    results = []
    
    for L in seq_lens:
        
        Q = (1/head_dim**0.5) * torch.randn((B,L,n_heads*head_dim), dtype=torch.float16).cuda()
        Q = Q.view(B,L,n_heads,head_dim).transpose(1,2)

        K = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
        K = K.view(B,L,n_kv_heads,head_dim).transpose(1,2)

        V = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
        V = V.view(B,L,n_kv_heads,head_dim).transpose(1,2)
        
        # we choose causal mask
        mask = (torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool).cuda()
        
        out = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
        
        # pytorch SDPA doesn't handle GQA natively
        n_rep = n_heads // n_kv_heads
        K_exp = K.repeat_interleave(n_rep, dim=1)
        V_exp = V.repeat_interleave(n_rep, dim=1)
        
        def my_flash_attn():
            kerns.flash_attn(qtype, qblock_size, qtype_size, Q, K, V, mask, out)
        
        def ref_flash_attn():
            refs.flash_attn(Q, K_exp, V_exp, mask)
        
        fn = my_flash_attn if which == "minfer" else ref_flash_attn

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label=f'flash_attn',
            sub_label=f'B={B}, L={L}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}',
            description=f'{which}',
        )
        results.append(timer.blocked_autorange())

        del Q, K, V, K_exp, V_exp, mask, out
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

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