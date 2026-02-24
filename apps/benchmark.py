# TODO: complete
from itertools import product
import argparse
import gc

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer, Compare
from torch.profiler import profile, ProfilerActivity, record_function

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

torch.backends.cuda.matmul.allow_fp16_accumulation = True # forces Pytorch to use HGEMM for fair benchmarking
INT_MAX = 2147483647

def rmsnorm(which: str):
    """rmsnorm against pytorch F.rms_norm"""
    
    kerns = KernelBackend("torch_ext")
    
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

        x = 1/(shape[-1]**0.5) * torch.randn((B, L, *shape), dtype=torch.float16, device="cuda")
        weight = torch.randn(shape[-1], dtype=torch.float16, device="cuda")
        out = torch.zeros_like(x)
        
        def my_rmsnorm():
            kerns.rmsnorm(eps, x, weight, out)
            torch.cuda.synchronize()
        
        def ref_rmsnorm():
            result = F.rms_norm(input=x, weight=weight, normalized_shape=(shape[-1],), eps=eps)
            torch.cuda.synchronize()
        
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

def il_rope(which: str):
    """il_rope against pytorch ref"""
    
    kerns = KernelBackend("torch_ext")
    
    test_cases = [
        (8,48,1024,128,64,1e6),
        (8,48,4096,128,64,1e6),
        (8,48,8192,128,64,1e6),
        (8,48,16384,128,64,1e6),
    ]
    
    results = []
    
    for B, n_heads, L, head_dim, rotary_dim, base_freq in test_cases:
        
        x = torch.randn((B, n_heads, L, head_dim), dtype=torch.float16, device="cuda")
        
        freqs = torch.arange(rotary_dim//2, dtype=torch.float32, device="cuda")
        freqs = base_freq**(-2.0*freqs/rotary_dim)
        freqs = torch.arange(L, dtype=torch.float32, device="cuda")[:, None] * freqs[None,:]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        def ref_il_rope():
            x_float = x.float()
            x_complex = torch.view_as_complex(x_float[..., :rotary_dim].reshape(*x_float.shape[:-1], rotary_dim//2, 2))
            x_rotated = torch.view_as_real(x_complex * freqs_complex).flatten(-2)
            x_float[..., :rotary_dim] = x_rotated
            x_float.half()
        
        def my_il_rope():
            kerns.il_rope(rotary_dim, 0, base_freq, x)
        
        fn = my_il_rope if which == "minfer" else ref_il_rope

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label='il_rope',
            sub_label=f'B={B}, n_heads={n_heads}, L={L}, head_dim={head_dim}, rotary_dim={rotary_dim}, base_freq={base_freq}',
            description=f'{which}',
        )
        results.append(timer.blocked_autorange())
        
        del x
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def neox_rope(which: str):
    """il_rope against pytorch ref"""
    
    kerns = KernelBackend("torch_ext")
    
    test_cases = [
        (8,48,1024,128,64,1e6),
        (8,48,4096,128,64,1e6),
        (8,48,8192,128,64,1e6),
        (8,48,16384,128,64,1e6),
    ]
    
    results = []
    
    for B, n_heads, L, head_dim, rotary_dim, base_freq in test_cases:
        
        x = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16, device="cuda")
        
        freqs = torch.arange(rotary_dim//2, dtype=torch.float32, device="cuda")
        freqs = base_freq**(-2.0*freqs/rotary_dim)
        freqs = torch.arange(L, dtype=torch.float32, device="cuda")[:, None] * freqs[None,:]
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

        def ref_neox_rope():
            x_float = x.float()
            x_complex = torch.view_as_complex(x_float[..., :rotary_dim].reshape(*x_float.shape[:-1], rotary_dim//2, 2))
            x_rotated = torch.view_as_real(x_complex * freqs_complex).flatten(-2)
            x_float[..., :rotary_dim] = x_rotated
            x_float.half()
        
        def my_neox_rope():
            kerns.neox_rope(rotary_dim, 0, base_freq, x)
        
        fn = my_neox_rope if which == "minfer" else ref_neox_rope

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label=f'neox_rope',
            sub_label=f'B={B}, n_heads={n_heads}, L={L}, head_dim={head_dim}, rotary_dim={rotary_dim}, base_freq={base_freq}',
            description=f'{which}',
        )
        results.append(timer.blocked_autorange())
        
        del x
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def gemm(which: str):
    """matmul against PyTorch @ op"""
    
    kerns = KernelBackend("torch_ext")
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

        def ref_matmul():
            with torch.no_grad():
                torch.addmm(bias_2d, input_2d, weight, out_dtype=torch.float16, beta=beta, alpha=alpha, out=out_2d)
        
        def my_matmul():
            with torch.no_grad():
                kerns.gemm(qtype, qblock_size, qtype_size, alpha, beta, input, weight, bias, out)

        fn = my_matmul if which == "minfer" else ref_matmul

        timer = Timer(
            stmt='fn()',
            globals={'fn': fn},
            label=f'matmul',
            sub_label=f'M={M}, K={K}, N={N}',
            description=f'{which}',
        )

        result = timer.blocked_autorange()

        results.append(result)

        del input, weight, bias, out
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def flash_attn(which: str):
    """flash_attn against pytorch SDPA"""

    kerns = KernelBackend("torch_ext")
    
    B, n_heads, n_kv_heads, head_dim = 4, 48, 8, 128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]
    
    seq_lens = [4096, 8192, 16384]
    results = []
    
    for L in seq_lens:
        print(f"\nRunning L={L}")
        
        Q = (1/head_dim**0.5) * torch.randn((B,L,n_heads*head_dim), dtype=torch.float16).cuda()
        Q = Q.view(B,L,n_heads,head_dim).transpose(1, 2)

        K1 = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
        K1 = K1.view(B,L,n_kv_heads,head_dim).transpose(1, 2)

        V1 = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
        V1 = V1.view(B,L,n_kv_heads,head_dim).transpose(1, 2)
        
        # we choose causal mask
        mask = (torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool).cuda()
        
        out = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
        
        # pytorch SDPA doesn't handle GQA natively
        n_rep = n_heads // n_kv_heads
        K2 = K1.repeat_interleave(n_rep, dim=1)
        V2 = V1.repeat_interleave(n_rep, dim=1)
        
        def my_flash_attn():
            kerns.flash_attn(qtype, qblock_size, qtype_size, Q, K1, V1, mask, out)
            torch.cuda.synchronize()
        
        def ref_flash_attn():
            """pytorch SDPA"""
            result = F.scaled_dot_product_attention(Q, K2, V2, attn_mask=mask)
            torch.cuda.synchronize()
        
        for name, fn in [("my_flash_attn", my_flash_attn), ("ref_flash_attn", ref_flash_attn)]:
            timer = Timer(
                stmt='fn()',
                globals={'fn': fn},
                label=f'flash_attn_L={L}',
                sub_label=name,
                description=f'B={B}, L={L}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}',
            )
            results.append(timer.blocked_autorange(min_run_time=1))

        del Q, K1, V1, K2, V2, mask, out
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
    parser.add_argument("--kernel", choices=[k for k in vars(KernelBackend) if not k.startswith('_')], required=True)
    parser.add_argument("--which", choices=["minfer", "ref"], required=True)
    args = parser.parse_args()
    
    KERNELS[args.kernel](which=args.which)