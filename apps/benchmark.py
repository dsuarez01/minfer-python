# TODO: complete 
# NOTE: compare torch default CUDA opns, and custom CUDA opns (eventually Triton)
import argparse
import gc

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer, Compare
from torch.profiler import profile, ProfilerActivity, record_function

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

def rmsnorm(backend: str = "torch_ext"):
    """rmsnorm against pytorch F.rms_norm"""
    
    kerns = KernelBackend(backend)
    
    B,L = 8,4096
    eps = 1e-6
    head_dim = 128
    n_heads = 48
    hidden_dim = 6144
    
    test_cases = [
        ("per_head", (B,L,n_heads,head_dim)),
        ("full_hidden", (B,L,hidden_dim)),
    ]
    
    results = []
    
    for name, shape in test_cases:
        print(f"\nRunning {name}, {shape}")
        
        x = 1/(shape[-1]**0.5) * torch.randn(shape, dtype=torch.float16).cuda()
        weight = torch.randn(shape[-1], dtype=torch.float16).cuda()
        out = torch.zeros_like(x).cuda()
        
        def my_rmsnorm():
            kerns.rmsnorm(eps, x, out, weight)
            torch.cuda.synchronize()
        
        def ref_rmsnorm():
            result = F.rms_norm(input=x, weight=weight, normalized_shape=(shape[-1],), eps=eps)
            torch.cuda.synchronize()
            return result
        
        for fn_name, fn in [("my_rmsnorm", my_rmsnorm), ("ref_rmsnorm", ref_rmsnorm)]:
            timer = Timer(
                stmt='fn()',
                globals={'fn': fn},
                label=f'rms_norm_{name}',
                sub_label=fn_name,
                description=f'shape={shape}',
            )
            results.append(timer.blocked_autorange(min_run_time=1))
        
        del x, weight, out
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def rope(backend: str = "torch_ext"):
    """rope (il + neox) against pytorch ref"""
    
    kerns = KernelBackend(backend)
    
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6
    
    test_cases = [
        ("il_rope"),
        ("neox_rope"),
    ]
    
    results = []
    
    for name in test_cases:
        print(f"\nRunning {name}")
        
        x = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
        
        def ref_rope():
            freqs = torch.arange(rotary_dim//2, dtype=torch.float32).cuda()
            freqs = base_freq**(-2.0*freqs/rotary_dim)
            freqs = torch.arange(L, dtype=torch.float32)[:, None].cuda() * freqs[None,:]
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            
            x_float = x.float()
            if name == "il_rope":
                x_reshaped = x_float.reshape(*x_float.shape[:-1], -1, 2)
                out = x_reshaped.clone()
                out[..., :rotary_dim//2, 0] = cos*x_reshaped[..., :rotary_dim//2, 0]-sin*x_reshaped[..., :rotary_dim//2, 1]
                out[..., :rotary_dim//2, 1] = sin*x_reshaped[..., :rotary_dim//2, 0]+cos*x_reshaped[..., :rotary_dim//2, 1]
                out = out.flatten(-2).half()
            else:
                x_stacked = torch.stack([x_float[...,:rotary_dim//2], x_float[...,rotary_dim//2:rotary_dim]], dim=-1)
                out = x_float.clone()
                out[..., :rotary_dim//2] = cos*x_stacked[..., :rotary_dim//2, 0]-sin*x_stacked[..., :rotary_dim//2, 1]
                out[..., rotary_dim//2:rotary_dim] = sin*x_stacked[..., :rotary_dim//2, 0]+cos*x_stacked[..., :rotary_dim//2, 1]
                out = out.half()
            torch.cuda.synchronize()
            return out
        
        def my_rope():
            x_copy = x.clone()
            getattr(kerns, name)(rotary_dim, 0, base_freq, x_copy)
            torch.cuda.synchronize()
        
        for fn_name, fn in [(f"my_{name}", my_rope), (f"ref_{name}", ref_rope)]:
            timer = Timer(
                stmt='fn()',
                globals={'fn': fn},
                label=f'rope_{name}',
                sub_label=fn_name,
                description=f'B={B}, L={L}, n_heads={n_heads}, head_dim={head_dim}',
            )
            results.append(timer.blocked_autorange(min_run_time=1))
        
        del x
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def matmul(backend: str = "torch_ext"):
    """matmul against PyTorch @ op"""
    
    kerns = KernelBackend(backend)
    
    B, L, in_dim = 8, 4096, 6144
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]
    
    test_cases = [
        ("large_out", 16384),
        ("small_out", 8),
    ]
    
    results = []
    
    for name, out_dim in test_cases:
        print(f"\nRunning {name}: out_dim={out_dim}...")
        
        x = torch.randn((B, L, in_dim), dtype=torch.float16).cuda()
        out = torch.zeros((B, L, out_dim), dtype=torch.float16).cuda()
        weight = (1/in_dim**0.5) * torch.randn((out_dim, in_dim), dtype=torch.float16).cuda()
        
        def my_matmul():
            kerns.matmul(qtype, qblock_size, qtype_size, x, out, weight)
            torch.cuda.synchronize()
        
        def ref_matmul():
            result = x @ weight.T
            torch.cuda.synchronize()
            return result
        
        for fn_name, fn in [("my_matmul", my_matmul), ("ref_matmul", ref_matmul)]:
            timer = Timer(
                stmt='fn()',
                globals={'fn': fn},
                label=f'matmul_{name}',
                sub_label=fn_name,
                description=f'B={B}, L={L}, in_dim={in_dim}, out_dim={out_dim}',
            )
            results.append(timer.blocked_autorange(min_run_time=1))
        
        del x, out, weight
        gc.collect()
        torch.cuda.empty_cache()
    
    compare = Compare(results)
    compare.print()

def flash_attn(backend: str = "torch_ext"):
    """flash_attn against pytorch SDPA"""

    kerns = KernelBackend(backend)
    
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
            kerns.flash_attn(qtype, qblock_size, qtype_size, mask, out, Q, K1, V1)
            torch.cuda.synchronize()
        
        def ref_flash_attn():
            """pytorch SDPA"""
            result = F.scaled_dot_product_attention(Q, K2, V2, attn_mask=mask)
            torch.cuda.synchronize()
            return result
        
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
    # rmsnorm()
    # rope()
    matmul()
    # flash_attn()