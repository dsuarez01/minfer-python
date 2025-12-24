# TODO: complete 
# NOTE: compare torch default CUDA opns, and custom CUDA opns (eventually Triton)
import argparse
import gc

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer, Compare

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

def flash_attn():
    """flash_attn against pytorch SDPA across different sequence lengths."""

    kerns = KernelBackend()
    
    B, n_heads, n_kv_heads, head_dim = 4, 48, 8, 128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]
    
    seq_lens = [4096, 8192, 16384]
    results = []
    
    for L in seq_lens:
        print(f"\nRunning L={L}...")
        
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
        
        def pytorch_sdpa():
            result = F.scaled_dot_product_attention(Q, K2, V2, attn_mask=mask)
            torch.cuda.synchronize()
            return result
        
        for name, fn in [("my_flash_attn", my_flash_attn), ("pytorch_sdpa", pytorch_sdpa)]:
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
    flash_attn()