# TODO: complete
import torch

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

def rmsnorm_head(backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, L, eps, n_heads, head_dim = 8, 4096, 1e-6, 48, 128
    
    x = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16).cuda()
    out = torch.zeros_like(x).cuda()
    weight = (1/head_dim**0.5) * torch.randn(head_dim, dtype=torch.float16).cuda()
    
    kerns.rmsnorm(eps, x, out, weight)
    torch.cuda.synchronize()

def rmsnorm_vec(backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, L, eps, hidden_dim = 8, 4096, 1e-6, 6144
    
    x = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    out = torch.zeros_like(x).cuda()
    weight = (1/hidden_dim**0.5) * torch.randn(hidden_dim, dtype=torch.float16).cuda()
    
    kerns.rmsnorm(eps, x, weight, out)
    torch.cuda.synchronize()

def rope(backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6
    
    x = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    kerns.il_rope(rotary_dim, 0, base_freq, x)
    torch.cuda.synchronize()
    
    x = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    kerns.neox_rope(rotary_dim, 0, base_freq, x)
    torch.cuda.synchronize()

def matmul(shape: tuple, backend: str = "torch_ext"):
    kerns = KernelBackend(backend)
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    M, K, N = shape

    x = torch.randn((M,K), dtype=torch.float16).cuda()
    out = torch.zeros((M,N), dtype=torch.float16).cuda()
    weight = (1/K**0.5) * torch.randn((K,N), dtype=torch.float16).cuda()
    
    kerns.matmul(qtype, qblock_size, qtype_size, x.unsqueeze(0), weight, out.unsqueeze(0))
    torch.cuda.synchronize()

def flash_attn(backend:str = "torch_ext"):
    kerns = KernelBackend(backend)
    B, L, n_heads, n_kv_heads, head_dim = 4,4096,48,8,128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    Q = torch.randn((B,L,n_heads*head_dim), dtype=torch.float16).cuda().view(B,L,n_heads,head_dim).transpose(1,2)
    K = torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda().view(B,L,n_kv_heads,head_dim).transpose(1,2)
    V = torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda().view(B,L,n_kv_heads,head_dim).transpose(1,2)
    mask = (torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool).cuda()
    out = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()

    kerns.flash_attn(qtype, qblock_size, qtype_size, Q, K, V, mask, out)
    torch.cuda.synchronize()

if __name__ == "__main__":
    # rmsnorm_head()
    # rmsnorm_vec()
    # rope()
    matmul(shape=(4096,4096,4096))
    # flash_attn()