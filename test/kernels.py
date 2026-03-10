from itertools import product
import gc

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from minfer.kernels import refs
from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

torch.backends.cuda.matmul.allow_fp16_accumulation = True
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

SUPPORTED_QTYPES = [
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
    "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
    "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "TQ1_0", "TQ2_0", 
    "MXFP4"
]

@pytest.mark.parametrize("backend", ["torch_ext"])
@pytest.mark.parametrize("shape", [(1024,6144), (16384,6144)])
@pytest.mark.parametrize("qtype_name", SUPPORTED_QTYPES)
def test_dequant(backend, qtype_name, shape):

    kerns = KernelBackend(backend)

    M, N = shape
    qtype = GGMLQuantizationType[qtype_name]
    
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]
    bytes_per_row = (N//qblock_size)*qtype_size
    
    data = torch.randn(shape, dtype=torch.float32)
    
    # round-trip on CPU is ground truth
    quantized = torch.zeros((M, bytes_per_row), dtype=torch.uint8)
    expected = torch.zeros(shape, dtype=torch.float32)
    
    kerns._quant(qtype, qblock_size, qtype_size, data, quantized)
    kerns._dequant(qtype, qblock_size, qtype_size, quantized, expected)
    
    # test dequant on GPU
    quantized = quantized.cuda()
    actual = torch.zeros(shape, dtype=torch.float32).cuda()
    
    grid = (M,)
    kerns._dequant(qtype, qblock_size, qtype_size, quantized, actual)
    
    torch.cuda.synchronize()

    assert torch.allclose(actual.cpu(), expected)

    torch.cuda.empty_cache()

@pytest.mark.parametrize("backend", ["torch_ext"])
@pytest.mark.parametrize("shape", [
    ((8,4096,6144)), # per-vec
    ((8,4096,48,128)), # per-head
])
def test_rmsnorm(backend, shape):
    kerns = KernelBackend(backend)
    eps = 1e-6
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    out = torch.zeros_like(x)
    weight = (1/shape[-1]**0.5) * torch.randn(shape[-1], dtype=torch.float16, device="cuda")
    expected = refs.rmsnorm(x, weight, eps)
    kerns.rmsnorm(eps, x, weight, out)
    torch.cuda.synchronize()
    assert torch.allclose(expected.cpu(), out.cpu(), atol=1e-3)
    torch.cuda.empty_cache()

@pytest.mark.parametrize("backend", ["torch_ext"])
@pytest.mark.parametrize("start_pos", [0, 2048])
@pytest.mark.parametrize("rope_type", ["il", "neox"])
def test_rope(backend, rope_type, start_pos):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6

    x = torch.randn((B, L, n_heads, head_dim), dtype=torch.float16, device="cuda")
    actual = x.clone()

    if rope_type == "il":
        expected = refs.il_rope(x, rotary_dim, start_pos, base_freq)
        kerns.il_rope(rotary_dim, start_pos, base_freq, actual)
    else:
        expected = refs.neox_rope(x, rotary_dim, start_pos, base_freq)
        kerns.neox_rope(rotary_dim, start_pos, base_freq, actual)

    torch.cuda.synchronize()
    # NOTE: atol adjusted for large angle vals (freq*pos)
    assert torch.allclose(expected.cpu(), actual.cpu(), atol=5e-3)
    torch.cuda.empty_cache()

@pytest.mark.parametrize("backend", ["torch_ext"])
def test_embed(backend):
    kerns = KernelBackend(backend)
    B, L, vocab_size, hidden_dim = 8, 4096, 128_000, 6144
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    token_ids = torch.randint(0, vocab_size, (B, L), device="cuda")
    weight = torch.randn((vocab_size, hidden_dim), dtype=torch.float16, device="cuda")
    actual = torch.zeros((B, L, hidden_dim), dtype=torch.float16, device="cuda")

    expected = weight[token_ids]
    kerns.embed(qtype, qblock_size, qtype_size, token_ids, weight, actual)
    torch.cuda.synchronize()
    assert torch.allclose(expected.cpu(), actual.cpu())
    torch.cuda.empty_cache()

# A: alpha * AB + beta * C
# B: alpha * AB^T + beta * C (NOTE: in progress)
SCALES = [(1.0,0.0),(1.0,1.0),(2.0,3.0),(0.5,2.0)]
MATMUL_SIZES = [512,1024,2048,4096,8192,16384]
@pytest.mark.parametrize("backend", ["torch_ext"])
@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("shape", [(s, s, s) for s in MATMUL_SIZES])
# @pytest.mark.parametrize("shape", product(MATMUL_SIZES,MATMUL_SIZES,MATMUL_SIZES))
def test_gemm(backend, shape, scales):

    kerns = KernelBackend(backend)
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    M, K, N = shape
    alpha, beta = scales

    # A: act. shape [B // dp_size, L, in_dim], weight shape [in_dim, out_dim]
    input_A = torch.randn((1, M, K), dtype=torch.float16, device="cuda")
    weight_A = (1/K**0.5) * torch.randn((K, N), dtype=torch.float16, device="cuda")
    bias_A = torch.randn((1, M, N), dtype=torch.float16, device="cuda")
    actual_A = torch.zeros((1, M, N), dtype=torch.float16, device="cuda")

    expected_A = alpha*(input_A@weight_A) + beta*bias_A
    with torch.no_grad():
        kerns.gemm(qtype, qblock_size, qtype_size, alpha, beta, input_A, weight_A, bias_A, actual_A)
    torch.cuda.synchronize()

    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=6e-1), "tests for case A, gemm"

    torch.cuda.empty_cache()

    # # output act. shape [B // dp_size, L, out_dim]

@pytest.mark.parametrize("backend", ["torch_ext"])
@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("shape", [(s, s, s) for s in MATMUL_SIZES])
# @pytest.mark.parametrize("shape", product(MATMUL_SIZES,MATMUL_SIZES,MATMUL_SIZES))
def test_gemm_opcheck(backend, shape, scales):

    M,K,N = shape
    alpha, beta = scales

    # kerns = KernelBackend(backend)
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]
    
    input = torch.randn((1,M,K), dtype=torch.float16, device='cuda')
    weight = (1/K**0.5) * torch.randn((K,N), dtype=torch.float16, device='cuda')
    bias = torch.randn((1,M,N), dtype=torch.float16, device='cuda')
    out = torch.zeros((1,M,N), dtype=torch.float16, device='cuda')
    
    torch.library.opcheck(
        torch.ops.minfer.gemm.default,
        (qtype, qblock_size, qtype_size, alpha, beta, input, weight, bias, out)
    )


# A: just the usual QKV projections
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_qkv(backend):

    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, n_kv_heads, head_dim = 8, 4096, 6144, 48, 8, 128
    q_qtype = GGMLQuantizationType.F16
    q_qblock_size, q_qtype_size = GGML_QUANT_SIZES[q_qtype]
    k_qtype = GGMLQuantizationType.F16
    k_qblock_size, k_qtype_size = GGML_QUANT_SIZES[k_qtype]
    v_qtype = GGMLQuantizationType.F16
    v_qblock_size, v_qtype_size = GGML_QUANT_SIZES[v_qtype]

    # input act. shape [B // dp_size, L, hidden_dim], 
    # output act. shape [B // dp_size, L, q_dim or kv_dim] 
    # weight shape [q_dim or kv_dim, hidden_dim]
    input_A = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A_Q = torch.zeros((B,L,n_heads*head_dim), dtype=torch.float16).cuda()
    actual_A_K = torch.zeros((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
    actual_A_V = torch.zeros((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
    
    weight_A_Q = (1/hidden_dim**0.5) * torch.randn((n_heads*head_dim,hidden_dim), dtype=torch.float16).cuda()
    weight_A_K = (1/hidden_dim**0.5) * torch.randn((n_kv_heads*head_dim,hidden_dim), dtype=torch.float16).cuda()
    weight_A_V = (1/hidden_dim**0.5) * torch.randn((n_kv_heads*head_dim,hidden_dim), dtype=torch.float16).cuda()

    expected_A_Q = (input_A @ weight_A_Q.T)
    expected_A_K = (input_A @ weight_A_K.T)
    expected_A_V = (input_A @ weight_A_V.T)

    kerns.qkv(
        q_qtype, k_qtype, v_qtype, 
        q_qblock_size, k_qblock_size, v_qblock_size,
        q_qtype_size, k_qtype_size, v_qtype_size, 
        input_A, weight_A_Q, weight_A_K, weight_A_V,
        actual_A_Q, actual_A_K, actual_A_V,
    )

    torch.cuda.synchronize()

    assert torch.allclose(expected_A_Q.cpu(), actual_A_Q.cpu(), atol=2e-1), "qkv: Q"
    assert torch.allclose(expected_A_K.cpu(), actual_A_K.cpu(), atol=2e-1), "qkv: K"
    assert torch.allclose(expected_A_V.cpu(), actual_A_V.cpu(), atol=2e-1), "qkv: V"

    torch.cuda.empty_cache()

    # output: Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]

# A: causal mask
# B: sliding window mask
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_flash_attn(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, n_kv_heads, head_dim = 8, 4096, 6144, 48, 8, 128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # expected Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]
    input_A_Q = torch.randn((B,L,n_heads*head_dim), dtype=torch.float16).cuda()
    input_A_Q = input_A_Q.view((B,L,n_heads,head_dim)).transpose(1,2)
    
    input_A_K = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
    input_A_K = input_A_K.view((B,L,n_kv_heads,head_dim)).transpose(1,2)
    
    input_A_V = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
    input_A_V = input_A_V.view((B,L,n_kv_heads,head_dim)).transpose(1,2)

    # NOTE: weird bug with torch tril and triu on CUDA, see e.g. https://github.com/pytorch/pytorch/issues/136611
    # mask_A = torch.ones(B, n_heads, L, L, dtype=torch.uint8).tril()
    # mask_A = mask_A.cuda()
    mask_A = (torch.arange(L).unsqueeze(0) <= torch.arange(L).unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool).cuda()

    actual_A = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()

    kerns.flash_attn(
        qtype, qblock_size, qtype_size, input_A_Q, input_A_K, input_A_V, mask_A, actual_A,
    )
    
    torch.cuda.synchronize()

    # pytorch SDPA doesn't handle GQA on its own
    n_rep = n_heads // n_kv_heads
    input_A_K = input_A_K.repeat_interleave(n_rep, dim=1)
    input_A_V = input_A_V.repeat_interleave(n_rep, dim=1)

    expected_A = F.scaled_dot_product_attention(input_A_Q, input_A_K, input_A_V, attn_mask=mask_A)
    
    print((actual_A - expected_A).abs().max())

    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=3e-1), "flash_attn"

    # expected Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]
    input_B_Q = torch.randn((B,L,n_heads*head_dim), dtype=torch.float16).cuda()
    input_B_Q = input_B_Q.view((B,L,n_heads,head_dim)).transpose(1,2)
    
    input_B_K = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
    input_B_K = input_B_K.view((B,L,n_kv_heads,head_dim)).transpose(1,2)
    
    input_B_V = (1/head_dim**0.5) * torch.randn((B,L,n_kv_heads*head_dim), dtype=torch.float16).cuda()
    input_B_V = input_B_V.view((B,L,n_kv_heads,head_dim)).transpose(1,2)

    # NOTE: weird bug with torch tril and triu on CUDA, see e.g. https://github.com/pytorch/pytorch/issues/136611
    # mask_B = (torch.tril(torch.ones(L, L), window_size-1) * torch.triu(torch.ones(L, L), -(window_size-1))).bool()
    # mask_B = mask_B.unsqueeze(0).unsqueeze(0)
    window_size = 256
    dist = torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
    mask_B = (dist.abs() < window_size).unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).to(torch.bool).cuda()

    actual_B = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()

    kerns.flash_attn(
        qtype, qblock_size, qtype_size, input_B_Q, input_B_K, input_B_V, mask_B, actual_B,
    )

    torch.cuda.synchronize()

    # pytorch SDPA doesn't handle GQA on its own
    n_rep = n_heads // n_kv_heads
    input_B_K = input_B_K.repeat_interleave(n_rep, dim=1)
    input_B_V = input_B_V.repeat_interleave(n_rep, dim=1)

    expected_B = F.scaled_dot_product_attention(input_B_Q, input_B_K, input_B_V, attn_mask=mask_B)

    assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=3e-1), "flash_attn"

    torch.cuda.empty_cache()
    # output act. shape [B // dp_size, L, hidden_dim]

# just the usual matmul + softmax before topk expert selection
# A: 8 exps
# B: 128 exps
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_moe_scoring(backend):
    kerns = KernelBackend(backend)
    B, L, n_experts_A, n_experts_B, n_act_exps_A, n_act_exps_B, hidden_dim = 8, 4096, 8, 128, 2, 8, 6144
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # performs matmul and fused (online) softmax
    # input act. shape [B // dp_size, L, hidden_dim]
    input_A = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()

    actual_scores_A = torch.zeros((B,L,n_experts_A), dtype=torch.float16).cuda()
    actual_topk_scores_A = torch.zeros((B,L,n_act_exps_A), dtype=torch.float16).cuda()
    actual_topk_indices_A = torch.zeros((B,L,n_act_exps_A), dtype=torch.uint8).cuda()

    weight_A = (1/hidden_dim**0.5) * torch.randn((n_experts_A, hidden_dim), dtype=torch.float16).cuda()
    expected_scores_A = F.softmax(input_A @ weight_A.T, dim=-1)
    expected_topk_scores_A, _ = torch.topk(expected_scores_A, k=n_act_exps_A, dim=-1)

    kerns.moe_scoring(qtype, qblock_size, qtype_size, input_A, weight_A, actual_topk_indices_A, actual_topk_scores_A, actual_scores_A)
    torch.cuda.synchronize()
    gathered_scores_A = torch.gather(actual_scores_A, dim=-1, index=actual_topk_indices_A.to(torch.int64))

    assert torch.allclose(expected_scores_A.cpu(), actual_scores_A.cpu(), atol=1e-3), "moe_scoring, scores"
    assert torch.allclose(expected_topk_scores_A.cpu(), actual_topk_scores_A.cpu(), atol=1e-3), "moe_scoring, topk scores"

    input_B = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()

    actual_scores_B = torch.zeros((B,L,n_experts_B), dtype=torch.float16).cuda()
    actual_topk_scores_B = torch.zeros((B,L,n_act_exps_B), dtype=torch.float16).cuda()
    actual_topk_indices_B = torch.zeros((B,L,n_act_exps_B), dtype=torch.uint8).cuda()

    weight_B = (1/hidden_dim**0.5) * torch.randn((n_experts_B, hidden_dim), dtype=torch.float16).cuda()
    expected_scores_B = F.softmax(input_B @ weight_B.T, dim=-1)
    expected_topk_scores_B, _ = torch.topk(expected_scores_B, k=n_act_exps_B, dim=-1)

    kerns.moe_scoring(qtype, qblock_size, qtype_size, input_B, weight_B, actual_topk_indices_B, actual_topk_scores_B, actual_scores_B)
    torch.cuda.synchronize()
    assert torch.allclose(expected_scores_B.cpu(), actual_scores_B.cpu(), atol=1e-3), "moe_scoring, scores"
    assert torch.allclose(expected_topk_scores_B.cpu(), actual_topk_scores_B.cpu(), atol=1e-3), "moe_scoring, topk scores"

    torch.cuda.empty_cache()

# A: the usual ffn opn.
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_ffn(backend):
    kerns = KernelBackend(backend)
    B, L, n_local_exps, hidden_dim, mlp_dim = 8, 4096, 2, 6144, 16384
    up_qtype = GGMLQuantizationType.F16
    up_qblock_size, up_qtype_size = GGML_QUANT_SIZES[up_qtype]
    gate_qtype = GGMLQuantizationType.F16
    gate_qblock_size, gate_qtype_size = GGML_QUANT_SIZES[gate_qtype]
    down_qtype = GGMLQuantizationType.F16
    down_qblock_size, down_qtype_size = GGML_QUANT_SIZES[down_qtype]

    # performs SwiGLU then downproj.
    # input act. shape [B // dp_size, L, hidden_dim]
    input_A = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((n_local_exps,B,L,hidden_dim), dtype=torch.float16).cuda()

    hb = torch.zeros((n_local_exps,B,L,mlp_dim), dtype=torch.float16).cuda()
    hb2 = torch.zeros_like(hb)

    # reduces variance of linear proj outputs to 1, limits overflow
    ws_A_gate = (1/hidden_dim**0.5) * torch.randn((n_local_exps, mlp_dim, hidden_dim), dtype=torch.float16).cuda()
    ws_A_up = (1/hidden_dim**0.5) * torch.randn((n_local_exps, mlp_dim, hidden_dim), dtype=torch.float16).cuda()
    ws_A_down = (1/mlp_dim**0.5) * torch.randn((n_local_exps, hidden_dim, mlp_dim), dtype=torch.float16).cuda()

    gate_out = torch.einsum("blh,emh->eblm", input_A, ws_A_gate)
    up_out = torch.einsum("blh,emh->eblm", input_A, ws_A_up)
    hidden = F.silu(gate_out) * up_out
    
    expected_A = torch.einsum('eblm,ehm->eblh', hidden, ws_A_down)

    kerns.ffn(
        up_qtype, gate_qtype, down_qtype, 
        up_qblock_size, gate_qblock_size, down_qblock_size,
        up_qtype_size, gate_qtype_size, down_qtype_size, 
        input_A, ws_A_up, ws_A_gate, ws_A_down,
        hb, hb2, actual_A,
    )

    torch.cuda.synchronize()

    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=2e-1), "ffn"

    torch.cuda.empty_cache()

    # output act. shape [n_local_exps, B // dp_size, L, hidden_dim]