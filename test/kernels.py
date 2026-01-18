# TODO: eventually add in Triton?
import numpy as np
import torch
import torch.nn.functional as F
import pytest

from minfer.kernels import KernelBackend
from minfer.const import GGMLQuantizationType, GGML_QUANT_SIZES

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

SUPPORTED_QTYPES = [
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K",
    "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
    "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "BF16",
    "TQ1_0", "TQ2_0", 
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
    
    data_A = torch.randn(shape, dtype=torch.float32)
    
    # round-trip on CPU is ground truth
    quantized_A = torch.zeros((M, bytes_per_row), dtype=torch.uint8)
    expected_A = torch.zeros(shape, dtype=torch.float32)
    
    kerns._quant(qtype, qblock_size, qtype_size, data_A, quantized_A)
    kerns._dequant(qtype, qblock_size, qtype_size, quantized_A, expected_A)
    
    # test dequant on GPU
    quantized_A = quantized_A.cuda()
    actual_A = torch.zeros(shape, dtype=torch.float32).cuda()
    
    grid = (M,)
    kerns._dequant(qtype, qblock_size, qtype_size, quantized_A, actual_A)
    
    torch.cuda.synchronize()

    assert torch.allclose(actual_A.cpu(), expected_A)

    torch.cuda.empty_cache()

## NOTE: for the rest of the tests FP16 dtype tensors are used (as appropriate)
## since dequant already tests the relevant usage patterns in the other kernels
## dp_size is used throughout but not actually factored in since it doesn't affect kernel usage

# A: [B , L, hidden_dim]
# B: [B, n_heads, L, head_dim]
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_rmsnorm(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, head_dim, eps = 8, 4096, 6144, 48, 128, 1e-6 # adjust as needed
    
    # test for rmsnorm applied across entire vector (act. shape [B // dp_size, L, hidden_dim], weight shape [hidden_dim,])
    input_A = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros_like(input_A)
    weight_A = (1/hidden_dim**0.5) * torch.randn((hidden_dim), dtype=torch.float16).cuda()
    expected_A = F.rms_norm(input=input_A, weight=weight_A, normalized_shape=(hidden_dim,),  eps=eps)
    kerns.rmsnorm(eps, input_A, weight_A, actual_A)
    torch.cuda.synchronize()
    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=1e-3), "rmsnorm: over entire vec"

    # test for rmsnorm applied across heads (act. shape [B // dp_size, n_heads, L, head_dim], weight shape [head_dim,])
    input_B = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_B = torch.zeros_like(input_B)
    weight_B = (1/head_dim**0.5) * torch.randn((head_dim,), dtype=torch.float16).cuda()
    expected_B = F.rms_norm(input=input_B, weight=weight_B, normalized_shape=(head_dim,),  eps=eps)
    kerns.rmsnorm(eps, input_B, weight_B, actual_B)
    torch.cuda.synchronize()
    assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=1e-3), "rmsnorm: per head"

    torch.cuda.empty_cache()
    # output act. identical shape to input in both cases

# A: interleaved rope
# B: neox rope
# TODO: change this test to operate on [B,L,n_heads,head_dim]
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_rope(backend):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6 # adjust as needed

    # test for IL rope (act. shape [B // dp_size, L, n_heads, head_dim])
    input_A = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16).float().cuda()
    actual_A = input_A.half().clone()

    # computing expected case (interleaved)
    freqs_A = torch.arange(rotary_dim // 2, dtype=torch.float32).cuda()
    freqs_A = base_freq ** (-2.0 * freqs_A / rotary_dim)
    freqs_A = torch.arange(L, dtype=torch.float32)[:, None].cuda() * freqs_A[None,:]
    cos_A = torch.cos(freqs_A)
    sin_A = torch.sin(freqs_A)

    cos_A = cos_A[None, :, None, :]
    sin_A = sin_A[None, :, None, :]

    input_A = input_A.reshape(*input_A.shape[:-1], -1, 2)
    expected_A = input_A.clone()
    expected_A[..., :rotary_dim//2, 0] = cos_A * input_A[..., :rotary_dim//2, 0] - sin_A * input_A[..., :rotary_dim//2, 1]
    expected_A[..., :rotary_dim//2, 1] = sin_A * input_A[..., :rotary_dim//2, 0] + cos_A * input_A[..., :rotary_dim//2, 1]
    expected_A = expected_A.flatten(-2).half()

    kerns.il_rope(rotary_dim, 0, base_freq, actual_A)
    torch.cuda.synchronize()
    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=5e-3), "rope: interleaved"

    # test for neox rope (act. shape [B // dp_size, L, n_heads, head_dim])
    input_B = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16).float().cuda()
    actual_B = input_B.half().clone()

    # computing expected case (neox)
    freqs_B = torch.arange(rotary_dim // 2, dtype=torch.float32).cuda()
    freqs_B = base_freq ** (-2.0 * freqs_B / rotary_dim)
    freqs_B = torch.arange(L, dtype=torch.float32)[:, None].cuda() * freqs_B[None,:]
    cos_B = torch.cos(freqs_B)
    sin_B = torch.sin(freqs_B)

    cos_B = cos_B[None, :, None, :]
    sin_B = sin_B[None, :, None, :]

    expected_B = input_B.clone()
    input_B = torch.stack([input_B[...,:rotary_dim//2], input_B[...,rotary_dim//2:rotary_dim]], dim=-1)
    expected_B[..., :rotary_dim//2] = cos_B * input_B[..., :rotary_dim//2, 0] - sin_B * input_B[..., :rotary_dim//2, 1]
    expected_B[..., rotary_dim//2:rotary_dim] = sin_B * input_B[..., :rotary_dim//2, 0] + cos_B * input_B[..., :rotary_dim//2, 1]
    expected_B = expected_B.half()

    kerns.neox_rope(rotary_dim, 0, base_freq, actual_B)
    torch.cuda.synchronize()
    assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=5e-3), "rope: neox"

    # test with non-zero start_pos
    start_pos = 2048
    input_C = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16).float().cuda()
    actual_C = input_C.half().clone()

    freqs_C = torch.arange(rotary_dim // 2, dtype=torch.float32).cuda()
    freqs_C = base_freq ** (-2.0 * freqs_C / rotary_dim)
    freqs_C = (torch.arange(L, dtype=torch.float32).cuda() + start_pos)[:, None] * freqs_C[None,:]
    cos_C = torch.cos(freqs_C)
    sin_C = torch.sin(freqs_C)

    cos_C = cos_C[None, :, None, :]
    sin_C = sin_C[None, :, None, :]

    input_C = input_C.reshape(*input_C.shape[:-1], -1, 2)
    expected_C = input_C.clone()
    expected_C[..., :rotary_dim//2, 0] = cos_C * input_C[..., :rotary_dim//2, 0] - sin_C * input_C[..., :rotary_dim//2, 1]
    expected_C[..., :rotary_dim//2, 1] = sin_C * input_C[..., :rotary_dim//2, 0] + cos_C * input_C[..., :rotary_dim//2, 1]
    expected_C = expected_C.flatten(-2).half()

    kerns.il_rope(rotary_dim, start_pos, base_freq, actual_C)
    torch.cuda.synchronize()
    assert torch.allclose(expected_C.cpu(), actual_C.cpu(), atol=5e-3), "rope: interleaved w/ start_pos"

    # test neox with non-zero start_pos
    input_D = torch.randn((B,L,n_heads,head_dim), dtype=torch.float16).float().cuda()
    actual_D = input_D.half().clone()

    freqs_D = torch.arange(rotary_dim // 2, dtype=torch.float32).cuda()
    freqs_D = base_freq ** (-2.0 * freqs_D / rotary_dim)
    freqs_D = (torch.arange(L, dtype=torch.float32).cuda() + start_pos)[:, None] * freqs_D[None,:]
    cos_D = torch.cos(freqs_D)
    sin_D = torch.sin(freqs_D)

    cos_D = cos_D[None, :, None, :]
    sin_D = sin_D[None, :, None, :]

    expected_D = input_D.clone()
    input_D = torch.stack([input_D[...,:rotary_dim//2], input_D[...,rotary_dim//2:rotary_dim]], dim=-1)
    expected_D[..., :rotary_dim//2] = cos_D * input_D[..., :rotary_dim//2, 0] - sin_D * input_D[..., :rotary_dim//2, 1]
    expected_D[..., rotary_dim//2:rotary_dim] = sin_D * input_D[..., :rotary_dim//2, 0] + cos_D * input_D[..., :rotary_dim//2, 1]
    expected_D = expected_D.half()

    kerns.neox_rope(rotary_dim, start_pos, base_freq, actual_D)
    torch.cuda.synchronize()
    assert torch.allclose(expected_D.cpu(), actual_D.cpu(), atol=5e-3), "rope: neox w/ start_pos"

    torch.cuda.empty_cache()

    # output act. identical shape to input
    # TODO: there's some precision issue here (large vals of angle=freq*pos), so had to manually adjust the rtol and atol values

# A: X @ W
# B: X @ W.T (NOTE: not finished right now)
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_matmul(backend):
    kerns = KernelBackend(backend)
    B, L, in_dim, out_dim = 8, 4096, 6144, 16384
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # A: act. shape [B // dp_size, L, in_dim], weight shape [in_dim, out_dim]
    input_A = torch.randn((B, L, in_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((B, L, out_dim), dtype=torch.float16).cuda()
    weight_A = (1/in_dim**0.5) * torch.randn((in_dim, out_dim), dtype=torch.float16).cuda()

    expected_A = input_A @ weight_A
    kerns.matmul(qtype, qblock_size, qtype_size, input_A, weight_A, actual_A)
    torch.cuda.synchronize()
    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=2e-1), "matmul"

    # # act. shape [B // dp_size, L, in_dim], weight shape [out_dim, in_dim]    
    # input_B = torch.randn((B, L, in_dim), dtype=torch.float16).cuda()
    # actual_B = torch.zeros((B, L, out_dim), dtype=torch.float16).cuda()
    # weight_B = (1/in_dim**0.5) * torch.randn((in_dim, out_dim), dtype=torch.float16).cuda()

    # expected_B = input_B @ weight_B
    # kerns.matmul(qtype, qblock_size, qtype_size, input_B, weight_B, actual_B)
    # torch.cuda.synchronize()
    # assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=2e-1), "matmul"

    # torch.cuda.empty_cache()

    # # output act. shape [B // dp_size, L, out_dim]

# A: just the usual embed
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_embed(backend):
    kerns = KernelBackend(backend)
    B, L, vocab_size, hidden_dim = 8, 4096, 128_000, 6144
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # act. shape [B // dp_size, L], weight shape [vocab_size, hidden_dim]
    input_A = torch.randint(0, vocab_size, (B,L)).cuda()
    actual_A = torch.zeros((B, L, hidden_dim), dtype=torch.float16).cuda()
    weight_A = torch.randn((vocab_size, hidden_dim), dtype=torch.float16).cuda()

    expected_A = weight_A[input_A]
    kerns.embed(qtype, qblock_size, qtype_size, input_A, weight_A, actual_A)
    torch.cuda.synchronize()
    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "embed"
    
    torch.cuda.empty_cache()
    # output act. shape [B // dp_size, L, hidden_dim]

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