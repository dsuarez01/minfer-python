# TODO: modify ALL tests to use RefKernelBackend once ready (for now, only test_dequant uses it)
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

@pytest.mark.parametrize("backend", ["torch_ext"]) # TODO: add triton once fixed (same applies to the rest of the kernels)
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
    
    kerns._quant(qtype, data_A, quantized_A, qblock_size, qtype_size)
    kerns._dequant(qtype, quantized_A, expected_A, qblock_size, qtype_size)
    
    # test dequant on GPU
    quantized_A = quantized_A.cuda()
    actual_A = torch.zeros(shape, dtype=torch.float32).cuda()
    
    grid = (M,)
    kerns._dequant(qtype, quantized_A, actual_A, qblock_size, qtype_size)
    
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
    weight_A = torch.randn((hidden_dim), dtype=torch.float16).cuda()
    expected_A = F.rms_norm(input=input_A, weight=weight_A, normalized_shape=(hidden_dim,),  eps=eps)
    kerns.rmsnorm(eps, input_A, actual_A, weight_A)
    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=1e-2), "rmsnorm: over entire vec"

    # test for rmsnorm applied across heads (act. shape [B // dp_size, n_heads, L, head_dim], weight shape [head_dim,])
    input_B = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_B = torch.zeros_like(input_B)
    weight_B = torch.randn((head_dim,), dtype=torch.float16).cuda()
    expected_B = F.rms_norm(input=input_B, weight=weight_B, normalized_shape=(head_dim,),  eps=eps)
    kerns.rmsnorm(eps, input_B, actual_B, weight_B)
    assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=1e-2), "rmsnorm: per head"

    # output act. identical shape to input in both cases

# A: interleaved rope
# B: neox rope
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_rope(backend):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6 # adjust as needed

    # test for IL rope (act. shape [B // dp_size, n_heads, L, head_dim])
    input_A = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).float().cuda()
    actual_A = input_A.half().clone()

    # computing expected case (interleaved)
    freqs_A = torch.arange(rotary_dim // 2, dtype=torch.float32).cuda()
    freqs_A = base_freq ** (-2.0 * freqs_A / rotary_dim)
    freqs_A = torch.arange(L, dtype=torch.float32)[:, None].cuda() * freqs_A[None,:]
    cos_A = torch.cos(freqs_A)
    sin_A = torch.sin(freqs_A)

    input_A = input_A.reshape(*input_A.shape[:-1], -1, 2)
    expected_A = input_A.clone()
    expected_A[..., :rotary_dim//2, 0] = cos_A * input_A[..., :rotary_dim//2, 0] - sin_A * input_A[..., :rotary_dim//2, 1]
    expected_A[..., :rotary_dim//2, 1] = sin_A * input_A[..., :rotary_dim//2, 0] + cos_A * input_A[..., :rotary_dim//2, 1]
    expected_A = expected_A.flatten(-2).half()

    kerns.il_rope(rotary_dim, 0, base_freq, actual_A)
    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=5e-3), "rope: interleaved"

    # test for neox rope (act. shape [B // dp_size, n_heads, L, head_dim])
    input_B = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).float().cuda()
    actual_B = input_B.half().clone()

    # computing expected case (neox)
    freqs_B = torch.arange(rotary_dim // 2, dtype=torch.float32).cuda()
    freqs_B = base_freq ** (-2.0 * freqs_B / rotary_dim)
    freqs_B = torch.arange(L, dtype=torch.float32)[:, None].cuda() * freqs_B[None,:]
    cos_B = torch.cos(freqs_B)
    sin_B = torch.sin(freqs_B)

    expected_B = input_B.clone()
    input_B = torch.stack([input_B[...,:rotary_dim//2], input_B[...,rotary_dim//2:rotary_dim]], dim=-1)
    expected_B[..., :rotary_dim//2] = cos_B * input_B[..., :rotary_dim//2, 0] - sin_B * input_B[..., :rotary_dim//2, 1]
    expected_B[..., rotary_dim//2:rotary_dim] = sin_B * input_B[..., :rotary_dim//2, 0] + cos_B * input_B[..., :rotary_dim//2, 1]
    expected_B = expected_B.half()

    kerns.neox_rope(rotary_dim, 0, base_freq, actual_B)
    assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=5e-3), "rope: neox"

    # output act. identical shape to input
    # TODO: need to add test where start pos is not zero

    # TODO: there's some precision issue here (large vals of angle=freq*pos), so had to manually adjust the rtol and atol values

# A: just the usual matmul
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_matmul(backend):
    kerns = KernelBackend(backend)
    B, L, in_dim, out_dim = 8, 4096, 6144, 16384
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # act. shape [B // dp_size, L, in_dim], weight shape [out_dim, in_dim]    
    input_A = torch.randn((B, L, in_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((B, L, out_dim), dtype=torch.float16).cuda()
    weight_A = torch.randn((out_dim, in_dim), dtype=torch.float16).cuda()

    expected_A = input_A @ weight_A.T
    kerns.matmul(qtype, qblock_size, qtype_size, input_A, actual_A, weight_A.view(torch.uint8))

    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=3e-1), "matmul"

    # output act. shape [B // dp_size, L, out_dim]

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
    kerns.embed(qtype, qblock_size, qtype_size, actual_A, input_A, weight_A.view(torch.uint8))
    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "embed"
    
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
        q_qtype, k_qtype, v_qtype, q_qblock_size, k_qblock_size, v_qblock_size,
        q_qtype_size, k_qtype_size, v_qtype_size, input_A, actual_A_Q, actual_A_K, actual_A_V,
        weight_A_Q.view(torch.uint8), weight_A_K.view(torch.uint8), weight_A_V.view(torch.uint8)
    )

    assert torch.allclose(expected_A_Q.cpu(), actual_A_Q.cpu(), atol=5e-3), "qkv: Q"
    assert torch.allclose(expected_A_K.cpu(), actual_A_K.cpu(), atol=5e-3), "qkv: K"
    assert torch.allclose(expected_A_V.cpu(), actual_A_V.cpu(), atol=5e-3), "qkv: V"

    # output: Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]

# A: just the usual flash attention
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_flash_attn(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, n_kv_heads, head_dim = 8, 4096, 6144, 48, 8, 128
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]
    # zeros and zeros_like should be randn or something right?
    input_A_Q = torch.randn((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    input_A_K = torch.randn((B,n_kv_heads,L,head_dim), dtype=torch.float16).cuda()
    input_A_V = torch.randn((B,n_kv_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((B,L,hidden_dim), dtype=torch.float16).cuda()

    expected_A = F.scaled_dot_product_attention(input_A_Q, input_A_K, input_A_V, is_causal=True)

    kerns.flash_attn(
        qtype, qblock_size, qtype_size, actual_A, input_A_Q, input_A_K.view(torch.uint8), input_A_V.view(torch.uint8)
    )
    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "flash_attn"
    
    # output act. shape [B // dp_size, L, hidden_dim]

# just the usual matmul + softmax before topk expert selection
# A: 8 exps
# B: 128 exps
@pytest.mark.parametrize("backend", ["torch_ext"])
def test_moe_scoring(backend):
    kerns = KernelBackend(backend)
    B, L, n_experts_A, n_experts_B, hidden_dim = 8, 4096, 8, 128, 6144
    qtype = GGMLQuantizationType.F16
    qblock_size, qtype_size = GGML_QUANT_SIZES[qtype]

    # performs matmul and fused (online) softmax
    # input act. shape [B // dp_size, L, hidden_dim]
    input_A = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((B,L,n_experts_A), dtype=torch.float16).cuda()
    weight_A = (1/hidden_dim**0.5) * torch.randn((n_experts_A, hidden_dim), dtype=torch.float16).cuda()
    expected_A = F.softmax(input_A @ weight_A.T, dim=-1)

    kerns.moe_scoring(qtype, qblock_size, qtype_size, input_A, actual_A, weight_A.view(torch.uint8))

    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=1e-3), "moe_scoring"

    input_B = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_B = torch.zeros((B,L,n_experts_B), dtype=torch.float16).cuda()
    weight_B = (1/hidden_dim**0.5) * torch.randn((n_experts_B, hidden_dim), dtype=torch.float16).cuda()

    expected_B = F.softmax(input_B @ weight_B.T, dim=-1)

    kerns.moe_scoring(qtype, qblock_size, qtype_size, input_B, actual_B, weight_B.view(torch.uint8))

    assert torch.allclose(expected_B.cpu(), actual_B.cpu(), atol=1e-3), "moe_scoring"

    # output act. shape [B // dp_size, L, n_experts]

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
        up_qtype, gate_qtype, down_qtype, up_qblock_size, gate_qblock_size, down_qblock_size,
        up_qtype_size, gate_qtype_size, down_qtype_size, input_A, actual_A, hb, hb2,
        ws_A_up.view(torch.uint8), ws_A_gate.view(torch.uint8), ws_A_down.view(torch.uint8)
    )

    assert torch.allclose(expected_A.cpu(), actual_A.cpu(), atol=2e-3), "ffn"

    # output act. shape [n_local_exps, B // dp_size, L, hidden_dim]