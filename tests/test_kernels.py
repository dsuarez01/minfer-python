# TODO: add me
import numpy as np
import torch
import torch.nn.functional as F
import pytest
from gguf import quants

from minfer.utils import LUT
from minfer.kernels import KernelBackend
from minfer.kernels.triton.kernels import _dequant as triton_dequant
from minfer.kernels.cuda.kernels import _dequant as cuda_dequant # type: ignore

@pytest.fixture(autouse=True)
def seed(request, randomly_seed):
    torch.manual_seed(randomly_seed)
    torch.cuda.manual_seed(randomly_seed)
    return randomly_seed

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# dequant needs to be tested but works a little differently
# it is not exposed in the KernelBackend interface
@pytest.mark.parametrize("backend,dequant", [("triton",triton_dequant),("cuda",cuda_dequant)])
@pytest.mark.parametrize("qtype", list(LUT.CFG.keys()))
@pytest.mark.parametrize("shape, block_m, block_n", [((128,128),16,16), ((128,128),1,128)])
def test_dequant(backend, dequant, qtype, shape, block_m, block_n):
    # see https://github.com/ggml-org/llama.cpp/tree/master/gguf-py/gguf

    data_A = np.random.randn(*shape).astype(np.float32)
    quantized_A = quants.quantize(data_A, qtype)
    expected_A = quants.dequantize(quantized_A, qtype)
    
    quantized_A = torch.from_numpy(quantized_A).cuda()
    actual_A = torch.zeros(shape, dtype=torch.float32).cuda()
    cfg = LUT.CFG[qtype]
    lut = torch.tensor(cfg.lut_vals, dtype=torch.float32).cuda()
    dequant(
        quantized_A, actual_A, lut,
        M=shape[0], N=shape[1],
        block_size=cfg.block_size, n_bits=cfg.nbits,
        packed_size=cfg.packed_nbytes, scale_size=cfg.scale_nbytes,
        has_zp=cfg.has_zp,
        BLOCK_M=block_m, BLOCK_N=block_n
    )
    actual_A = actual_A.cpu().numpy()
    assert np.allclose(actual_A, expected_A, rtol=1e-2, atol=1e-3)

## NOTE: for the rest of the tests FP16 dtype tensors are used (as appropriate)
## since dequant already tests the relevant usage patterns in the other kernels
## dp_size is used throughout but not actually factored in since it doesn't affect kernel usage

# A: [B, L, hidden_dim]
# B: [B, n_heads, L, head_dim]
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_rmsnorm(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, head_dim, eps = 8, 4096, 6144, 48, 128, 1e-6 # adjust as needed
    
    # test for rmsnorm applied across entire vector (act. shape [B // dp_size, L, hidden_dim], weight shape [hidden_dim,])
    input_A = torch.randn((B*L,hidden_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros_like(input_A)
    weight_A = torch.randn((hidden_dim), dtype=torch.float16).cuda()
    expected_A = F.rms_norm(input=input_A, weight=weight_A, normalized_shape=(hidden_dim,),  eps=eps)
    # TODO: add rmsnorm call on actual_A here
    assert torch.allclose(expected_A, actual_A), "rmsnorm: over entire vec"

    # test for rmsnorm applied across heads (act. shape [B // dp_size, n_heads, L, head_dim], weight shape [head_dim,])
    input_B = torch.randn((B*n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_B = torch.zeros_like(input_B)
    weight_B = torch.randn((head_dim,), dtype=torch.float16).cuda()
    expected_B = F.rms_norm(input=input_B, weight=weight_B, normalized_shape=(head_dim,),  eps=eps)
    # TODO: add rmsnorm call on actual_B here
    assert torch.allclose(expected_B.cpu(), actual_B.cpu()), "rmsnorm: per head"

    # output act. identical shape to input in both cases

# A: interleaved rope
# B: neox rope
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_rope(backend):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, rotary_dim, base_freq = 8, 4096, 48, 128, 64, 1e6 # adjust as needed

    # test for IL rope (act. shape [B // dp_size, n_heads, L, head_dim])
    input_A = torch.randn((B*n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros_like(input_A).cuda()

    # computing expected case (interleaved)
    freqs_A = torch.arange(head_dim // 2, dtype=torch.float16).cuda()
    freqs_A = torch.where(freqs_A < rotary_dim // 2, freqs_A, 0.0)
    freqs_A = base_freq ** (-2.0 * freqs_A / rotary_dim)
    freqs_A = torch.arange(L, dtype=torch.float16)[:, None].cuda() * freqs_A[None,:]
    freqs_A = torch.polar(torch.ones_like(freqs_A), freqs_A)

    input_A = torch.view_as_complex(input_A.reshape(*input_A.shape[:-1], -1, 2))
    expected_A = torch.view_as_real(input_A * freqs_A).flatten(-2)

    # TODO: add rope call on actual_A here
    assert torch.allclose(expected_A, actual_A), "rope: interleaved"

    # test for neox rope (act. shape [B // dp_size, n_heads, L, head_dim])
    input_B = torch.randn((B*n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_B = torch.zeros_like(input_B).cuda()

    # computing expected case (neox)
    freqs_B = torch.arange(head_dim // 2, dtype=torch.float16).cuda()
    freqs_B = torch.where(freqs_B < rotary_dim // 2, freqs_B, 0.0)
    freqs_B = base_freq ** (-2.0 * freqs_B / rotary_dim)
    freqs_B = torch.arange(L, dtype=torch.float16)[:, None].cuda() * freqs_B[None,:]
    freqs_B = torch.polar(torch.ones_like(freqs_B), freqs_B)

    input_B = torch.view_as_complex(torch.stack([input_B[...,:head_dim//2], input_B[...,head_dim//2:]], dim=-1))
    expected_B = torch.view_as_real(input_B * freqs_B)
    expected_B = torch.cat([expected_B[...,0], expected_B[...,1]], dim=-1)

    # TODO: add rope call on actual_B here
    assert torch.allclose(expected_B.cpu(), actual_B.cpu()), "rope: neox"

    # output act. identical shape to input

# A: just the usual matmul
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_matmul(backend):
    kerns = KernelBackend(backend)
    B, L, in_dim, out_dim = 8, 4096, 6144, 16384

    # act. shape [B // dp_size, L, in_dim], weight shape [out_dim, in_dim]    
    input_A = torch.randn((B*L, in_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((B*L, out_dim), dtype=torch.float16).cuda()
    weight_A = torch.randn((out_dim, in_dim), dtype=torch.float16).cuda()

    expected_A = input_A @ weight_A.T
    # TODO: add matmul call on actual_A here
    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "matmul"

    # output act. shape [B // dp_size, L, out_dim]

# A: just the usual embed
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_embed(backend):
    kerns = KernelBackend(backend)
    B, L, vocab_size, hidden_dim = 8, 4096, 128_000, 6144

    # act. shape [B // dp_size, L], weight shape [vocab_size, hidden_dim]
    input_A = torch.randint(0, vocab_size, (B,L)).cuda()
    actual_A = torch.zeros((B, L, hidden_dim), dtype=torch.float16).cuda()
    weight_A = torch.randn((vocab_size, hidden_dim), dtype=torch.float16).cuda()

    expected_A = weight_A[input_A]
    # TODO: add embed call on actual_A here
    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "embed"
    
    # output act. shape [B // dp_size, L, hidden_dim]

# A: just the usual QKV projections
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_qkv(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, n_kv_heads, head_dim = 8, 4096, 6144, 48, 8, 128

    # act. shape [B // dp_size, L, hidden_dim], weight shape [q_dim or kv_dim, hidden_dim]
    input_A = torch.randn((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A_Q = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_A_K = torch.zeros((B,n_kv_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_A_V = torch.zeros((B,n_kv_heads,L,head_dim), dtype=torch.float16).cuda()
    
    weight_A_Q = torch.randn((n_heads*head_dim,hidden_dim), dtype=torch.float16).cuda()
    weight_A_K = torch.randn((n_kv_heads*head_dim,hidden_dim), dtype=torch.float16).cuda()
    weight_A_V = torch.randn((n_kv_heads*head_dim,hidden_dim), dtype=torch.float16).cuda()

    expected_A_Q = (input_A @ weight_A_Q.T).reshape(B, L, n_heads, head_dim).transpose(1,2)
    expected_A_K = (input_A @ weight_A_K.T).reshape(B, L, n_kv_heads, head_dim).transpose(1,2)
    expected_A_V = (input_A @ weight_A_V.T).reshape(B, L, n_kv_heads, head_dim).transpose(1,2)

    # TODO: add qkv call on actual_Q,K,V here

    assert torch.allclose(expected_A_Q.cpu(), actual_A_Q.cpu()), "qkv: Q"
    assert torch.allclose(expected_A_K.cpu(), actual_A_K.cpu()), "qkv: K"
    assert torch.allclose(expected_A_V.cpu(), actual_A_V.cpu()), "qkv: V"

    # output: Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]

# A: just the usual flash attention
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_flash_attn(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, n_kv_heads, head_dim = 8, 4096, 6144, 48, 8, 128

    # Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]
    input_A_Q = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    input_A_K = torch.zeros((B,n_kv_heads,L,head_dim), dtype=torch.float16).cuda()
    input_A_V = torch.zeros_like(input_A_K)

    actual_A = torch.zeros((B,L,hidden_dim), dtype=torch.float16).cuda()
    expected_A = F.scaled_dot_product_attention(input_A_Q, input_A_K, input_A_V, is_causal=True)

    # TODO: add flash_attn call on actual_A here

    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "flash_attn"
    
    # output act. shape [B // dp_size, L, hidden_dim]

# A: just the usual matmul + softmax before topk expert selection
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_moe_scoring(backend):
    kerns = KernelBackend(backend)
    B, L, n_experts, hidden_dim = 8, 4096, 8, 6144

    # performs matmul and fused (online) softmax
    # input act. shape [B // dp_size, L, hidden_dim]
    input_A = torch.zeros((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros((B,L,n_experts), dtype=torch.float16).cuda()
    weight_A = torch.randn((n_experts, hidden_dim), dtype=torch.float16).cuda()

    expected_A = F.softmax(input_A @ weight_A.T, dim=-1)

    # TODO: add moe_scoring call on actual_A here

    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "moe_scoring"

    # output act. shape [B // dp_size, L, n_experts]

# A: the usual ffn opn. per expert
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_ffn(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, mlp_dim = 8, 4096, 6144, 16384

    # performs SwiGLU then downproj.
    # input act. shape [B // dp_size, L, hidden_dim]
    input_A = torch.zeros((B,L,hidden_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros_like(input_A)

    weight_A_gate = torch.randn((mlp_dim, hidden_dim), dtype=torch.float16).cuda()
    weight_A_up = torch.randn((mlp_dim, hidden_dim), dtype=torch.float16).cuda()
    weight_A_down = torch.randn((hidden_dim, mlp_dim), dtype=torch.float16).cuda()

    expected_A = (F.silu((input_A @ weight_A_gate.T)) * (input_A @ weight_A_up.T)) @ weight_A_down.T

    # TODO: add ffn call on actual_A here

    assert torch.allclose(expected_A.cpu(), actual_A.cpu()), "ffn"

    # output act. shape [B // dp_size, L, hidden_dim]