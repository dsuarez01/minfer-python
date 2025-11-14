# TODO: add me
import torch
import torch.nn.functional as F
import pytest

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
@pytest.mark.parametrize("backend,dequant", [("triton", triton_dequant),("cuda", cuda_dequant)])
def test_dequant(backend, dequant):

    # need to add tests here for every quantized dtype in the lookup table? (see minfer.utils)
    # in gguf.quants, there are many useful fcns/methods: might be useful to compare against
    # e.g. (see https://github.com/ggml-org/llama.cpp/tree/master/gguf-py/gguf)
    """
    def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        if qtype == GGMLQuantizationType.F32:
            return data.astype(np.float32, copy=False)
        elif qtype == GGMLQuantizationType.F16:
            return data.astype(np.float16, copy=False)
        elif (q := _type_traits.get(qtype)) is not None:
            return q.quantize(data)
        else:
            raise NotImplementedError(f"Quantization for {qtype.name} is not yet implemented")
    
    def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        if qtype == GGMLQuantizationType.F32:
            return data.view(np.float32)
        elif qtype == GGMLQuantizationType.F16:
            return data.view(np.float16).astype(np.float32)
        elif (q := _type_traits.get(qtype)) is not None:
            return q.dequantize(data)
        else:
            raise NotImplementedError(f"Dequantization for {qtype.name} is not yet implemented")

    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))

    @classmethod
    def dequantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))

    """
    # round-trip test: 
    # - start with numpy array (FP32 and/or FP16)
    # - quantize it using gguf-py
    # - convert to torch tensor (underlying np dtype is uint8 for all of these)
    # - dequantize using kernel, convert back to np array (FP32 and/or FP16)
    # - compare, see if same
    # should be possible to do this for blocks + rows, both (should be) supported by kernel

    return

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
    assert torch.allclose(expected_B, actual_B), "rmsnorm: per head"

    # output act. identical shape to input in both cases

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_rope(backend):
    kerns = KernelBackend(backend)
    B, L, n_heads, head_dim, base_freq = 8, 4096, 48, 128, 1e6 # adjust as needed

    # test for rope applied across heads (act. shape [B // dp_size, n_heads, L, head_dim])
    input_A = torch.randn((B*n_heads,L,head_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros_like(input_A).cuda()

    # computing expected case (interleaved)
    freqs_A = torch.arange(head_dim // 2, dtype=torch.float16).cuda()
    freqs_A = base_freq ** (-2.0 * freqs_A / head_dim)
    freqs_A = torch.arange(L, dtype=torch.float16)[:, None].cuda() * freqs_A[None,:]
    
    input_A = torch.view_as_complex(input_A.reshape(*input_A.shape[:-1], -1, 2))
    freqs_A = torch.view_as_complex(freqs_A)
    expected_A = torch.view_as_real(input_A * freqs_A).flatten(-2)

    # TODO: add rope call on actual_A here
    assert torch.allclose(expected_A, actual_A), "rope: interleaved"

    # TODO: support neox/two-halves later too

    # output act. identical shape to input

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_matmul(backend):
    kerns = KernelBackend(backend)
    B, L, in_dim, out_dim = 8, 4096, 6144, 16384

    # act. shape [B // dp_size, L, in_dim], weight shape [out_dim, in_dim]    
    input_A = torch.randn((B*L, in_dim), dtype=torch.float16).cuda()
    actual_A = torch.zeros_like(input_A)
    weight_A = torch.randn((out_dim, in_dim), dtype=torch.float16).cuda()

    expected_A = input_A @ weight_A.T
    # TODO: add matmul call on actual_A here
    assert torch.allclose(expected_A, actual_A), "matmul"

    # output act. shape [B // dp_size, L, out_dim]

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
    assert torch.allclose(expected_A, actual_A), "embed"
    
    # output act. shape [B // dp_size, L, hidden_dim]

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

    assert torch.allclose(expected_A_Q, actual_A_Q), "qkv: Q"
    assert torch.allclose(expected_A_K, actual_A_K), "qkv: K"
    assert torch.allclose(expected_A_V, actual_A_V), "qkv: V"

    # output: Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_flash_attn(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim, n_heads, n_kv_heads, head_dim = 8, 4096, 6144, 48, 8, 128

    # Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]
    input_A_Q = torch.zeros((B,n_heads,L,head_dim), dtype=torch.float16).cuda()
    input_A_K = torch.zeros((B,n_kv_heads,L,head_dim), dtype=torch.float16).cuda()
    input_A_V = torch.zeros_like(input_A_K)

    actual_A = torch.zeros((B,L,hidden_dim), dtype=torch.float16).cuda()
    expected_A = F.scaled_dot_product_attention(input_A_Q, input_A_K, input_A_V)

    # TODO: add flash_attn call on actual_A here

    assert torch.allclose(expected_A, actual_A), "flash_attn"
    
    # output act. shape [B // dp_size, L, hidden_dim]

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

    assert torch.allclose(expected_A, actual_A), "moe_scoring"

    # output act. shape [B // dp_size, L, n_experts]

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

    assert torch.allclose(expected_A, actual_A), "ffn"

    # output act. shape [B // dp_size, L, hidden_dim]