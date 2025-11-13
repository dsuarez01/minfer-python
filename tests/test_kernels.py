# TODO: add me
import torch
import pytest

from minfer.kernels import KernelBackend
from minfer.kernels.triton.kernels import _dequant as triton_dequant
from minfer.kernels.cuda.kernels import _dequant as cuda_dequant # type: ignore

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

## NOTE: for the rest of the tests FP32 / FP16 dtype tensors are used (as appropriate)
## since dequant already tests the relevant usage patterns in the other kernels
## dp_size is used throughout but not actually factored in since it doesn't affect kernel usage

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_rmsnorm(backend):
    kerns = KernelBackend(backend)
    B, L, hidden_dim = 8, 4096, 6144 # adjust as needed
    input = torch.randn((B,L,hidden_dim), dtype=torch.float16)
    weight = torch.randn((hidden_dim), dtype=torch.float16)
    # test for rmsnorm applied across entire vector (act. shape [B // dp_size, L, hidden_dim], weight shape [hidden_dim,])

    # test for rmsnorm applied across heads (act. shape [B // dp_size, n_heads, L, head_dim], weight shape [head_dim,])

    # in both cases: output act. shape identical to input

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_rope(backend):
    kerns = KernelBackend(backend)

    # test for rope applied across heads (act. shape [B // dp_size, n_heads, L, head_dim])
    # output act. shape identical to input

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_matmul(backend):
    kerns = KernelBackend(backend)

    # act. shape [B // dp_size, L, dim_in], weight shape [dim_out, dim_in]
    # output act. shape [B // dp_size, L, dim_out]

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_embed(backend):
    kerns = KernelBackend(backend)

    # act. shape [B // dp_size, L], weight shape [vocab_size, hidden_dim]
    # output act. shape [B // dp_size, L, hidden_dim]

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_qkv(backend):
    kerns = KernelBackend(backend)

    # act. shape [B // dp_size, L, hidden_dim], weight shape [q_dim or kv_dim, hidden_dim]
    # output: Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_flash_attn(backend):
    kerns = KernelBackend(backend)

    # Q shape [B // dp_size, n_heads, L, head_dim], K and V shape [B // dp_size, n_kv_heads, L, head_dim]
    # output act. shape [B // dp_size, L, hidden_dim]
    # can use F.scaled_dot_product_attention from pytorch for this?

@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_moe_scoring(backend):
    kerns = KernelBackend(backend)

    # performs matmul and fused (online) softmax
    # input act. shape [B // dp_size, L, hidden_dim]
    # output act. shape [B // dp_size, L, n_experts]


@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_ffn(backend):
    kerns = KernelBackend(backend)

    # performs SwiGLU then downproj.
    # input act. shape [B // dp_size, L, hidden_dim]
    # output act. shape [B // dp_size, L, hidden_dim]