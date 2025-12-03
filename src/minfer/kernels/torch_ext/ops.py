import torch
from torch import Tensor
from minfer.const import GGMLQuantizationType

# TODO: need to make CUDA/C++ kernels compatible with torch.compile here

def dequant(qtype: GGMLQuantizationType, x: Tensor, y: Tensor, block_size: int, type_size: int):
    """
    Dequantizes the rows of x into y
    
    block_size is num dequantized elements per block
    type_size is byte size of block
    """
    return torch.ops.minfer.dequant_row.default(qtype, x, y, block_size, type_size)

@torch.library.register_fake("minfer::dequant")
def _(qtype, x, y, block_size, type_size):
    torch._check(x.dtype == torch.uint8)
    torch._check(y.dtype in (torch.float32, torch.float16))
    torch._check(x.device == y.device)
    torch._check(x.size(0) == y.size(0))

def quant(qtype: GGMLQuantizationType, x: Tensor, y: Tensor, block_size: int, type_size: int):
    """
    Quantizes the rows of x into y
    
    block_size is num dequantized elements per block
    type_size is byte size of block
    """
    return torch.ops.minfer.quant_row.default(qtype, x, y, block_size, type_size)

@torch.library.register_fake("minfer::quant")
def _(qtype, x, y, block_size, type_size):
    torch._check(x.dtype == torch.float32)
    torch._check(y.dtype == torch.uint8)
    torch._check(x.device == y.device)
    torch._check(x.size(0) == y.size(0))

def rmsnorm():
    return

@torch.library.register_fake("minfer::rmsnorm")
def _():
    return

def il_rope():
    return

@torch.library.register_fake("minfer::il_rope")
def _():
    return

def neox_rope():
    return

@torch.library.register_fake("minfer::neox_rope")
def _():
    return

def matmul():
    return

@torch.library.register_fake("minfer::matmul")
def _():
    return

def embed():
    return

@torch.library.register_fake("minfer::embed")
def _():
    return

def qkv():
    return

@torch.library.register_fake("minfer::qkv")
def _():
    return

def flash_attn():
    return

@torch.library.register_fake("minfer::flash_attn")
def _():
    return

def moe_scoring():
    return

@torch.library.register_fake("minfer::moe_scoring")
def _():
    return

def ffn():
    return

@torch.library.register_fake("minfer::ffn")
def _():
    return