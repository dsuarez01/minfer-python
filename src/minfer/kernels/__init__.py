from . import _C, ops

# TODO 1: need to make CUDA/C++ kernels compatible with torch.compile here
# TODO 2: adjust KernelBackend to handle this logic as necessary

class KernelBackend:
    def __init__(self, backend: str):
        if backend == "triton":
            from .triton import _dequant_row, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
        elif backend == "cuda":

            # from .cuda import _dequant_row, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
        elif backend == "_ref":
            from ._ref import _dequant_row, _quant_row, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
            self._quant_row = _quant_row
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'triton', 'cuda', or '_ref'.")
        
        self._dequant_row = _dequant_row
        self.rmsnorm = rmsnorm
        self.il_rope = il_rope
        self.neox_rope = neox_rope
        self.matmul = matmul
        self.embed = embed
        self.qkv = qkv
        self.flash_attn = flash_attn
        self.moe_scoring = moe_scoring
        self.ffn = ffn