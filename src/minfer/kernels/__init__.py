

# TODO: adjust KernelBackend to handle ops logic as necessary

class KernelBackend:
    def __init__(self, backend: str):
        if backend == "triton":
            from .triton import _dequant_row, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
        elif backend == "torch_ext":
            from .torch_ext import _dequant_row, _quant_row, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
            self._quant_row = _quant_row
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'triton' or 'torch_ext'.")
        
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