# TODO: logic needs to be fixed: when choosing triton as the backend, 
# still need _quant to check Triton against
class KernelBackend:
    def __init__(self, backend: str):
        if backend == "triton": # TODO: need to change this (arguments, fixing row-specific behavior references) once fixed
            from .triton import _dequant, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
        elif backend == "torch_ext":
            from .torch_ext import _dequant, _quant, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
            self._quant = _quant
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'triton' or 'torch_ext'.")
        
        self._dequant = _dequant
        self.rmsnorm = rmsnorm
        self.il_rope = il_rope
        self.neox_rope = neox_rope
        self.matmul = matmul
        self.embed = embed
        self.qkv = qkv
        self.flash_attn = flash_attn
        self.moe_scoring = moe_scoring
        self.ffn = ffn