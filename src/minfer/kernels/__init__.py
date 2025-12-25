# TODO: eventually add Triton?
class KernelBackend:
    def __init__(self, backend : str):
        if backend == "torch_ext":
            from .torch_ext import _dequant, _quant, rmsnorm, il_rope, neox_rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
            self._dequant = _dequant
            self._quant = _quant
            self.rmsnorm = rmsnorm
            self.il_rope = il_rope
            self.neox_rope = neox_rope
            self.matmul = matmul
            self.embed = embed
            self.qkv = qkv
            self.flash_attn = flash_attn
            self.moe_scoring = moe_scoring
            self.ffn = ffn
        else:
            raise ValueError(f"KernelBackend init: unsupported backend {backend}")