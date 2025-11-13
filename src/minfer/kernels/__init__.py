class KernelBackend:
    def __init__(self, backend: str):
        if backend == "triton":
            from .triton import rmsnorm, rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
            self.rmsnorm = rmsnorm
            self.rope = rope
            self.matmul = matmul
            self.embed = embed
            self.qkv = qkv
            self.flash_attn = flash_attn
            self.moe_scoring = moe_scoring
            self.ffn = ffn
        elif backend == "cuda":
            from .cuda import rmsnorm, rope, matmul, embed, qkv, flash_attn, moe_scoring, ffn
            self.rmsnorm = rmsnorm
            self.rope = rope
            self.matmul = matmul
            self.embed = embed
            self.qkv = qkv
            self.flash_attn = flash_attn
            self.moe_scoring = moe_scoring
            self.ffn = ffn
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'triton' or 'cuda'")