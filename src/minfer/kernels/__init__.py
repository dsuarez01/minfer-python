class KernelBackend:
    def __init__(self, backend: str):
        if backend == "triton":
            from .triton import embed, rmsnorm, matmul, qkv, flash_attn, moe_scores, moe_experts
            self.embed = embed
            self.rmsnorm = rmsnorm
            self.matmul = matmul
            self.qkv = qkv
            self.flash_attn = flash_attn
            self.moe_scores = moe_scores
            self.moe_experts = moe_experts
        elif backend == "cuda":
            from .cuda import embed, rmsnorm, matmul, qkv, flash_attn, moe_scores, moe_experts
            self.embed = embed
            self.rmsnorm = rmsnorm
            self.matmul = matmul
            self.qkv = qkv
            self.flash_attn = flash_attn
            self.moe_scores = moe_scores
            self.moe_experts = moe_experts
        elif backend == "cpu": # fallback
            from .cpu import embed, rmsnorm, matmul, qkv, flash_attn, moe_scores, moe_experts
            self.embed = embed
            self.rmsnorm = rmsnorm
            self.matmul = matmul
            self.qkv = qkv
            self.flash_attn = flash_attn
            self.moe_scores = moe_scores
            self.moe_experts = moe_experts
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'triton', 'cuda', or 'cpu'")