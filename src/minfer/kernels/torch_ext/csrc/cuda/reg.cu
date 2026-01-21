#include <torch/csrc/stable/library.h>

#include "quants/op.cuh"
#include "rmsnorm/op.cuh"
#include "il_rope/op.cuh"
#include "neox_rope/op.cuh"
#include "matmul/op.cuh"
#include "embed/op.cuh"
#include "qkv/op.cuh"
#include "flash_attn/op.cuh"
#include "moe_scoring/op.cuh"
#include "ffn/op.cuh"

namespace minfer {

    STABLE_TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
        m.impl("dequant", TORCH_BOX(&dequant_cuda));
        m.impl("rmsnorm", TORCH_BOX(&rmsnorm_cuda));
        m.impl("il_rope", TORCH_BOX(&il_rope_cuda));
        m.impl("neox_rope", TORCH_BOX(&neox_rope_cuda));
        m.impl("matmul", TORCH_BOX(&matmul_cuda));
        m.impl("embed", TORCH_BOX(&embed_cuda));
        m.impl("qkv", TORCH_BOX(&qkv_cuda));
        m.impl("flash_attn", TORCH_BOX(&flash_attn_cuda));
        m.impl("moe_scoring", TORCH_BOX(&moe_scoring_cuda));
        m.impl("ffn", TORCH_BOX(&ffn_cuda));
    }

}