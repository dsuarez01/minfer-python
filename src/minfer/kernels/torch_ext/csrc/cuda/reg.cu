#include <torch/library.h>

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

    TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
        m.impl("dequant", &dequant_cuda);
        m.impl("rmsnorm", &rmsnorm_cuda);
        m.impl("il_rope", &il_rope_cuda);
        m.impl("neox_rope", &neox_rope_cuda);
        m.impl("matmul", &matmul_cuda);
        m.impl("embed", &embed_cuda);
        m.impl("qkv", &qkv_cuda);
        m.impl("flash_attn", &flash_attn_cuda);
        m.impl("moe_scoring", &moe_scoring_cuda);
        m.impl("ffn", &ffn_cuda);
    }

}