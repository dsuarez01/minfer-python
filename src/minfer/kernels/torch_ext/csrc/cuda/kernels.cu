#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>
// #include <ATen/cuda/CUDAContext.h>

// TODO: complete me!
namespace minfer {

void rmsnorm_cuda() {
    TORCH_CHECK(false, "rmsnorm not implemented");
}

void il_rope_cuda() {
    TORCH_CHECK(false, "il_rope not implemented");
}

void neox_rope_cuda() {
    TORCH_CHECK(false, "neox_rope not implemented");
}

void matmul_cuda() {
    TORCH_CHECK(false, "matmul not implemented");
}

void embed_cuda() {
    TORCH_CHECK(false, "embed not implemented");
}

void qkv_cuda() {
    TORCH_CHECK(false, "qkv not implemented");
}

void flash_attn_cuda() {
    TORCH_CHECK(false, "flash_attn not implemented");
}

void moe_scoring_cuda() {
    TORCH_CHECK(false, "moe_scoring not implemented");
}

void ffn_cuda() {
    TORCH_CHECK(false, "ffn not implemented");
}

TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
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