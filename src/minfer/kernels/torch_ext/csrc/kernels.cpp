#include <Python.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>



// TODO: complete me!
namespace minfer {

void rmsnorm_cpu() {
    TORCH_CHECK(false, "rmsnorm not implemented");
}

void il_rope_cpu() {
    TORCH_CHECK(false, "il_rope not implemented");
}

void neox_rope_cpu() {
    TORCH_CHECK(false, "neox_rope not implemented");
}

void matmul_cpu() {
    TORCH_CHECK(false, "matmul not implemented");
}

void embed_cpu() {
    TORCH_CHECK(false, "embed not implemented");
}

void qkv_cpu() {
    TORCH_CHECK(false, "qkv not implemented");
}

void flash_attn_cpu() {
    TORCH_CHECK(false, "flash_attn not implemented");
}

void moe_scoring_cpu() {
    TORCH_CHECK(false, "moe_scoring not implemented");
}

void ffn_cpu() {
    TORCH_CHECK(false, "ffn not implemented");
}

// TORCH_LIBRARY_IMPL(minfer, CPU, m) {
//     m.impl("rmsnorm", &rmsnorm_cpu);
//     m.impl("il_rope", &il_rope_cpu);
//     m.impl("neox_rope", &neox_rope_cpu);
//     m.impl("matmul", &matmul_cpu);
//     m.impl("embed", &embed_cpu);
//     m.impl("qkv", &qkv_cpu);
//     m.impl("flash_attn", &flash_attn_cpu);
//     m.impl("moe_scoring", &moe_scoring_cpu);
//     m.impl("ffn", &ffn_cpu);
// }

}