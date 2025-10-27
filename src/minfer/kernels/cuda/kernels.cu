#include <torch/extension.h>

// TODO: complete
void embed() {}
void rmsnorm() {}
void matmul() {}
void qkv() {}
void flash_attn() {}
void moe_scores() {}
void moe_experts() {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embed", &embed);
    m.def("rmsnorm", &rmsnorm);
    m.def("matmul", &matmul);
    m.def("qkv", &qkv);
    m.def("flash_attn", &flash_attn);
    m.def("moe_scores", &moe_scores);
    m.def("moe_experts", &moe_experts);
}