#include <torch/extension.h>

// TODO: complete
void rmsnorm() {
    TORCH_CHECK(false, "rmsnorm not implemented");
}
void il_rope() {
    TORCH_CHECK(false, "il_rope not implemented");
}
void neox_rope() {
    TORCH_CHECK(false, "neox_rope not implemented");
}
void matmul() {
    TORCH_CHECK(false, "matmul not implemented");
}
void embed() {
    TORCH_CHECK(false, "embed not implemented");
}
void qkv() {
    TORCH_CHECK(false, "qkv not implemented");
}
void flash_attn() {
    TORCH_CHECK(false, "flash_attn not implemented");
}
void moe_scoring() {
    TORCH_CHECK(false, "moe_scoring not implemented");
}
void ffn() {
    TORCH_CHECK(false, "ffn not implemented");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm);
    m.def("il_rope", &il_rope);
    m.def("neox_rope", &neox_rope);
    m.def("matmul", &matmul);
    m.def("embed", &embed);
    m.def("qkv", &qkv);
    m.def("flash_attn", &flash_attn);
    m.def("moe_scoring", &moe_scoring);
    m.def("ffn", &ffn);
}