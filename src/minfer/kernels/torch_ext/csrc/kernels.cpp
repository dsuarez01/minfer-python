#include <Python.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

extern "C" {
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

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

TORCH_LIBRARY(minfer, m) {
    m.def("dequant(int qtype, Tensor x, Tensor(a!) y, int qblock_size, int qtype_size) -> ()");
    m.def("quant(int qtype, Tensor x, Tensor(a!) y, int qblock_size, int qtype_size) -> ()");
    m.def("rmsnorm() -> ()");
    m.def("il_rope() -> ()");
    m.def("neox_rope() -> ()");
    m.def("matmul() -> ()");
    m.def("embed() -> ()");
    m.def("qkv() -> ()");
    m.def("flash_attn() -> ()");
    m.def("moe_scoring() -> ()");
    m.def("ffn() -> ()");
}

TORCH_LIBRARY_IMPL(minfer, CPU, m) {
    m.impl("rmsnorm", &rmsnorm_cpu);
    m.impl("il_rope", &il_rope_cpu);
    m.impl("neox_rope", &neox_rope_cpu);
    m.impl("matmul", &matmul_cpu);
    m.impl("embed", &embed_cpu);
    m.impl("qkv", &qkv_cpu);
    m.impl("flash_attn", &flash_attn_cpu);
    m.impl("moe_scoring", &moe_scoring_cpu);
    m.impl("ffn", &ffn_cpu);
}

}