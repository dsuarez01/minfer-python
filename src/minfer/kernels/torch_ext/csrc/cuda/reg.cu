#include <Python.h>
#include <torch/extension.h>

#include "quants/kernel.cuh"
#include "rmsnorm/kernel.cuh"
#include "il_rope/kernel.cuh"
#include "neox_rope/kernel.cuh"
#include "matmul/kernel.cuh"
#include "embed/kernel.cuh"
#include "qkv/kernel.cuh"
#include "flash_attn/kernel.cuh"
#include "moe_scoring/kernel.cuh"
#include "ffn/kernel.cuh"

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

TORCH_LIBRARY(minfer, m) {
    m.def("dequant(int qtype, int qblock_size, int qtype_size, Tensor x, Tensor(a!) y) -> ()");
    m.def("quant(int qtype, int qblock_size, int qtype_size, Tensor x, Tensor(a!) y) -> ()");
    m.def("rmsnorm(float eps, Tensor input, Tensor w, Tensor(a!) out) -> ()");
    m.def("il_rope(int rotary_dim, int start_pos, float freq_base, Tensor(a!) x) -> ()");
    m.def("neox_rope(int rotary_dim, int start_pos, float freq_base, Tensor(a!) x) -> ()");
    m.def("matmul(int qtype_int, int qblock_size, int qtype_size, Tensor x, Tensor w, Tensor(a!) out) -> ()");
    m.def("embed(int qtype_int, int qblock_size, int qtype_size, Tensor token_ids, Tensor w, Tensor(a!) x) -> ()");
    m.def("qkv(int q_qtype_int, int k_qtype_int, int v_qtype_int, int q_qblock_size, int k_qblock_size, int v_qblock_size, int q_qtype_size, int k_qtype_size, int v_qtype_size, Tensor x, Tensor wq, Tensor wk, Tensor wv, Tensor(a!) q_out, Tensor(a!) k_out, Tensor(a!) v_out) -> ()");
    m.def("flash_attn(int qtype_int, int qblock_size, int qtype_size, Tensor q, Tensor k, Tensor v, Tensor(a!) mask, Tensor(a!) out) -> ()");
    m.def("moe_scoring(int qtype_int, int qblock_size, int qtype_size, Tensor x, Tensor w, Tensor(a!) act_exps, Tensor(a!) act_exps_scores, Tensor(a!) scores) -> ()");
    m.def("ffn(int up_qtype_int, int gate_qtype_int, int down_qtype_int, int up_qblock_size, int gate_qblock_size, int down_qblock_size, int up_qtype_size, int gate_qtype_size, int down_qtype_size, Tensor input, Tensor ws_up, Tensor ws_gate, Tensor ws_down, Tensor(a!) hb, Tensor(a!) hb2, Tensor(a!) out) -> ()");
}

// TODO: add this back in once finished
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