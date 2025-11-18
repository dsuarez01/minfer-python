#include <torch/extension.h>

// NOTE: sometimes llama-cpp will change things on their end
// Discrepancy will cause errors... Is there a better way to deal w this?
enum class GGMLQuantizationType : int {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39
};

__global__ void _dequant_row(GGMLQuantizationType qtype) {
    switch (qtype) {
        case GGMLQuantizationType::Q4_0: __dequant_row_q4_0(); break;
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

__device__ void __dequant_row(GGMLQuantizationType qtype) {
    switch (qtype) {
        case GGMLQuantizationType::Q4_0: __dequant_row_q4_0(); break;
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

__device__ void __dequant_row_q4_0() {
    TORCH_CHECK(false, "_dequant_row_q4_0 not implemented");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_dequant_row", &_dequant_row);
}