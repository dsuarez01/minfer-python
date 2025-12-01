#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
// #include <ATen/cuda/CUDAContext.h>

#include "quants_impl.cuh"

namespace minfer {

// TODO: complete me!
void dequant_row_cuda(
    int qtype_int, 
    const at::Tensor& x, 
    at::Tensor& y, 
    int64_t b, 
    int64_t k
) {
    // TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
    TORCH_CHECK(x.size(0) == y.size(0), "x and y must have the same number of rows");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous")
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(y.dtype() == at::kByte, "y must be uint8 (byte)");

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);
    
    TORCH_CHECK(false, "_dequant_row not implemented");
}

// TODO: write a separate one of these for calling on the host?
// might actually have to template here on the type (float or half)
template <typename T>
__device__ void dequant_block_cuda(
    int qtype_int,
    const uint8_t* __restrict__ w,
    T* __restrict__ y,
    int stride,
    int tid,
    int n_threads
) {
    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_row_q4_0<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q4_1: dequant_row_q4_1<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q5_0: dequant_row_q5_0<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q5_1: dequant_row_q5_1<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q8_0: dequant_row_q8_0<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::MXFP4: dequant_row_mxfp4<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q2_K: dequant_row_q2_K<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q3_K: dequant_row_q3_K<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q4_K: dequant_row_q4_K<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q5_K: dequant_row_q5_K<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q6_K: dequant_row_q6_K<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::TQ1_0: dequant_row_tq1_0<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::TQ2_0: dequant_row_tq2_0<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_row_iq2_xxs<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ2_XS: dequant_row_iq2_xs<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ2_S: dequant_row_iq2_s<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_row_iq3_xxs<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ3_S: dequant_row_iq3_s<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ1_S: dequant_row_iq1_s<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ1_M: dequant_row_iq1_m<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ4_NL: dequant_row_iq4_nl<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::IQ4_XS: dequant_row_iq4_xs<T>(w, y, stride, tid, n_threads); break;
        case GGMLQuantizationType::Q8_K: dequant_row_q8_K<T>(w, y, stride, tid, n_threads); break;
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
    m.impl("dequant_row", &dequant_row_cuda);
}

}