#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

#include <cassert>

#include "impl_common.hpp"
#include "quants_impl.cuh"

namespace minfer {

template <typename T>
__global__ void dequant_cuda_(
    GGMLQuantizationType qtype,
    const uint8_t* __restrict__ xr,
    T* __restrict__ y,
    int64_t block_size,
    int64_t type_size,
    int64_t b,
    int64_t k
) {

    int row = blockIdx.x;
    int block_in_row = blockIdx.y;

    const uint8_t* w = xr + row*b + block_in_row*type_size;
    T* out = y + row*k + block_in_row*block_size;

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_block_q4_0<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q4_1: dequant_block_q4_1<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q5_0: dequant_block_q5_0<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q5_1: dequant_block_q5_1<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q8_0: dequant_block_q8_0<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::MXFP4: dequant_block_mxfp4<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q2_K: dequant_block_q2_K<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q3_K: dequant_block_q3_K<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q4_K: dequant_block_q4_K<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q5_K: dequant_block_q5_K<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q6_K: dequant_block_q6_K<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::TQ1_0: dequant_block_tq1_0<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::TQ2_0: dequant_block_tq2_0<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_block_iq2_xxs<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ2_XS: dequant_block_iq2_xs<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ2_S: dequant_block_iq2_s<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_block_iq3_xxs<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ3_S: dequant_block_iq3_s<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ1_S: dequant_block_iq1_s<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ1_M: dequant_block_iq1_m<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ4_NL: dequant_block_iq4_nl<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::IQ4_XS: dequant_block_iq4_xs<T>(w, out, 1, threadIdx.x); break;
        case GGMLQuantizationType::Q8_K: dequant_block_q8_K<T>(w, out, 1, threadIdx.x); break;
        default: assert(false && "Unsupported dtype");
    }
}

// TODO: complete me!
// only call on 2D tensors for right now, 
// no need for larger since this is just to test the impl
void dequant_cuda(
    int64_t qtype_int, 
    const at::Tensor& x, 
    at::Tensor& y, 
    int64_t block_size, // num deq elems in block
    int64_t type_size // byte size of block
) {
    TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
    TORCH_CHECK(x.dim() == 2 && y.dim() == 2);
    TORCH_CHECK(x.size(0) == y.size(0), "x and y must have the same number of rows");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kByte, "x must be uint8 (byte)");
    TORCH_CHECK(y.scalar_type() == at::kFloat || y.scalar_type() == at::kHalf, "y must be float or half");

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);
    
    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

    int n_rows = x.size(0);
    int n_blocks_per_row = x.size(-1) / type_size;
    dim3 grid(n_rows, n_blocks_per_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const uint8_t* __restrict__ x_ptr = x.data_ptr<uint8_t>();
    
    switch (y.scalar_type()) {
        case at::kFloat: {
            float* __restrict__ y_ptr = y.data_ptr<float>();
            dequant_cuda_<float><<<grid, block_size, 0, stream>>>(qtype, x_ptr, y_ptr, block_size, type_size, x.size(-1), y.size(-1));
            break;
        }
        case at::kHalf: {
            half* __restrict__ y_ptr = reinterpret_cast<half*>(y.data_ptr<at::Half>());
            dequant_cuda_<half><<<grid, block_size, 0, stream>>>(qtype, x_ptr, y_ptr, block_size, type_size, x.size(-1), y.size(-1));
            break;
        }
        default: TORCH_CHECK(false, "Expected y scalar dtype to be float32 or float16, got ", y.scalar_type()); break;
    }
}

TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
    m.impl("dequant", &dequant_cuda);
}

}