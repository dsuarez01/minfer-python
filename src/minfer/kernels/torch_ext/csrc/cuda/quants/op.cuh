#pragma once

#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

// TODO: complete me!
// only works for 2D tensors for right now,
// no need for larger since this is just to test the impl
void dequant_cuda(
    int64_t qtype_int, 
    int64_t qblock_size, // num deq elems in block
    int64_t qtype_size, // byte size of block
    const at::Tensor& x, 
    at::Tensor& y
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

    unsigned int n_rows = x.size(0);
    unsigned int n_qblocks_per_row = x.size(-1) / qtype_size;
    dim3 grid(n_rows, n_qblocks_per_row);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const uint8_t* __restrict__ x_ptr = x.data_ptr<uint8_t>();
    
    switch (y.scalar_type()) {
        case at::kFloat: {
            float* __restrict__ y_ptr = y.data_ptr<float>();
            constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(float);
            unsigned int thrs_per_block = qblock_size/ELEMS_PER_THR;
            dequant_cuda_<float><<<grid, thrs_per_block, 0, stream>>>(qtype, qblock_size, qtype_size, x_ptr, y_ptr, x.size(-1), y.size(-1));
            break;
        }
        case at::kHalf: {
            half* __restrict__ y_ptr = reinterpret_cast<half*>(y.data_ptr<at::Half>());
            constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(half);
            unsigned int thrs_per_block = qblock_size/ELEMS_PER_THR;
            dequant_cuda_<half><<<grid, thrs_per_block, 0, stream>>>(qtype, qblock_size, qtype_size, x_ptr, y_ptr, x.size(-1), y.size(-1));
            break;
        }
        default: TORCH_CHECK(false, "Expected y scalar dtype to be float32 or float16, got ", y.scalar_type()); break;
    }
}