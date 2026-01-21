#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

// TODO: complete me!
// only works for 2D tensors for right now,
// no need for larger since this is just to test the impl

namespace minfer {

    using namespace impl;

    void dequant_cuda(
        int64_t qtype_int, 
        int64_t qblock_size, // num deq elems in block
        int64_t qtype_size, // byte size of block
        const torch::stable::Tensor& x, 
        torch::stable::Tensor& y
    ) {
        STD_TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
        STD_TORCH_CHECK(x.dim() == 2 && y.dim() == 2);
        STD_TORCH_CHECK(x.size(0) == y.size(0), "x and y must have the same number of rows");
        STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        STD_TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
        STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Byte, "x must be uint8 (byte)");
        STD_TORCH_CHECK(y.scalar_type() == torch::headeronly::ScalarType::Float || y.scalar_type() == torch::headeronly::ScalarType::Half, "y must be float or half");

        STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CUDA);
        STD_TORCH_CHECK(y.device().type() == torch::headeronly::DeviceType::CUDA);
        
        GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

        unsigned int n_rows = x.size(0);
        unsigned int n_qblocks_per_row = x.size(-1) / qtype_size;
        dim3 grid(n_rows, n_qblocks_per_row);
        
        auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
        void* stream_ptr = nullptr;
        TORCH_ERROR_CODE_CHECK(
            aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
        );
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

        const auto* __restrict__ x_ptr = x.const_data_ptr<uint8_t>();
        
        switch (y.scalar_type()) {
            case torch::headeronly::ScalarType::Float: {
                auto* __restrict__ y_ptr = y.mutable_data_ptr<float>();
                constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(float);
                unsigned int thrs_per_block = qblock_size/ELEMS_PER_THR;
                dequant_cuda_<float><<<grid, thrs_per_block, 0, stream>>>(qtype, qblock_size, qtype_size, x_ptr, y_ptr, x.size(-1), y.size(-1));
                break;
            }
            case torch::headeronly::ScalarType::Half: {
                auto* __restrict__ y_ptr = reinterpret_cast<half*>(y.mutable_data_ptr<torch::headeronly::Half>());
                constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(half);
                unsigned int thrs_per_block = qblock_size/ELEMS_PER_THR;
                dequant_cuda_<half><<<grid, thrs_per_block, 0, stream>>>(qtype, qblock_size, qtype_size, x_ptr, y_ptr, x.size(-1), y.size(-1));
                break;
            }
            default: STD_TORCH_CHECK(false, "Expected y scalar dtype to be float32 or float16, got ", y.scalar_type()); break;
        }

        STD_CUDA_KERNEL_LAUNCH_CHECK();
    }
}