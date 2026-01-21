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

namespace minfer {
    using namespace impl;

    void rmsnorm_cuda(
        double eps,
        const torch::stable::Tensor& in,
        const torch::stable::Tensor& w,
        torch::stable::Tensor& out
    ) {
        // checks
        STD_TORCH_CHECK(in.device().type() == torch::headeronly::DeviceType::CUDA);
        STD_TORCH_CHECK(out.device().type() == torch::headeronly::DeviceType::CUDA);
        
        STD_TORCH_CHECK(in.is_contiguous());
        STD_TORCH_CHECK(out.is_contiguous());
        STD_TORCH_CHECK(w.is_contiguous());

        STD_TORCH_CHECK(in.scalar_type() == torch::headeronly::ScalarType::Half);
        STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
        STD_TORCH_CHECK(w.scalar_type() == torch::headeronly::ScalarType::Half);

        STD_TORCH_CHECK(in.dim() == 3 || in.dim() == 4);

        STD_TORCH_CHECK(in.sizes().equals(out.sizes()));
        STD_TORCH_CHECK(in.size(-1) == w.size(0) || in.size(-1) % w.size(0) == 0); // per-head or per-entire vector
        
        const half* in_ptr = reinterpret_cast<const half*>(in.const_data_ptr<half_t>());
        half* out_ptr = reinterpret_cast<half*>(out.mutable_data_ptr<half_t>());
        const half* w_ptr = reinterpret_cast<const half*>(w.const_data_ptr<half_t>());

        // handles both [B,L,D] and [B,L,n_heads,head_dim]
        size_t dim = w.size(0);
        unsigned int n_blocks = in.numel() / dim;

        unsigned int block_size = min(1024u, max(128u, (unsigned int)((dim+8-1)/8)));
        block_size = (block_size+32-1)/32 * 32;

        auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
        void* stream_ptr = nullptr;
        TORCH_ERROR_CODE_CHECK(
            aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
        );
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

        rmsnorm_cuda_impl<<<n_blocks, block_size, 0, stream>>>(dim, eps, in_ptr, w_ptr, out_ptr);

        STD_CUDA_KERNEL_LAUNCH_CHECK();
    }
}