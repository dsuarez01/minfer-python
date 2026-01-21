#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

namespace minfer {

using namespace impl;

void il_rope_cuda(
    int64_t rotary_dim,
    int64_t start_pos,
    double freq_base,
    torch::stable::Tensor& x // [B,L,n_heads,head_dim] (BEFORE transpose for flash attn, but AFTER split)
) {
    STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(x.is_contiguous());
    STD_TORCH_CHECK(x.dim() == 4);
    STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(rotary_dim <= x.size(3));

    half* x_ptr = reinterpret_cast<half*>(x.mutable_data_ptr<half_t>());

    size_t B = x.size(0);
    size_t L = x.size(1);
    size_t n_heads = x.size(2);
    size_t head_dim = x.size(3);

    dim3 grid(
        static_cast<unsigned int>(B*L),
        static_cast<unsigned int>(n_heads),
        static_cast<unsigned int>((rotary_dim/2+31)/32)
    );

    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
    );
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    il_rope_cuda_impl<<<grid, 32u, 0, stream>>>(L, rotary_dim, head_dim, start_pos, freq_base, x_ptr);

    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}