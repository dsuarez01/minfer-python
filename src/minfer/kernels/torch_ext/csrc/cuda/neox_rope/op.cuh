#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

void neox_rope_cuda(
    int64_t rotary_dim,
    int64_t start_pos,
    double freq_base,
    at::Tensor& x // [B,L,n_heads,head_dim] (BEFORE transpose for flash attn, but AFTER split)
) {
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(rotary_dim <= x.size(-1));

    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    size_t B = x.size(0);
    size_t L = x.size(1);
    size_t n_heads = x.size(2);
    size_t head_dim = x.size(3);

    dim3 grid(
        static_cast<unsigned int>(B*L), 
        static_cast<unsigned int>(n_heads), 
        static_cast<unsigned int>((rotary_dim/2+31)/32)
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    neox_rope_cuda_impl<<<grid, 32u, 0, stream>>>(L, rotary_dim, head_dim, start_pos, freq_base, x_ptr);
}