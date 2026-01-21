#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

namespace minfer::impl {

__global__ void neox_rope_cuda_impl(
    size_t L,
    size_t rotary_dim,
    size_t head_dim,
    size_t start_pos,
    float freq_base,
    half* __restrict__ x
) {

    unsigned int B_idx = blockIdx.x / L;
    unsigned int L_idx = blockIdx.x % L;
    unsigned int head_idx = blockIdx.y;
    unsigned int pair_idx = blockIdx.z * 32 + threadIdx.x;
    
    unsigned int n_heads = gridDim.y;
    unsigned int pos = start_pos + L_idx;
    
    if (static_cast<size_t>(pair_idx) >= rotary_dim/2) return;

    half* x_head = x + B_idx*L*n_heads*head_dim + L_idx*n_heads*head_dim + head_idx*head_dim;

    float freq = 1.0f / pow(freq_base, (2.0f * pair_idx) / rotary_dim);
    float angle = pos * freq;

    float x_0 = __half2float(x_head[pair_idx]);
    float x_1 = __half2float(x_head[pair_idx+rotary_dim/2]);

    x_head[pair_idx] = __float2half(cos(angle)*x_0 - sin(angle)*x_1);
    x_head[pair_idx+rotary_dim/2] = __float2half(sin(angle)*x_0 + cos(angle)*x_1);
}

}