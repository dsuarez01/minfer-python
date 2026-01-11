#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

__global__ void il_rope_cuda_impl(
    int64_t n_heads,
    int64_t rotary_dim,
    int64_t head_dim,
    int64_t start_pos,
    float freq_base,
    half* __restrict__ x
) {
    int L = gridDim.z / ((rotary_dim/2+31)/32);
    int head_idx = blockIdx.y;
    int pair_block = blockIdx.z % ((rotary_dim/2+31)/32);
    int seq_idx = blockIdx.z / ((rotary_dim/2+31)/32);
    int pair_idx = pair_block * 32 + threadIdx.x;
    int pos = start_pos + seq_idx;
    
    if (pair_idx >= rotary_dim / 2) return;

    half* x_head = x + (blockIdx.x*n_heads*L + head_idx*L + seq_idx)*head_dim;

    float freq = 1.0f / pow(freq_base, 2.0f * pair_idx / rotary_dim);
    float angle = pos * freq;

    int idx = 2*pair_idx;

    float x_0 = __half2float(x_head[idx]);
    float x_1 = __half2float(x_head[idx+1]);

    x_head[idx] = __float2half(cos(angle)*x_0 - sin(angle)*x_1);
    x_head[idx+1] = __float2half(sin(angle)*x_0 + cos(angle)*x_1);
}