#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

namespace minfer::impl {

// FlashAttention-2 forward pass impl.
// Refer to: https://arxiv.org/abs/2307.08691
constexpr float HALF_MIN = -65504.0f;
struct QKVStrides {
    int64_t q0, q1, q2;
    int64_t kv0, kv1, kv2;
};
template <int WARPS_PER_BLOCK, int HEAD_DIM>
__global__ void flash_attn_cuda_impl(
    QKVStrides strides,
    int64_t L, int64_t n_kv_heads,
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const uint8_t* __restrict__ mask,
    half* __restrict__ out
) {
    
}

}