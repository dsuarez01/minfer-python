#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

namespace minfer::impl {

template <int TILE_M, int N_EXPS, int TOP_K>
__global__ void moe_scoring_cuda_impl(
    size_t M, size_t K,
    const half* __restrict__ x,
    const half* __restrict__ w,
    uint8_t* __restrict__ act_exps,
    half* __restrict__ act_exps_scores,
    half* __restrict__ scores
) {
    
}

}