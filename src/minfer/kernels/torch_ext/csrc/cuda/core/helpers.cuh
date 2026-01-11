#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"
#include "quants/impls.cuh"

// helpers for warp-level and block-level reductions
// refer to: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
template <int WIDTH=32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
    unsigned mask = (1ull << WIDTH) - 1;

    for (int offset = WIDTH/2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, WIDTH);
    }
    return __shfl_sync(mask, v, 0, WIDTH);
}

template <int WIDTH=32>
__device__ __forceinline__ float warp_reduce_max(float v) {
    unsigned mask = (1ull << WIDTH) - 1;

    for (int offset = WIDTH/2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset, WIDTH));
    }
    return __shfl_sync(mask, v, 0, WIDTH);
}

__device__ __forceinline__ float blockreduce_sum(float* vs, float v, int tid) {
    int warp_id = tid/32;
    int lane_id = tid%32;
    int n_warps = blockDim.x/32;

    v = warp_reduce_sum(v);
    if (lane_id == 0) vs[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        v = (lane_id < n_warps) ? vs[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
    }
    if (tid == 0) vs[0] = v;
    __syncthreads();
    return vs[0];
}

__device__ __forceinline__ float blockreduce_max(float* vs, float v, int tid) {
    int warp_id = tid/32;
    int lane_id = tid%32;
    int n_warps = blockDim.x/32;

    v = warp_reduce_max(v);
    if (lane_id == 0) vs[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        v = (lane_id < n_warps) ? vs[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
    }
    if (tid == 0) vs[0] = v;
    __syncthreads();
    return vs[0];
}

// helper for dispatch in matmul_cuda_impl
// might template this later on to additionally support float
__device__ __forceinline__ void dequant_block(
    int64_t qtype_int,
    int64_t tid,
    half* __restrict__ y,
    const uint8_t* __restrict__ w
) {
    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_block_q4_0<half>(w, y, tid); break;
        case GGMLQuantizationType::Q4_1: dequant_block_q4_1<half>(w, y, tid); break;
        case GGMLQuantizationType::Q5_0: dequant_block_q5_0<half>(w, y, tid); break;
        case GGMLQuantizationType::Q5_1: dequant_block_q5_1<half>(w, y, tid); break;
        case GGMLQuantizationType::Q8_0: dequant_block_q8_0<half>(w, y, tid); break;
        case GGMLQuantizationType::MXFP4: dequant_block_mxfp4<half>(w, y, tid); break;
        case GGMLQuantizationType::Q2_K: dequant_block_q2_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q3_K: dequant_block_q3_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q4_K: dequant_block_q4_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q5_K: dequant_block_q5_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q6_K: dequant_block_q6_K<half>(w, y, tid); break;
        case GGMLQuantizationType::TQ1_0: dequant_block_tq1_0<half>(w, y, tid); break;
        case GGMLQuantizationType::TQ2_0: dequant_block_tq2_0<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_block_iq2_xxs<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ2_XS: dequant_block_iq2_xs<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ2_S: dequant_block_iq2_s<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_block_iq3_xxs<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ3_S: dequant_block_iq3_s<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ1_S: dequant_block_iq1_s<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ1_M: dequant_block_iq1_m<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ4_NL: dequant_block_iq4_nl<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ4_XS: dequant_block_iq4_xs<half>(w, y, tid); break;
        case GGMLQuantizationType::Q8_K: dequant_block_q8_K<half>(w, y, tid); break;
        case GGMLQuantizationType::BF16: dequant_block_bf16<half>(w, y, tid); break;
        default: assert(false && "Unsupported dtype"); // this gets compiled out in non-debug builds...
    }
}