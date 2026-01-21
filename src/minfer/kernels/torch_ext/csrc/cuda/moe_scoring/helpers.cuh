#pragma once

#include "cuda/core/helpers.cuh"

namespace minfer::impl {
    
}

// template <int THR_PER_ROW>
// __device__ __forceinline__ float row_reduce_max(float* vs, float v, int row_in_tile, int thr_in_row) {
//     constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;
//     constexpr int WARP_WIDTH = WARPS_PER_ROW > 1 ? 32 : THR_PER_ROW;

//     v = warp_reduce_max<WARP_WIDTH>(v);

//     int warp_in_row = thr_in_row / 32;
//     if (thr_in_row % 32 == 0) vs[row_in_tile*WARPS_PER_ROW+warp_in_row]=v;
//     __syncthreads();
    
//     if (WARPS_PER_ROW > 1 && warp_in_row == 0 && thr_in_row < WARPS_PER_ROW) {
//         v = vs[row_in_tile*WARPS_PER_ROW + thr_in_row];
//         v = warp_reduce_max<WARPS_PER_ROW>(v);
//     }

//     if (thr_in_row == 0) vs[row_in_tile*WARPS_PER_ROW] = v;
//     __syncthreads();
//     return vs[row_in_tile*WARPS_PER_ROW];
// }

// template <int THR_PER_ROW>
// __device__ __forceinline__ float row_reduce_sum(float* vs, float v, int row_in_tile, int thr_in_row) {
//     constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;
//     constexpr int WARP_WIDTH = WARPS_PER_ROW > 1 ? 32 : THR_PER_ROW;

//     v = warp_reduce_sum<WARP_WIDTH>(v);

//     int warp_in_row = thr_in_row / 32;
//     if (thr_in_row % 32 == 0) vs[row_in_tile*WARPS_PER_ROW+warp_in_row]=v;
//     __syncthreads();
    
//     if (WARPS_PER_ROW > 1 && warp_in_row == 0 && thr_in_row < WARPS_PER_ROW) {
//         v = vs[row_in_tile*WARPS_PER_ROW + thr_in_row];
//         v = warp_reduce_sum<WARPS_PER_ROW>(v);
//     }

//     if (thr_in_row == 0) vs[row_in_tile*WARPS_PER_ROW] = v;
//     __syncthreads();
//     return vs[row_in_tile*WARPS_PER_ROW];
// }