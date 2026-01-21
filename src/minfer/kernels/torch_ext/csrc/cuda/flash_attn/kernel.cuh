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
    
    // __shared__ half l_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_N];
    // __shared__ half k_shared[WMMA_N][WMMA_K];
    // __shared__ half pv_shared[WARPS_PER_BLOCK*WMMA_M][HEAD_DIM];
    // __shared__ half out_shared[WARPS_PER_BLOCK*WMMA_M][HEAD_DIM];
    
    // __shared__ float ds[WARPS_PER_BLOCK*WMMA_M];
    // __shared__ float ms[WARPS_PER_BLOCK*WMMA_M];

    // __shared__ float ms_old[WARPS_PER_BLOCK*WMMA_M];
    // __shared__ float ds_old[WARPS_PER_BLOCK*WMMA_M];

    // int64_t n_heads = gridDim.y;
    // int64_t batch_idx = blockIdx.x;
    // int64_t q_head_idx = blockIdx.y;
    // int64_t kv_head_idx = q_head_idx / (n_heads/n_kv_heads);
    // int64_t seq_tile = blockIdx.z;

    // const half* q_row = q + batch_idx*strides.q0 + q_head_idx*strides.q1 + seq_tile*WARPS_PER_BLOCK*WMMA_M*strides.q2;
    // const half* k_head = k + batch_idx*strides.kv0 + kv_head_idx*strides.kv1;
    // const half* v_head = v + batch_idx*strides.kv0 + kv_head_idx*strides.kv1;
    
    // half* out_row = out + batch_idx*n_heads*L*HEAD_DIM + q_head_idx*L*HEAD_DIM + seq_tile*WARPS_PER_BLOCK*WMMA_M*HEAD_DIM;
    // const uint8_t* mask_row = mask + batch_idx*L*L + seq_tile*WARPS_PER_BLOCK*WMMA_M*L; // [B, 1, L, L]

    // for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M; idx+=blockDim.x) {
    //     ms[idx] = HALF_MIN;
    //     ds[idx] = 0.0f;
    // }

    // for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
    //     out_shared[idx/HEAD_DIM][idx%HEAD_DIM] = __float2half(0.0f);
    // }

    // __syncthreads();

    // for (int n_tile=0; n_tile<L; n_tile+=WMMA_N) {
    //     const half* k_row = k_head + n_tile*strides.kv2;
    //     const half* v_row = v_head + n_tile*strides.kv2;
    //     const uint8_t* mask_tile = mask_row + n_tile;

    //     compute_tile_wmma_f16(
    //         L, HEAD_DIM, L,
    //         strides.q2, strides.kv2, WMMA_N,
    //         k_shared,
    //         q_row, k_row, (half*)l_shared
    //     );

    //     // apply mask and scale before computing ms and ds
    //     for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*WMMA_N; idx+=blockDim.x) {
    //         int r = idx / WMMA_N;
    //         int c = idx % WMMA_N;

    //         l_shared[r][c] = mask_tile[r*L+c] ? __float2half(__half2float(l_shared[r][c]) / sqrtf((float)HEAD_DIM)) : __float2half(HALF_MIN);
    //     }

    //     for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M; idx+=blockDim.x) {
    //         ms_old[idx] = ms[idx];
    //         ds_old[idx] = ds[idx];
    //     }

    //     __syncthreads();

    //     update_dm<WARPS_PER_BLOCK>(ds, ms, l_shared);

    //     __syncthreads();

    //     for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*WMMA_N; idx+=blockDim.x) {
    //         int r = idx/WMMA_N;
    //         int c = idx%WMMA_N;
    //         l_shared[r][c] = __float2half(expf(__half2float(l_shared[r][c])-ms[r]));
    //     }

    //     for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
    //         int r = idx/HEAD_DIM;
    //         int c = idx%HEAD_DIM;

    //         out_shared[r][c] = __float2half(__half2float(out_shared[r][c])*expf(ms_old[r]-ms[r]));
    //     }

    //     __syncthreads();


    //     #pragma unroll
    //     for (int n=0; n<HEAD_DIM; n+=WMMA_N) {
    //         // compute_tile_wmma_f16_T(
    //         //     WARPS_PER_BLOCK*WMMA_M, WMMA_N, HEAD_DIM,
    //         //     WMMA_N, strides.kv2, HEAD_DIM,
    //         //     (half*)l_shared, v_row + n,  &pv_shared[0][n]
    //         // );
    //     }

    //     __syncthreads();

    //     for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
    //         int r = idx/HEAD_DIM;
    //         int c = idx%HEAD_DIM;
    //         out_shared[r][c] = __float2half(__half2float(out_shared[r][c]) + __half2float(pv_shared[r][c]));
    //     }
    // }

    // for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
    //     int r = idx/HEAD_DIM;
    //     int c = idx%HEAD_DIM;
    //     out_row[r*HEAD_DIM+c] = (ds[r] > 0.0f) ? __float2half(__half2float(out_shared[r][c]) / ds[r]) : __float2half(0.0f);
    // }
}

}