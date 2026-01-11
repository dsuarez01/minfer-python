#pragma once

#include <cuda_fp16.h>

// helper (TODO: complete me once you know exactly where this will be used)
template <int WARPS_PER_BLOCK>
__device__ __forceinline__ void update_dm(
    float* __restrict__ ds,
    float* __restrict__ ms,
    const half (&tile)[WARPS_PER_BLOCK*WMMA_M][WMMA_N]
) {
    int warpId = threadIdx.x/32;
    int lane = threadIdx.x%32;
    int rowInWarp = lane/2;
    int thrInRow = lane%2;
    int globalRow = warpId*WMMA_M+rowInWarp;

    // row max
    float localMax = -INFINITY;
    for (int c=thrInRow; c<WMMA_N; c+=2) {
        localMax = fmaxf(localMax, __half2float(tile[globalRow][c]));
    }
    float partnerMax = __shfl_xor_sync(0xffffffff, localMax, 1);
    float row_m = fmaxf(localMax, partnerMax);

    // row sum
    float localSum = 0.0f;
    for (int c=thrInRow; c<WMMA_N; c+=2) {
        localSum += expf(__half2float(tile[globalRow][c])-row_m);
    }
    float partnerSum = __shfl_xor_sync(0xffffffff, localSum, 1);
    float row_d = localSum + partnerSum;

    if (thrInRow == 0) {
        float m_old = ms[globalRow];
        float d_old = ds[globalRow];
        float m_new = fmaxf(m_old, row_m);
        float d_new = d_old*expf(m_old-m_new) + row_d*expf(row_m-m_new);
        ms[globalRow] = m_new;
        ds[globalRow] = d_new;
    }
}