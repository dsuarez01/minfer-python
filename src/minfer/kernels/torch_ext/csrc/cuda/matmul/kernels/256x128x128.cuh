#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"
#include "../helpers.cuh"

namespace minfer::impl {

template <
    unsigned int DIM_BM,
    unsigned int DIM_BK,
    unsigned int DIM_BN,
    unsigned int DIM_WM,
    unsigned int DIM_WK,
    unsigned int DIM_WN,
    unsigned int WARPS_K,
    unsigned int NUM_THRS
>
__global__ void xw_256x128x128(
    size_t M,
    size_t K,
    size_t N,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out
) {

    const unsigned int BLOCKS_K = (K+DIM_BK-1)/DIM_BK;

    constexpr unsigned int DIM_MM = 16;
    constexpr unsigned int DIM_MK = 16;
    constexpr unsigned int DIM_MN = 8;

    constexpr unsigned int MMAS_M = (DIM_WM+DIM_MM-1)/DIM_MM;
    constexpr unsigned int MMAS_K = (DIM_WK+DIM_MK-1)/DIM_MK;
    constexpr unsigned int MMAS_N = (DIM_WN+DIM_MN-1)/DIM_MN;

    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;
    const unsigned int warp_m = threadIdx.y;
    const unsigned int warp_n = threadIdx.x/32;
    const unsigned int lane_idx = threadIdx.x%32;

    const size_t stride_x = K;
    const size_t stride_w = N;
    const size_t stride_out = N;

    extern __shared__ half shmem[];
    half* shmem_x = shmem;
    half* shmem_w = &shmem[DIM_BM*DIM_BK];

    uint32_t x_reg[MMAS_M][MMAS_K][4];
    uint32_t w_reg[MMAS_K][MMAS_N][2];
    uint32_t acc_reg[MMAS_M][MMAS_N][2];

    memset(acc_reg, 0, sizeof(acc_reg));

    // for each K block tile
    for (int block_k = 0; block_k < BLOCKS_K; ++block_k) {
        
        const half* block_x = x + block_m * DIM_BM * stride_x + block_k * DIM_BK;
        const half* block_w = w + block_k * DIM_BK * stride_w + block_n * DIM_BN;

        // load from gmem to x_shmem and w_shmem
        // toShmem<DIM_MM, DIM_BM, DIM_BK, NUM_THRS>(stride_x, block_x, shmem_x);
        // toShmem<DIM_MK, DIM_BK, DIM_BN, NUM_THRS>(stride_w, block_w, shmem_w);
        toShmem<DIM_BM, DIM_BK, NUM_THRS>(stride_x, block_x, shmem_x);
        toShmem<DIM_BK, DIM_BN, NUM_THRS>(stride_w, block_w, shmem_w);

        __syncthreads();
    
        for (int warp_k = 0; warp_k < WARPS_K; ++warp_k) {

            const half* warp_x = shmem_x + warp_m * DIM_WM * DIM_BK + warp_k * DIM_WK;
            const half* warp_w = shmem_w + warp_k * DIM_WK * DIM_BN + warp_n * DIM_WN;

            uint32_t byte_ofst_warp_x = __cvta_generic_to_shared(warp_x);
            uint32_t byte_ofst_warp_w = __cvta_generic_to_shared(warp_w);

            // load from shmem into registers
            for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) { // x_shmem
                for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) {
                    ldmatrix_m16n16<DIM_MM,DIM_MK,DIM_BK>(
                        mma_m,
                        mma_k,
                        lane_idx,
                        byte_ofst_warp_x,
                        x_reg[mma_m][mma_k]
                    );
                }
            }

            for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) { // w_shmem
                for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
                    ldmatrix_m16n8<DIM_MK,DIM_MN,DIM_BN>(
                        mma_k,
                        mma_n,
                        lane_idx,
                        byte_ofst_warp_w,
                        w_reg[mma_k][mma_n]
                    );
                }
            }

            for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) {

                for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) {

                    for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
                        mma_sync_m16n8k16(
                            mma_m,
                            mma_k,
                            mma_n,
                            x_reg[mma_m][mma_k],
                            w_reg[mma_k][mma_n],
                            acc_reg[mma_m][mma_n]
                        );
                    }
                }
            }
        }

        __syncthreads();
    }

    half* block_out = out + block_m * DIM_BM * stride_out + block_n * DIM_BN;
    half* warp_out = block_out + warp_m * DIM_WM * stride_out + warp_n * DIM_WN;

    for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) {
        for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
            half* mma_out = warp_out + mma_m * DIM_MM * stride_out + mma_n * DIM_MN;
            stmatrix_m16n8(
                lane_idx,
                stride_out * sizeof(half), 
                mma_out, 
                acc_reg[mma_m][mma_n]
            );
        }
    }
}

}