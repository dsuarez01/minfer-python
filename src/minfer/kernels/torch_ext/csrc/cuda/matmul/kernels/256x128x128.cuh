#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

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

    constexpr unsigned int BLOCKS_K = (K+DIM_BK-1)/DIM_BK;

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

    const unsigned int stride_x = K;
    const unsigned int stride_w = N;
    const unsigned int stride_out = N;

    extern __shared__ half shmem[];
    half* shmem_x = shmem;
    half* shmem_w = &shmem[DIM_BM*DIM_BK];

    // declare, init registers
    uint32_t acc_reg[MMAS_M][MMAS_N][2];
    half (&acc_reg_)[MMAS_M][MMAS_N][4] = reinterpret_cast<half(&)[MMAS_M][MMAS_N][4]>(acc_reg);

    for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) {
        for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
            acc_reg_[mma_m][mma_n][0] = 0;
            acc_reg_[mma_m][mma_n][1] = 0;
            acc_reg_[mma_m][mma_n][2] = 0;
            acc_reg_[mma_m][mma_n][3] = 0;
        }
    }

    uint32_t x_reg[MMAS_M][MMAS_K][4];
    uint32_t w_reg[MMAS_K][MMAS_N][2];


    // for each K block tile
    for (int block_k = 0; block_k < BLOCKS_K; ++block_k) {
        
        half* block_x = x + block_m * DIM_BM * stride_x + block_k * DIM_BK;
        half* block_w = w + block_k * DIM_BK * stride_w + block_n * DIM_BN;

        // load from gmem to x_shmem and w_shmem
        toShmem<DIM_BM, DIM_BK, NUM_THRS>(K, block_x, shmem_x);
        toShmem<DIM_BK, DIM_BN, NUM_THRS>(N, block_w, shmem_w);

        __syncthreads();
    
        for (int warp_k = 0; warp_k < WARPS_K; ++warp_k) {

            half* warp_x = block_x + warp_m * DIM_WM * DIM_BK + warp_k * DIM_WK;
            half* warp_w = block_w + warp_k * DIM_WK * DIM_BN + warp_n * DIM_WN;

            uint32_t byte_ofst_warp_x = cvta_to_shared_u32(warp_x);
            uint32_t byte_ofst_warp_w = cvta_to_shared_u32(warp_w);

            // load from shmem into registers
            for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) { // x_shmem
                for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) {
                    size_t byte_ofst_mma_x = (mma_m * DIM_MM * DIM_BK + mma_k * DIM_MK) * sizeof(half);

                    size_t byte_ofst_thr_x = (threadIdx.x % DIM_MM) * DIM_BK * sizeof(half);

                    size_t tot_byte_ofst_x = byte_ofst_warp_x + byte_ofst_mma_x + byte_ofst_thr_x;

                    // ld matrix
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                        "{%0, %1, %2, %3}, [%4];"
                        : "=r"(x_reg[mma_m][mma_k][0]), "=r"(x_reg[mma_m][mma_k][1]), 
                          "=r"(x_reg[mma_m][mma_k][2]), "=r"(x_reg[mma_m][mma_k][3])
                        : "r"(tot_byte_ofst_x)
                    );
                }
            }

            for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) { // w_shmem
                for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
                    size_t byte_ofst_mma_w = (mma_k * DIM_MK * DIM_BN + mma_n * DIM_MN) * sizeof(half);

                    size_t byte_ofst_thr_w = (threadIdx.x % DIM_MK) * DIM_BN * sizeof(half);

                    size_t tot_byte_ofst_w = byte_ofst_warp_w + byte_ofst_mma_w + byte_ofst_thr_w;

                    // ld matrix
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                        "{%0, %1}, [%2];"
                        : "=r"(w_reg[mma_k][mma_n][0]), "=r"(w_reg[mma_k][mma_n][1])
                        : "r"(tot_byte_ofst_w)
                    );
                }
            }

            for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) {

                for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) {

                    for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
                        // mma sync on x fragment and w fragment
                        asm volatile (
                            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, "
                            "{%2, %3, %4, %5}, "
                            "{%6, %7}, "
                            "{%8, %9};"
                            : "=r"(acc_reg[mma_m][mma_n][0]), "=r"(acc_reg[mma_m][mma_n][1])
                            : "r"(x_reg[mma_m][mma_k][0]), "r"(x_reg[mma_m][mma_k][1]), "r"(x_reg[mma_m][mma_k][2]), "r"(x_reg[mma_m][mma_k][3])
                              "r"(w_reg[mma_k][mma_n][0]), "r"(w_reg[mma_k][mma_n][1])
                              "r"(acc_reg[mma_m][mma_n][0]), "r"(acc_reg[mma_m][mma_n][1])
                        );
                    }
                }
            }
        }

        __syncthreads();
    }

    half* block_out = out + block_m * DIM_BM * out_stride + block_n * DIM_BN;
    half* warp_out = block_out + warp_m * DIM_WM * out_stride + warp_n * DIM_WN;

    for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) {
        for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {
            half* mma_out = warp_out + mma_m * DIM_MM * out_stride + mma_n * DIM_MN;
            toGmem_m16n8(out_stride * sizeof(half), mma_out, acc_reg_[mma_m][mma_n]);
        }
    }
}