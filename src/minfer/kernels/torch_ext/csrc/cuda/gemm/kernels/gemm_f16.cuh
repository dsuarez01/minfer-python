#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include "../helpers.cuh"

namespace minfer::impl {

template <
    unsigned int DIM_BM,
    unsigned int DIM_BK,
    unsigned int DIM_BN,
    unsigned int DIM_WM,
    unsigned int DIM_WK,
    unsigned int DIM_WN,
    unsigned int DIM_MM,
    unsigned int DIM_MK,
    unsigned int DIM_MN,
    unsigned int TILES_K,
    unsigned int NUM_THRS
>
__global__ void ab_sync_impl(
    size_t M,
    size_t K,
    size_t N,
    float alpha,
    float beta,
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ C,
    half* __restrict__ D
) {

    const unsigned int blocks_k = (K+DIM_BK-1)/DIM_BK;

    constexpr unsigned int MMAS_M = (DIM_WM+DIM_MM-1)/DIM_MM;
    constexpr unsigned int MMAS_K = (DIM_WK+DIM_MK-1)/DIM_MK;
    constexpr unsigned int MMAS_N = (DIM_WN+DIM_MN-1)/DIM_MN;

    // for caching gmem loads into regs
    constexpr unsigned int ELEMS_THR_A = DIM_BM/(NUM_THRS/(DIM_BK/8));
    constexpr unsigned int ELEMS_THR_B = DIM_BK/(NUM_THRS/(DIM_BN/8));

    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;
    const unsigned int warp_m = threadIdx.y;
    const unsigned int warp_n = threadIdx.x/32;
    const unsigned int lane_idx = threadIdx.x%32;

    const size_t stride_A = K;
    const size_t stride_B = N;

    extern __shared__ half shmem[];
    half* shmem_base_A = shmem;
    half* shmem_base_B = &shmem[2*DIM_BM*DIM_BK];

    uint32_t regs_A[TILES_K][MMAS_M][MMAS_K][(DIM_MM/8)*(DIM_MK/8)];
    uint32_t regs_B[TILES_K][MMAS_K][MMAS_N][(DIM_MK/8)*(DIM_MN/8)];
    uint32_t regs_acc[MMAS_M][MMAS_N][(DIM_MM/8)*(DIM_MN/8)];

    int4 cache_regs_A[ELEMS_THR_A];
    int4 cache_regs_B[ELEMS_THR_B];

    // load beta*C into regs_acc, use shmem_base_CD as staging area
    {
        const size_t stride_CD = N;
        const half* block_C = C + block_m * DIM_BM * stride_CD + block_n * DIM_BN;
        half* shmem_base_CD = shmem;
        half* shmem_warp_CD = shmem_base_CD + warp_m * DIM_WM * DIM_BN + warp_n * DIM_WN;
        prologue<DIM_MM, DIM_MN, MMAS_M, MMAS_N, DIM_BM, DIM_BN, NUM_THRS>(lane_idx, stride_CD, alpha, beta, block_C, shmem_base_CD, shmem_warp_CD, regs_acc);
    }
    
    __syncthreads();

    // fetch first block (block_k=0) to shmem
    const half* block_A = A + block_m * DIM_BM * stride_A + 0 * DIM_BK;
    const half* block_B = B + 0 * DIM_BK * stride_B + block_n * DIM_BN;

    toShmemSync<DIM_MM, DIM_BM, DIM_BK, NUM_THRS>(stride_A, block_A, shmem_base_A);
    toShmemSync<DIM_MK, DIM_BK, DIM_BN, NUM_THRS>(stride_B, block_B, shmem_base_B);
    
    __syncthreads();

// i.e. do not unroll
#pragma unroll 1
    for (int block_k = 0; block_k < blocks_k; ++block_k) {

        if (block_k != blocks_k-1) {
            const half* block_A = A + block_m * DIM_BM * stride_A + (block_k+1) * DIM_BK;
            const half* block_B = B + (block_k+1) * DIM_BK * stride_B + block_n * DIM_BN;

            // prefetch block_k+1 into regs
            gmemToRegSync<DIM_MM, DIM_BM, DIM_BK, NUM_THRS, ELEMS_THR_A>(stride_A, block_A, cache_regs_A);
            gmemToRegSync<DIM_MK, DIM_BK, DIM_BN, NUM_THRS, ELEMS_THR_B>(stride_B, block_B, cache_regs_B);
        }

        const int shmem_pipe_read = block_k & 1;
        const int shmem_pipe_write = shmem_pipe_read ^ 1;
    
        const half* shmem_read_A = shmem_base_A + shmem_pipe_read * DIM_BM * DIM_BK;
        const half* shmem_read_B = shmem_base_B + shmem_pipe_read * DIM_BK * DIM_BN;

        const half* warp_A = shmem_read_A + warp_m * DIM_WM * DIM_BK + 0 * DIM_WK;
        const half* warp_B = shmem_read_B + 0 * DIM_WK * DIM_BN + warp_n * DIM_WN;

        ldmatrix<false, DIM_MM, DIM_MK, MMAS_M, MMAS_K, DIM_BK>(lane_idx, warp_A, regs_A[0]);
        ldmatrix<true, DIM_MK, DIM_MN, MMAS_K, MMAS_N, DIM_BN>(lane_idx, warp_B, regs_B[0]);

#pragma unroll
        for (int tile_k = 0; tile_k < TILES_K; ++tile_k) {

            // x,w shmem -> regs for tile_k+1
            if (tile_k != TILES_K-1) {
                const half* warp_A = shmem_read_A + warp_m * DIM_WM * DIM_BK + (tile_k+1) * DIM_WK;
                const half* warp_B = shmem_read_B + (tile_k+1) * DIM_WK * DIM_BN + warp_n * DIM_WN;

                ldmatrix<false, DIM_MM, DIM_MK, MMAS_M, MMAS_K, DIM_BK>(lane_idx, warp_A, regs_A[tile_k+1]);
                ldmatrix<true, DIM_MK, DIM_MN, MMAS_K, MMAS_N, DIM_BN>(lane_idx, warp_B, regs_B[tile_k+1]);
            }
            
            // compute on tile_k
            mma_sync<DIM_MM, DIM_MK, DIM_MN, MMAS_M, MMAS_K, MMAS_N>(regs_A[tile_k], regs_B[tile_k], regs_acc);
        }

        if (block_k != blocks_k-1) {
            half* shmem_write_A = shmem_base_A + shmem_pipe_write * DIM_BM * DIM_BK;
            half* shmem_write_B = shmem_base_B + shmem_pipe_write * DIM_BK * DIM_BN;

            // load block_k+1 into shmem
            regToShmemSync<DIM_MM, DIM_BM, DIM_BK, NUM_THRS, ELEMS_THR_A>(shmem_write_A, cache_regs_A);
            regToShmemSync<DIM_MK, DIM_BK, DIM_BN, NUM_THRS, ELEMS_THR_B>(shmem_write_B, cache_regs_B);
        }

        __syncthreads();
    }
 
    // acc regs -> alpha*AB -> shmem
    {
        const size_t stride_CD = N;
        half* block_D = D + block_m * DIM_BM * stride_CD + block_n * DIM_BN;
        half* shmem_base_CD = shmem;
        half* shmem_warp_CD = shmem_base_CD + warp_m * DIM_WM * DIM_BN + warp_n * DIM_WN;
        epilogue<DIM_MM, DIM_MN, MMAS_M, MMAS_N, DIM_BM, DIM_BN, NUM_THRS>(lane_idx, stride_CD, alpha, shmem_warp_CD, shmem_base_CD, block_D, regs_acc);
    }
}

// improvement 3: pipelining
template <
    unsigned int DIM_BM,
    unsigned int DIM_BK,
    unsigned int DIM_BN,
    unsigned int DIM_WM,
    unsigned int DIM_WK,
    unsigned int DIM_WN,
    unsigned int DIM_MM,
    unsigned int DIM_MK,
    unsigned int DIM_MN,
    unsigned int TILES_K,
    unsigned int K_PIPE_MAX,
    unsigned int NUM_THRS
>
__global__ void ab_async_impl(
    size_t M,
    size_t K,
    size_t N,
    float alpha,
    float beta,
    const half* __restrict__ A,
    const half* __restrict__ B,
    const half* __restrict__ C,
    half* __restrict__ D
) {

    const unsigned int blocks_k = (K+DIM_BK-1)/DIM_BK;

    constexpr unsigned int MMAS_M = (DIM_WM+DIM_MM-1)/DIM_MM;
    constexpr unsigned int MMAS_K = (DIM_WK+DIM_MK-1)/DIM_MK;
    constexpr unsigned int MMAS_N = (DIM_WN+DIM_MN-1)/DIM_MN;

    const unsigned int block_m = blockIdx.y;
    const unsigned int block_n = blockIdx.x;
    const unsigned int warp_m = threadIdx.y;
    const unsigned int warp_n = threadIdx.x/32;
    const unsigned int lane_idx = threadIdx.x%32;

    const size_t stride_A = K;
    const size_t stride_B = N;

    extern __shared__ half shmem[];
    half* shmem_base_A = shmem;
    half* shmem_base_B = &shmem[K_PIPE_MAX*DIM_BM*DIM_BK];

    uint32_t regs_A[TILES_K][MMAS_M][MMAS_K][(DIM_MM/8)*(DIM_MK/8)];
    uint32_t regs_B[TILES_K][MMAS_K][MMAS_N][(DIM_MK/8)*(DIM_MN/8)];
    uint32_t regs_acc[MMAS_M][MMAS_N][(DIM_MM/8)*(DIM_MN/8)];

    // load beta*C into regs_acc, use shmem_base_CD as staging area
    {
        const size_t stride_CD = N;
        const half* block_C = C + block_m * DIM_BM * stride_CD + block_n * DIM_BN;
        half* shmem_base_CD = shmem;
        half* shmem_warp_CD = shmem_base_CD + warp_m * DIM_WM * DIM_BN + warp_n * DIM_WN;
        prologue<DIM_MM, DIM_MN, MMAS_M, MMAS_N, DIM_BM, DIM_BN, NUM_THRS>(lane_idx, stride_CD, alpha, beta, block_C, shmem_base_CD, shmem_warp_CD, regs_acc);
    }

    __syncthreads();

    // pipeline ctrl (https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
    int k_block_count = blocks_k;
    int k_block_next = 0;

    // prefetch first K_PIPE_MAX-1 tiles
#pragma unroll 1
    for (int k_pipe=0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
        const half* block_A = A + block_m*DIM_BM*stride_A + k_block_next*DIM_BK;
        const half* block_B = B + k_block_next*DIM_BK*stride_B + block_n*DIM_BN;
        
        half* shmem_A = shmem_base_A + k_pipe * DIM_BM * DIM_BK;
        half* shmem_B = shmem_base_B + k_pipe * DIM_BK * DIM_BN;

        toShmemAsync<DIM_MM, DIM_BM, DIM_BK, NUM_THRS>(stride_A, block_A, shmem_A);
        toShmemAsync<DIM_MK, DIM_BK, DIM_BN, NUM_THRS>(stride_B, block_B, shmem_B);
        cp_async_fence();

        --k_block_count;
        if (k_block_count > 0) ++k_block_next;
    }

    int shmem_pipe_read = 0;
    int shmem_pipe_write = K_PIPE_MAX-1;

    const half* shmem_read_A = shmem_base_A + shmem_pipe_read * DIM_BM * DIM_BK;
    const half* shmem_read_B = shmem_base_B + shmem_pipe_read * DIM_BK * DIM_BN;

    // preload first frag into regs
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    const half* warp_A = shmem_read_A + warp_m * DIM_WM * DIM_BK + 0 * DIM_WK;
    const half* warp_B = shmem_read_B + 0 * DIM_WK * DIM_BN + warp_n * DIM_WN;

    ldmatrix<false, DIM_MM, DIM_MK, MMAS_M, MMAS_K, DIM_BK>(lane_idx, warp_A, regs_A[0]);
    ldmatrix<true, DIM_MK, DIM_MN, MMAS_K, MMAS_N, DIM_BN>(lane_idx, warp_B, regs_B[0]);

    // for each K block tile (this iterates blocks_k times)
#pragma unroll 1
    while (k_block_count > -((int)K_PIPE_MAX-1)) {
        
#pragma unroll
        for (int tile_k = 0; tile_k < TILES_K; ++tile_k) {

            if (tile_k == 0) {
                const half* block_A = A + block_m*DIM_BM*stride_A + k_block_next*DIM_BK;
                const half* block_B = B + k_block_next*DIM_BK*stride_B + block_n*DIM_BN;
                
                half* shmem_write_A = shmem_base_A + shmem_pipe_write * DIM_BM * DIM_BK;
                half* shmem_write_B = shmem_base_B + shmem_pipe_write * DIM_BK * DIM_BN;

                toShmemAsync<DIM_MM, DIM_BM, DIM_BK, NUM_THRS>(stride_A, block_A, shmem_write_A);
                toShmemAsync<DIM_MK, DIM_BK, DIM_BN, NUM_THRS>(stride_B, block_B, shmem_write_B);
                cp_async_fence();

                --k_block_count;
                if (k_block_count > 0) ++k_block_next;
                shmem_pipe_write = shmem_pipe_read; // (circular buffer)
                shmem_pipe_read = (shmem_pipe_read == K_PIPE_MAX-1) ? 0 : shmem_pipe_read+1;
            }

            // mma on regs for tile_k
            mma_sync<DIM_MM, DIM_MK, DIM_MN, MMAS_M, MMAS_K, MMAS_N>(
                regs_A[tile_k],
                regs_B[tile_k],
                regs_acc
            );

            if (tile_k == TILES_K-1) {

                shmem_read_A = shmem_base_A + shmem_pipe_read * DIM_BM * DIM_BK;
                shmem_read_B = shmem_base_B + shmem_pipe_read * DIM_BK * DIM_BN;
                
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // x,w shmem -> regs for tile_k_nxt
            const int tile_k_nxt = (tile_k == TILES_K-1) ? 0 : tile_k+1;
            const half* warp_A = shmem_read_A + warp_m * DIM_WM * DIM_BK + tile_k_nxt * DIM_WK;
            const half* warp_B = shmem_read_B + tile_k_nxt * DIM_WK * DIM_BN + warp_n * DIM_WN;
            ldmatrix<false, DIM_MM, DIM_MK, MMAS_M, MMAS_K, DIM_BK>(lane_idx, warp_A, regs_A[tile_k_nxt]);
            ldmatrix<true, DIM_MK, DIM_MN, MMAS_K, MMAS_N, DIM_BN>(lane_idx, warp_B, regs_B[tile_k_nxt]);
        }
    }

    // worst case is probably: ((K_PIPE_MAX-1)*(DIM_BM*DIM_BK+DIM_BK*DIM_BN)*2 bytes)/(864e9 bytes/s)
    // likely negligible compared to kernel runtime
    cp_async_wait<0>();
    __syncthreads();

    // regs -> shmem -> gmem
    {
        const size_t stride_CD = N;
        half* block_D = D + block_m * DIM_BM * stride_CD + block_n * DIM_BN;
        half* shmem_base_CD = shmem;
        half* shmem_warp_CD = shmem_base_CD + warp_m * DIM_WM * DIM_BN + warp_n * DIM_WN;
        epilogue<DIM_MM, DIM_MN, MMAS_M, MMAS_N, DIM_BM, DIM_BN, NUM_THRS>(lane_idx, stride_CD, alpha, shmem_warp_CD, shmem_base_CD, block_D, regs_acc);
    }
}

}