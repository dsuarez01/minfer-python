#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

// helper to compute TILE_SIZE_M x TILE_SIZE_N portion of result (X@W.T)
template<typename Config>
__device__ __forceinline__ void compute_tile_wmma_f16(
    int tile_i, int tile_j,
    int m_tiles, int n_tiles,
    int64_t M, int64_t K, int64_t N,
    half (*shmem)[Config::CHUNK_K*WMMA_K+Config::SKEW_HALF],
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out
) {
    static constexpr auto WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    static constexpr auto W_COL_MAJOR = Config::W_COL_MAJOR;
    static constexpr auto CHUNK_K = Config::CHUNK_K;
    static constexpr auto SKEW_HALF = Config::SKEW_HALF;
    static constexpr auto WARP_ROW_TILES = Config::WARP_ROW_TILES;
    static constexpr auto WARP_COL_TILES = Config::WARP_COL_TILES;
    static constexpr auto BLOCK_ROW_WARPS = Config::BLOCK_ROW_WARPS;
    static constexpr auto BLOCK_COL_WARPS = Config::BLOCK_COL_WARPS;
    static constexpr auto BLOCK_ROW_TILES = Config::BLOCK_ROW_TILES;
    static constexpr auto BLOCK_COL_TILES = Config::BLOCK_COL_TILES;
    static constexpr auto SHMEM_STRIDE = Config::SHMEM_STRIDE;
    static constexpr auto SHMEM_OFFSET = Config::SHMEM_OFFSET;
    static constexpr auto CHUNK_COPY_LINE_LANES = Config::CHUNK_COPY_LINE_LANES;
    static constexpr auto CHUNK_COPY_LINES_PER_WARP = Config::CHUNK_COPY_LINES_PER_WARP;
    static constexpr auto TILE_SIZE_M = Config::TILE_SIZE_M;
    static constexpr auto TILE_SIZE_N = Config::TILE_SIZE_N;

    const unsigned int warpId = threadIdx.x/32;
    const unsigned int laneId = threadIdx.x%32;

    const size_t shmem_idx_w_off = WMMA_M*BLOCK_ROW_TILES;
    float* shmem_warp_tile_ptr = (float*)&shmem[0][0] 
                               + (warpId/BLOCK_COL_WARPS)*WARP_ROW_TILES*WMMA_M*SHMEM_STRIDE
                               + (warpId%BLOCK_COL_WARPS)*SHMEM_OFFSET;
    float* shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId*WMMA_M*SHMEM_STRIDE;

    using W_LAYOUT = std::conditional_t<W_COL_MAJOR, wmma::col_major, wmma::row_major>;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> x_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, W_LAYOUT> w_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {
        for (int j=0; j<WARP_COL_TILES; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int64_t w_stride = W_COL_MAJOR ? K : N;
    const int64_t w_tile = W_COL_MAJOR ? WMMA_N : WMMA_K;
    const half* warp_ptr = (warpId<(WARPS_PER_BLOCK/2)) 
                        ? (x+WMMA_M*K*(warpId%WARP_COL_TILES)*WARP_ROW_TILES) 
                        : (w+w_tile*w_stride*(warpId%WARP_COL_TILES)*WARP_ROW_TILES);

    int K_TILES = (K+WMMA_K-1)/WMMA_K;

    // iterating over global K dimension, CHUNK_K 16x16 tiles at a time
    for (int tile_k=0; tile_k<K_TILES; tile_k+=CHUNK_K) {
        
        // load X and W into shared memory
        // each warp loads 32 rows of X (warps 0-3) or W (warps 4-7)
        size_t shmem_idx = (warpId<(WARPS_PER_BLOCK/2)) 
                            ? (WMMA_M * (warpId%WARP_COL_TILES) * WARP_ROW_TILES)
                            : (WMMA_N * (warpId%WARP_COL_TILES) * WARP_ROW_TILES + shmem_idx_w_off);

        const int64_t xw_stride = (warpId < (WARPS_PER_BLOCK/2)) ? K : w_stride;
        int4* lane_ptr = (int4*)(warp_ptr 
            + (laneId/CHUNK_COPY_LINE_LANES)*xw_stride 
            + tile_k*WMMA_K 
            + (laneId%CHUNK_COPY_LINE_LANES)*8
        );

        shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

        #pragma unroll
        for (int i=0; i<((WARP_SIZE/2)/CHUNK_COPY_LINES_PER_WARP)*WARP_ROW_TILES; ++i) {
            *((int4*)&shmem[shmem_idx][0] + (laneId%CHUNK_COPY_LINE_LANES)) = *lane_ptr;

            lane_ptr = (int4*)((half*)lane_ptr + CHUNK_COPY_LINES_PER_WARP*xw_stride);
            shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

        // process CHUNK_K 16x16 tiles, accumulate to acc
        #pragma unroll
        for (int k_step=0; k_step<CHUNK_K; ++k_step) {

            // iterate over grid w/in warp, compute tiles for acc
            #pragma unroll
            for (int i=0; i<WARP_ROW_TILES; ++i) {
                size_t  shmem_idx_x = (warpId/WARP_ROW_TILES)*WARP_ROW_TILES*WMMA_M + (i*WMMA_M);
                const half* tile_ptr = &shmem[shmem_idx_x][k_step*WMMA_K];

                wmma::load_matrix_sync(x_frag[i], tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);

                #pragma unroll
                for (int j=0; j<WARP_COL_TILES; ++j) {
                    if (i==0) {
                        size_t shmem_idx_w = shmem_idx_w_off + (warpId%2)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                        const half* tile_ptr = &shmem[shmem_idx_w][k_step*WMMA_K];

                        wmma::load_matrix_sync(w_frag[j], tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                    }

                    wmma::mma_sync(acc[i][j], x_frag[i], w_frag[j], acc[i][j]);
                }

            }

            __syncthreads();
        }
    }


    // acc fragments stored to shmem
    if (warpId*WMMA_M < TILE_SIZE_M) {
        #pragma unroll
        for (int i=0; i<WARP_ROW_TILES; ++i) {
            #pragma unroll
            for (int j=0; j<WARP_COL_TILES; ++j) {
                float* tile_ptr = shmem_warp_tile_ptr + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N; 
                wmma::store_matrix_sync(tile_ptr, acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);
            }
        }
    }

    __syncthreads();

    // convert float data to half in shmem
    if (warpId*WMMA_M < TILE_SIZE_M) {
        #pragma unroll
        for (int i=0; i<WMMA_M; ++i) {
            if (laneId*8 < TILE_SIZE_N) {
                float4 f_data = *(float4*)((shmem_warp_stream_ptr+i*SHMEM_STRIDE+laneId*4));
                half2 h_data[2];
                h_data[0] = __float22half2_rn(make_float2(f_data.x, f_data.y));
                h_data[1] = __float22half2_rn(make_float2(f_data.z, f_data.w));
                *((int2*)(shmem_warp_stream_ptr+i*SHMEM_STRIDE+laneId*4)) = *((int2*)h_data);
            }
        }
    }

    __syncthreads();

    // stream tiles of out (in shmem) to gmem
    if (warpId*WMMA_M < TILE_SIZE_M) {
        
        const size_t gmem_idx = warpId*WMMA_M*N;
        half* dst_gmem_warp_stream_ptr = &out[gmem_idx];

        #pragma unroll
        for (int i=0; i<WMMA_M; ++i) {
            *((int2*)(dst_gmem_warp_stream_ptr+i*N)+laneId) = *((int2*)(shmem_warp_stream_ptr+i*SHMEM_STRIDE)+laneId*2);
        }
    }
}