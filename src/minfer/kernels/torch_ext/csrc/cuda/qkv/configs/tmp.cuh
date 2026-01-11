#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

// helper to compute TILE_SIZE_M x TILE_SIZE_N portion of result for QKV projections
// reduces loads from global mem of input x by x3 factor
template <typename Config>
__device__ __forceinline__ void compute_qkv_tile_f16(
    int tile_i, int tile_j,
    int m_tiles, int n_kv_tiles,
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    half (*shmem)[Config::CHUNK_K*WMMA_K+Config::SKEW_HALF],
    const half* __restrict__ x,
    const half* __restrict__ wq,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {
    static constexpr auto WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    static constexpr auto CHUNK_K = Config::CHUNK_K;
    static constexpr auto SKEW_HALF = Config::SKEW_HALF;
    static constexpr auto WARP_ROW_TILES = Config::WARP_ROW_TILES;
    static constexpr auto WARP_COL_TILES = Config::WARP_COL_TILES;
    static constexpr auto BLOCK_ROW_WARPS = Config::BLOCK_ROW_WARPS;
    static constexpr auto BLOCK_COL_WARPS = Config::BLOCK_COL_WARPS;
    static constexpr auto SHMEM_STRIDE = Config::SHMEM_STRIDE;
    static constexpr auto SHMEM_OFFSET = Config::SHMEM_OFFSET;
    static constexpr auto CHUNK_COPY_LINE_LANES = Config::CHUNK_COPY_LINE_LANES;
    static constexpr auto CHUNK_COPY_LINES_PER_WARP = Config::CHUNK_COPY_LINES_PER_WARP;
    static constexpr auto TILE_SIZE_M = Config::TILE_SIZE_M;
    static constexpr auto TILE_SIZE_N = Config::TILE_SIZE_N;

    const bool compute_kv = (tile_j < n_kv_tiles);

    const unsigned int warpId = threadIdx.x/32;
    const unsigned int laneId = threadIdx.x%32;

    const size_t shmem_idx_wq_off = TILE_SIZE_M;
    const size_t shmem_idx_wk_off = 2*TILE_SIZE_M;
    const size_t shmem_idx_wv_off = 3*TILE_SIZE_M;

    const size_t shmem_q_out_off = 0;
    const size_t shmem_k_out_off = TILE_SIZE_M*TILE_SIZE_N;
    const size_t shmem_v_out_off = 2*TILE_SIZE_M*TILE_SIZE_N;

    float* shmem_warp_tile_ptr = (float*)&shmem[0][0]
                               + (warpId/BLOCK_COL_WARPS)*WMMA_M*WARP_ROW_TILES*SHMEM_STRIDE
                               + (warpId%BLOCK_COL_WARPS)*SHMEM_OFFSET;
    float* shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId*WMMA_M*SHMEM_STRIDE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> x_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> wq_frag[WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> wk_frag[WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> wv_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> q_acc[WARP_ROW_TILES][WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> k_acc[WARP_ROW_TILES][WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> v_acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {
        for (int j=0; j<WARP_COL_TILES; ++j) {
            wmma::fill_fragment(q_acc[i][j], 0.0f);
            if (compute_kv) {
                wmma::fill_fragment(k_acc[i][j], 0.0f);
                wmma::fill_fragment(v_acc[i][j], 0.0f);
            }
            
        }
    }

    const half* warp_ptr = (warpId<(WARPS_PER_BLOCK/4)) ? (x+(warpId%2)*(TILE_SIZE_M/2)*K) 
                         : (warpId<2*(WARPS_PER_BLOCK/4)) ? (wq+(warpId%2)*(TILE_SIZE_N/2)*K)
                         : (warpId<3*(WARPS_PER_BLOCK/4)) ? (wk+(warpId%2)*(TILE_SIZE_N/2)*K)
                         : (wv+(warpId%2)*(TILE_SIZE_N/2)*K);

    int K_TILES = (K+WMMA_K-1)/WMMA_K;

    // iterating over global K dimension, CHUNK_K 16x16 tiles at a time
    for (int tile_k=0; tile_k<K_TILES; tile_k+=CHUNK_K) {
        
        // load X, WQ, WK, WV into shared memory
        // 2 warps per X, WQ, WK, WV
        size_t shmem_idx = (warpId<(WARPS_PER_BLOCK/2)) ? ((warpId%2)*(TILE_SIZE_M/2))
                       : (warpId<2*(WARPS_PER_BLOCK/2)) ? ((warpId%2)*(TILE_SIZE_N/2) + shmem_idx_wq_off)
                       : (warpId<3*(WARPS_PER_BLOCK/2)) ? ((warpId%2)*(TILE_SIZE_N/2) + shmem_idx_wk_off)
                       : ((warpId%2)*(TILE_SIZE_N/2) + shmem_idx_wv_off);

        int4* lane_ptr = (int4*)(
            warp_ptr 
            + (laneId/CHUNK_COPY_LINE_LANES)*K 
            + tile_k*WMMA_K 
            + (laneId%CHUNK_COPY_LINE_LANES)
        );

        shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

        if (warpId < (WARPS_PER_BLOCK/4)) {
            // X has TILE_M rows to load
            #pragma unroll
            for (int i=0; i<(TILE_SIZE_M/2)/CHUNK_COPY_LINES_PER_WARP; ++i) {
                *((int4*)&shmem[shmem_idx][0] + (laneId%CHUNK_COPY_LINE_LANES)) = *lane_ptr;
                lane_ptr = (int4*)((half*)lane_ptr + CHUNK_COPY_LINES_PER_WARP*K);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
        } else {
            // each of W{Q,K,V} has TILE_N rows to load
            #pragma unroll
            for (int i=0; i<(TILE_SIZE_N/2)/CHUNK_COPY_LINES_PER_WARP; ++i) {
                *((int4*)&shmem[shmem_idx][0] + (laneId%CHUNK_COPY_LINE_LANES)) = *lane_ptr;
                lane_ptr = (int4*)((half*)lane_ptr + CHUNK_COPY_LINES_PER_WARP*K);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
        }

        __syncthreads();

        // process CHUNK_K 16x16 tiles
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
                        size_t shmem_idx_wq = shmem_idx_wq_off + (warpId%BLOCK_COL_WARPS)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                        const half* wq_tile_ptr = &shmem[shmem_idx_wq][k_step*WMMA_K];
                        
                        wmma::load_matrix_sync(wq_frag[j], wq_tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);

                        if (compute_kv) {
                            size_t shmem_idx_wk = shmem_idx_wk_off + (warpId%BLOCK_COL_WARPS)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                            const half* wk_tile_ptr = &shmem[shmem_idx_wk][k_step*WMMA_K];

                            size_t shmem_idx_wv = shmem_idx_wv_off + (warpId%BLOCK_COL_WARPS)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                            const half* wv_tile_ptr = &shmem[shmem_idx_wv][k_step*WMMA_K];

                            wmma::load_matrix_sync(wk_frag[j], wk_tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                            wmma::load_matrix_sync(wv_frag[j], wv_tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                        }
                    }

                    wmma::mma_sync(q_acc[i][j], x_frag[i], wq_frag[j], q_acc[i][j]);
                    if (compute_kv) {
                        wmma::mma_sync(k_acc[i][j], x_frag[i], wk_frag[j], k_acc[i][j]);
                        wmma::mma_sync(v_acc[i][j], x_frag[i], wv_frag[j], v_acc[i][j]);
                    }
                }
            }

            __syncthreads();
        }
    }


    // acc fragments stored to shmem
    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {

        #pragma unroll
        for (int j=0; j<WARP_COL_TILES; ++j) {
            float* q_tile_ptr = shmem_warp_tile_ptr + shmem_q_out_off + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N;
            wmma::store_matrix_sync(q_tile_ptr, q_acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);

            if (compute_kv) {
                float* k_tile_ptr = shmem_warp_tile_ptr + shmem_k_out_off + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N;
                float* v_tile_ptr = shmem_warp_tile_ptr + shmem_v_out_off + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N;
                wmma::store_matrix_sync(k_tile_ptr, k_acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);
                wmma::store_matrix_sync(v_tile_ptr, v_acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);
            }
        }
    }

    __syncthreads();

    // stream tiles of q_out, k_out, v_out (in shmem) to gmem
    const size_t q_gmem_idx = (tile_i*TILE_SIZE_M+warpId*WMMA_M)*N_Q + tile_j*TILE_SIZE_N;
    half* q_dst_gmem_warp_stream_ptr = &q_out[q_gmem_idx];
    float* shmem_q_stream_ptr = shmem_warp_stream_ptr + shmem_q_out_off;

    #pragma unroll
    for (int i=0; i<WMMA_K; ++i) {
        *((int2*)(q_dst_gmem_warp_stream_ptr+i*N_Q)+laneId) = *((int2*)(shmem_q_stream_ptr+i*SHMEM_STRIDE) + laneId);
    }

    if (compute_kv) {
        const size_t kv_gmem_idx = (tile_i*TILE_SIZE_M+warpId*WMMA_M)*N_KV + tile_j*TILE_SIZE_N;
    
        half* k_dst_gmem_warp_stream_ptr = &k_out[kv_gmem_idx];
        half* v_dst_gmem_warp_stream_ptr = &v_out[kv_gmem_idx];

        float* shmem_k_stream_ptr = shmem_warp_stream_ptr + shmem_k_out_off;
        float* shmem_v_stream_ptr = shmem_warp_stream_ptr + shmem_v_out_off;

        #pragma unroll
        for (int i=0; i<WMMA_K; ++i) {
            *((int2*)(k_dst_gmem_warp_stream_ptr+i*N_KV)+laneId) = *((int2*)(shmem_k_stream_ptr+i*SHMEM_STRIDE) + laneId);
            *((int2*)(v_dst_gmem_warp_stream_ptr+i*N_KV)+laneId) = *((int2*)(shmem_v_stream_ptr+i*SHMEM_STRIDE) + laneId);
        }
    }
}

// helper to compute TILE_SIZE_M x TILE_SIZE_N portion of result for QKV projections
// reduces loads from global mem of input x by x3 factor
template <typename Config>
__device__ __forceinline__ void compute_qkv_tile_quant(
    int tile_i, int tile_j,
    int m_tiles, int n_kv_tiles,
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size,  int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    half (*shmem)[Config::CHUNK_K*WMMA_K+Config::SKEW_HALF],
    const half* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {
    static constexpr auto WARPS_PER_BLOCK = Config::WARPS_PER_BLOCK;
    static constexpr auto CHUNK_K = Config::CHUNK_K;
    static constexpr auto SKEW_HALF = Config::SKEW_HALF;
    static constexpr auto WARP_ROW_TILES = Config::WARP_ROW_TILES;
    static constexpr auto WARP_COL_TILES = Config::WARP_COL_TILES;
    static constexpr auto BLOCK_ROW_WARPS = Config::BLOCK_ROW_WARPS;
    static constexpr auto BLOCK_COL_WARPS = Config::BLOCK_COL_WARPS;
    static constexpr auto SHMEM_STRIDE = Config::SHMEM_STRIDE;
    static constexpr auto SHMEM_OFFSET = Config::SHMEM_OFFSET;
    static constexpr auto CHUNK_COPY_LINE_LANES = Config::CHUNK_COPY_LINE_LANES;
    static constexpr auto CHUNK_COPY_LINES_PER_WARP = Config::CHUNK_COPY_LINES_PER_WARP;
    static constexpr auto TILE_SIZE_M = Config::TILE_SIZE_M;
    static constexpr auto TILE_SIZE_N = Config::TILE_SIZE_N;

    const bool compute_kv = (tile_j < n_kv_tiles);

    const unsigned int warpId = threadIdx.x/32;
    const unsigned int laneId = threadIdx.x%32;
    const int n_warps = blockDim.x/WARP_SIZE;

    const size_t shmem_idx_wq_off = TILE_SIZE_M;
    const size_t shmem_idx_wk_off = 2*TILE_SIZE_M;
    const size_t shmem_idx_wv_off = 3*TILE_SIZE_M;

    const size_t shmem_q_out_off = 0;
    const size_t shmem_k_out_off = TILE_SIZE_M*TILE_SIZE_N;
    const size_t shmem_v_out_off = 2*TILE_SIZE_M*TILE_SIZE_N;

    float* shmem_warp_tile_ptr = (float*)&shmem[0][0]
                               + (warpId/BLOCK_COL_WARPS)*WMMA_M*WARP_ROW_TILES*SHMEM_STRIDE
                               + (warpId%BLOCK_COL_WARPS)*SHMEM_OFFSET;
    float* shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId*WMMA_M*SHMEM_STRIDE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> x_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> wq_frag[WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> wk_frag[WARP_COL_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> wv_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> q_acc[WARP_ROW_TILES][WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> k_acc[WARP_ROW_TILES][WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> v_acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {
        for (int j=0; j<WARP_COL_TILES; ++j) {
            wmma::fill_fragment(q_acc[i][j], 0.0f);
            if (compute_kv) {
                wmma::fill_fragment(k_acc[i][j], 0.0f);
                wmma::fill_fragment(v_acc[i][j], 0.0f);
            }
        }
    }

    int K_TILES = (K+WMMA_K-1)/WMMA_K;

    // iterating over global K dimension, CHUNK_K 16x16 tiles at a time
    for (int tile_k=0; tile_k<K_TILES; tile_k+=CHUNK_K) {
        
        // load X, WQ, WK, WV into shared memory
        // each warp loads 32 rows of X (warps 0-3) or W (warps 4-7)
        
        // all warps load X into shmem
        size_t shmem_idx = (warpId/BLOCK_COL_WARPS)*(WMMA_M*WARP_ROW_TILES)
                         + (warpId%BLOCK_COL_WARPS)*(WMMA_M*WARP_ROW_TILES/BLOCK_COL_WARPS);
        
        int4* lane_ptr = (int4*)(
            x
            + (warpId/BLOCK_COL_WARPS)*(WMMA_M*WARP_ROW_TILES)*K
            + (warpId%BLOCK_COL_WARPS)*(WMMA_M*WARP_ROW_TILES/BLOCK_COL_WARPS)*K
            + (laneId/CHUNK_COPY_LINE_LANES)*K
            + tile_k*WMMA_K
            + (laneId%CHUNK_COPY_LINE_LANES)
        );
        
        shmem_idx += laneId/CHUNK_COPY_LINE_LANES;
        
        #pragma unroll
        for (int i=0; i<((WARP_SIZE/2)/CHUNK_COPY_LINES_PER_WARP)*WARP_ROW_TILES/BLOCK_COL_WARPS; ++i) {
            *((int4*)&shmem[shmem_idx][0] + (laneId%CHUNK_COPY_LINE_LANES)) = *lane_ptr;
            lane_ptr = (int4*)((half*)lane_ptr + CHUNK_COPY_LINES_PER_WARP*K);
            shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        // load WQ into shmem (all warps participate)
        int qblocks_per_row = (CHUNK_K*WMMA_K)/q_qblock_size;
        int n_qblocks = TILE_SIZE_N*qblocks_per_row;
        int qblocks_per_warp = n_qblocks/n_warps;

        int deq_elems_per_lane = sizeof(int4)/sizeof(half); // each lane dequants 8 elems
        int active_lanes = q_qblock_size / deq_elems_per_lane;

        for (int idx=0; idx<qblocks_per_warp; ++idx) {
            int qb_idx = warpId + idx*n_warps;
            int i = qb_idx/qblocks_per_row;
            int j = qb_idx%qblocks_per_row;

            if (laneId < active_lanes) {
                const uint8_t* wq_qblock = wq + i*(K/q_qblock_size)*q_qtype_size + (tile_k*WMMA_K/q_qblock_size+j)*q_qtype_size;
                half* shmem_out = &shmem[shmem_idx_wq_off+i][j*q_qblock_size];
                dequant_block(q_qtype_int, laneId, shmem_out, wq_qblock);
            }
        }

        __syncthreads();

        // process CHUNK_K 16x16 tiles, X and WQ
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
                        size_t shmem_idx_wq = shmem_idx_wq_off + (warpId%BLOCK_COL_WARPS)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                        const half* wq_tile_ptr = &shmem[shmem_idx_wq][k_step*WMMA_K];
                        
                        wmma::load_matrix_sync(wq_frag[j], wq_tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                    }

                    wmma::mma_sync(q_acc[i][j], x_frag[i], wq_frag[j], q_acc[i][j]);
                }
            }

            __syncthreads();
        }

        if (compute_kv) {

            // load WK into shmem (all warps participate)
            qblocks_per_row = (CHUNK_K*WMMA_K)/k_qblock_size;
            n_qblocks = TILE_SIZE_N*qblocks_per_row;
            qblocks_per_warp = n_qblocks/n_warps;

            deq_elems_per_lane = sizeof(int4)/sizeof(half); // each lane dequants 8 elems
            active_lanes = k_qblock_size / deq_elems_per_lane;

            for (int idx=0; idx<qblocks_per_warp; ++idx) {
                int qb_idx = warpId + idx*n_warps;
                int i = qb_idx/qblocks_per_row;
                int j = qb_idx%qblocks_per_row;

                if (laneId < active_lanes) {
                    const uint8_t* wk_qblock = wk + i*(K/k_qblock_size)*k_qtype_size + (tile_k*WMMA_K/k_qblock_size+j)*k_qtype_size;
                    half* shmem_out = &shmem[shmem_idx_wk_off+i][j*k_qblock_size];
                    dequant_block(k_qtype_int, laneId, shmem_out, wk_qblock);
                }
            }

            __syncthreads();

            // process CHUNK_K 16x16 tiles, X and WK
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
                            size_t shmem_idx_wk = shmem_idx_wk_off + (warpId%BLOCK_COL_WARPS)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                            const half* wk_tile_ptr = &shmem[shmem_idx_wk][k_step*WMMA_K];
                            
                            wmma::load_matrix_sync(wk_frag[j], wk_tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                        }

                        wmma::mma_sync(k_acc[i][j], x_frag[i], wk_frag[j], k_acc[i][j]);
                    }
                }

                __syncthreads();
            }
            
            // load WV into shmem (all warps participate)
            qblocks_per_row = (CHUNK_K*WMMA_K)/v_qblock_size;
            n_qblocks = TILE_SIZE_N*qblocks_per_row;
            qblocks_per_warp = n_qblocks/n_warps;

            deq_elems_per_lane = sizeof(int4)/sizeof(half); // each lane dequants 8 elems
            active_lanes = v_qblock_size / deq_elems_per_lane;

            for (int idx=0; idx<qblocks_per_warp; ++idx) {
                int qb_idx = warpId + idx*n_warps;
                int i = qb_idx/qblocks_per_row;
                int j = qb_idx%qblocks_per_row;

                if (laneId < active_lanes) {
                    const uint8_t* wv_qblock = wv + i*(K/v_qblock_size)*v_qtype_size + (tile_k*WMMA_K/v_qblock_size+j)*v_qtype_size;
                    half* shmem_out = &shmem[shmem_idx_wv_off+i][j*v_qblock_size];
                    dequant_block(v_qtype_int, laneId, shmem_out, wv_qblock);
                }
            }

            __syncthreads();

            // process CHUNK_K 16x16 tiles, X and WV
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
                            size_t shmem_idx_wv = shmem_idx_wv_off + (warpId%BLOCK_COL_WARPS)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                            const half* wv_tile_ptr = &shmem[shmem_idx_wv][k_step*WMMA_K];
                            
                            wmma::load_matrix_sync(wv_frag[j], wv_tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                        }

                        wmma::mma_sync(v_acc[i][j], x_frag[i], wv_frag[j], v_acc[i][j]);
                    }
                }

                __syncthreads();
            }
        }
    }

    // acc fragments stored to shmem
    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {

        #pragma unroll
        for (int j=0; j<WARP_COL_TILES; ++j) {
            float* q_tile_ptr = shmem_warp_tile_ptr + shmem_q_out_off + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N;
            wmma::store_matrix_sync(q_tile_ptr, q_acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);

            if (compute_kv) {
                float* k_tile_ptr = shmem_warp_tile_ptr + shmem_k_out_off + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N;
                float* v_tile_ptr = shmem_warp_tile_ptr + shmem_v_out_off + i*WMMA_M*SHMEM_STRIDE + j*WMMA_N;
                wmma::store_matrix_sync(k_tile_ptr, k_acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);
                wmma::store_matrix_sync(v_tile_ptr, v_acc[i][j], SHMEM_STRIDE, wmma::mem_row_major);
            }
        }
    }

    __syncthreads();

    // stream tiles of q_out, k_out, v_out (in shmem) to gmem
    const size_t q_gmem_idx = (tile_i*TILE_SIZE_M+warpId*WMMA_M)*N_Q + tile_j*TILE_SIZE_N;
    half* q_dst_gmem_warp_stream_ptr = &q_out[q_gmem_idx];
    float* shmem_q_stream_ptr = shmem_warp_stream_ptr + shmem_q_out_off;

    #pragma unroll
    for (int i=0; i<WMMA_K; ++i) {
        *((int2*)(q_dst_gmem_warp_stream_ptr+i*N_Q)+laneId) = *((int2*)(shmem_q_stream_ptr+i*SHMEM_STRIDE) + laneId);
    }

    if (compute_kv) {
        const size_t kv_gmem_idx = (tile_i*TILE_SIZE_M+warpId*WMMA_M)*N_KV + tile_j*TILE_SIZE_N;
    
        half* k_dst_gmem_warp_stream_ptr = &k_out[kv_gmem_idx];
        half* v_dst_gmem_warp_stream_ptr = &v_out[kv_gmem_idx];

        float* shmem_k_stream_ptr = shmem_warp_stream_ptr + shmem_k_out_off;
        float* shmem_v_stream_ptr = shmem_warp_stream_ptr + shmem_v_out_off;

        #pragma unroll
        for (int i=0; i<WMMA_K; ++i) {
            *((int2*)(k_dst_gmem_warp_stream_ptr+i*N_KV)+laneId) = *((int2*)(shmem_k_stream_ptr+i*SHMEM_STRIDE) + laneId);
            *((int2*)(v_dst_gmem_warp_stream_ptr+i*N_KV)+laneId) = *((int2*)(shmem_v_stream_ptr+i*SHMEM_STRIDE) + laneId);
        }
    }
}