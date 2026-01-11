#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

template <typename Config>
__global__ void qkv_f16_cuda_impl(
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    const half* __restrict__ x,
    const half* __restrict__ wq,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {
    static constexpr auto CHUNK_K = Config::CHUNK_K;
    static constexpr auto SKEW_HALF = Config::SKEW_HALF;
    static constexpr auto TILE_SIZE_M = Config::TILE_SIZE_M;
    static constexpr auto TILE_SIZE_N = Config::TILE_SIZE_N;

    extern __shared__ half shmem_raw[];
    half (*shmem)[CHUNK_K*WMMA_K+SKEW_HALF] = reinterpret_cast<half(*)[CHUNK_K*WMMA_K+SKEW_HALF]>(shmem_raw);

    int m_tiles = (M+TILE_SIZE_M-1)/TILE_SIZE_M;
    int n_tiles = (N_Q+TILE_SIZE_N-1)/TILE_SIZE_N; // assume: N_Q (q_dim) >= N_KV (kv_dim)
    int n_kv_tiles = (N_KV+TILE_SIZE_N-1)/TILE_SIZE_N;

    // split up work, assign across all SMs (each block corresponds to one SM)
    for (int block_pos=blockIdx.x; block_pos<m_tiles*n_tiles; block_pos+=gridDim.x) {

        int tile_i = block_pos/n_tiles;
        int tile_j = block_pos%n_tiles;

        const half* x_slice = x + tile_i*TILE_SIZE_M*K;
        const half* wq_slice = wq + tile_j*TILE_SIZE_N*K;
        const half* wk_slice = (tile_j < n_kv_tiles) ? wk + tile_j*TILE_SIZE_N*K : nullptr;
        const half* wv_slice = (tile_j < n_kv_tiles) ? wv + tile_j*TILE_SIZE_N*K : nullptr;
        half* q_out_tile = q_out + tile_i*TILE_SIZE_M*N_Q + tile_j*TILE_SIZE_N;
        half* k_out_tile = (tile_j < n_kv_tiles) ? k_out + tile_i*TILE_SIZE_M*N_KV + tile_j*TILE_SIZE_N : nullptr;
        half* v_out_tile = (tile_j < n_kv_tiles) ? v_out + tile_i*TILE_SIZE_M*N_KV + tile_j*TILE_SIZE_N : nullptr;
        
        compute_qkv_tile_f16<Config>(
            tile_i, tile_j,
            m_tiles, n_kv_tiles,
            M, K, N_Q, N_KV,
            shmem,
            x_slice, 
            wq_slice, wk_slice, wv_slice,
            q_out_tile, k_out_tile, v_out_tile
        );
    }
}

template <typename Config>
__global__ void qkv_quant_cuda_impl(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size,  int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    const half* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {
    static constexpr auto CHUNK_K = Config::CHUNK_K;
    static constexpr auto SKEW_HALF = Config::SKEW_HALF;
    static constexpr auto TILE_SIZE_M = Config::TILE_SIZE_M;
    static constexpr auto TILE_SIZE_N = Config::TILE_SIZE_N;

    extern __shared__ half shmem_raw[];
    half (*shmem)[CHUNK_K*WMMA_K+SKEW_HALF] = reinterpret_cast<half(*)[CHUNK_K*WMMA_K+SKEW_HALF]>(shmem_raw);

    int m_tiles = (M+TILE_SIZE_M-1)/TILE_SIZE_M;
    int n_tiles = (N_Q+TILE_SIZE_N-1)/TILE_SIZE_N;
    int n_kv_tiles = (N_KV+TILE_SIZE_N-1)/TILE_SIZE_N;

    // split up work, assign across all SMs (each block corresponds to one SM)
    for (int block_pos=blockIdx.x; block_pos<m_tiles*n_tiles; block_pos+=gridDim.x) {

        int tile_i = block_pos/n_tiles;
        int tile_j = block_pos%n_tiles;

        const half* x_slice = x + tile_i*TILE_SIZE_M*K;
        const uint8_t* wq_slice = wq + tile_j*TILE_SIZE_N*(K/q_qblock_size)*q_qtype_size;
        const uint8_t* wk_slice = (tile_j < n_kv_tiles) ? wk + tile_j*TILE_SIZE_N*(K/k_qblock_size)*k_qtype_size : nullptr;
        const uint8_t* wv_slice = (tile_j < n_kv_tiles) ? wv + tile_j*TILE_SIZE_N*(K/v_qblock_size)*v_qtype_size : nullptr;
        half* q_out_tile = q_out + tile_i*TILE_SIZE_M*N_Q + tile_j*TILE_SIZE_N;
        half* k_out_tile = (tile_j < n_kv_tiles) ? k_out + tile_i*TILE_SIZE_M*N_KV + tile_j*TILE_SIZE_N : nullptr;
        half* v_out_tile = (tile_j < n_kv_tiles) ? v_out + tile_i*TILE_SIZE_M*N_KV + tile_j*TILE_SIZE_N : nullptr;
        
        compute_qkv_tile_quant<Config>(
            tile_i, tile_j,
            m_tiles, n_tiles,
            q_qtype_int, k_qtype_int, v_qtype_int,
            q_qblock_size,  k_qblock_size, v_qblock_size,
            q_qtype_size, k_qtype_size, v_qtype_size,
            M, K, N_Q, N_KV,
            shmem,
            x_slice,
            wq_slice, wk_slice, wv_slice,
            q_out_tile, k_out_tile, v_out_tile
        );
    }
}