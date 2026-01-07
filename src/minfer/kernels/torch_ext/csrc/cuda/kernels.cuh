#pragma once

#include <cstddef>

constexpr size_t const_max(size_t a, size_t b) {
    return a > b ? a : b;
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_SIZE = 32;

// for matmul
template<int TILE_M, int TILE_N, int CHUNK_K_VAL, int WARP_GRID_ROWS, int WARP_GRID_COLS, bool W_IS_COL_MAJOR>
struct MatmulConfig {

    static constexpr int WARPS_PER_BLOCK = WARP_GRID_ROWS * WARP_GRID_COLS;
    static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

    static constexpr int TILE_SIZE_M = TILE_M;
    static constexpr int TILE_SIZE_N = TILE_N;
    static constexpr int CHUNK_K = CHUNK_K_VAL;
    static constexpr bool W_COL_MAJOR = W_IS_COL_MAJOR; // controls how WMMA interprets the W fragment (in all cases W is row-major)
    
    static constexpr int CHUNK_LINE_BYTES = CHUNK_K_VAL*WMMA_K*2;  // sizeof(half) = 2
    static constexpr int WARP_COPY_BYTES = WARP_SIZE*16;  // sizeof(int4) = 16
    static constexpr int CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES/CHUNK_LINE_BYTES;
    static constexpr int CHUNK_COPY_LINE_LANES = WARP_SIZE/CHUNK_COPY_LINES_PER_WARP;
    
    // tile layout across the entire TILE_SIZE_M x TILE_SIZE_N block
    static constexpr int BLOCK_ROW_TILES = TILE_M/WMMA_M;
    static constexpr int BLOCK_COL_TILES = TILE_N/WMMA_N;

    // warp layout across the TILE_SIZE_M x TILE_SIZE_N block
    static constexpr int BLOCK_ROW_WARPS = WARP_GRID_ROWS;
    static constexpr int BLOCK_COL_WARPS = WARP_GRID_COLS;
    
    // tile layout across single warp
    static constexpr int WARP_ROW_TILES = BLOCK_ROW_TILES/BLOCK_ROW_WARPS;
    static constexpr int WARP_COL_TILES = BLOCK_COL_TILES/BLOCK_COL_WARPS;
    
    static constexpr int SHMEM_STRIDE = WMMA_N*BLOCK_COL_TILES;  // how many elements per block row (NOTE: not strictly necessary since this is just TILE_N, including just for clarity)
    static constexpr int SHMEM_OFFSET = WMMA_N*WARP_COL_TILES;   // how many elements per warp row
    
    // load matrix sync requires 256-bit (32-byte) alignment
    // hence the minimum possible padding is 16 half elements, i.e. 32 bytes (256+32=288%32=0)
    // relevant when WMMA interprets W as column major (X@W.T), otherwise massive bank conflicts
    static constexpr int SKEW_HALF = W_IS_COL_MAJOR ? 16 : 0;
    
    static constexpr size_t SHMEM_SZ = const_max(
        (TILE_M+TILE_N)*(CHUNK_K_VAL*WMMA_K+SKEW_HALF)*2, // X+W tiles (written to shmem as half)
        TILE_M*TILE_N*4                          // out tiles (written to shmem as float)
    );
};

//pre-defined configs
using Config_Matmul_F16_T = MatmulConfig<128, 128, 8, 4, 2, true>;   // X @ W.T (more frequently the case)
using Config_Matmul_F16 = MatmulConfig<128, 128, 8, 4, 2, false>;    // X @ W
using Config_Matmul_Quant_T = MatmulConfig<64, 64, 16, 4, 2, true>;  // X @ W.T, W quantized (more frequently the case)
using Config_Matmul_Quant = MatmulConfig<64, 64, 16, 4, 2, false>;   // X @ W, W quantized

// for QKV
template<int TILE_M, int TILE_N, int CHUNK_K_VAL, int WARP_GRID_ROWS, int WARP_GRID_COLS>
struct QKVConfig {

    static constexpr int WARPS_PER_BLOCK = WARP_GRID_ROWS * WARP_GRID_COLS;
    static constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

    static constexpr int TILE_SIZE_M = TILE_M;
    static constexpr int TILE_SIZE_N = TILE_N;
    static constexpr int CHUNK_K = CHUNK_K_VAL;
        
    static constexpr int CHUNK_LINE_BYTES = CHUNK_K_VAL*WMMA_K*2;  // sizeof(half) = 2
    static constexpr int WARP_COPY_BYTES = WARP_SIZE*16;  // sizeof(int4) = 16
    static constexpr int CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES/CHUNK_LINE_BYTES;
    static constexpr int CHUNK_COPY_LINE_LANES = WARP_SIZE/CHUNK_COPY_LINES_PER_WARP;
    
    // tile layout across the entire TILE_SIZE_M x TILE_SIZE_N block
    static constexpr int BLOCK_ROW_TILES = TILE_M/WMMA_M;
    static constexpr int BLOCK_COL_TILES = TILE_N/WMMA_N;

    // warp layout across the TILE_SIZE_M x TILE_SIZE_N block
    static constexpr int BLOCK_ROW_WARPS = WARP_GRID_ROWS;
    static constexpr int BLOCK_COL_WARPS = WARP_GRID_COLS;
    
    // tile layout across single warp
    static constexpr int WARP_ROW_TILES = BLOCK_ROW_TILES/BLOCK_ROW_WARPS;
    static constexpr int WARP_COL_TILES = BLOCK_COL_TILES/BLOCK_COL_WARPS;
    
    static constexpr int SHMEM_STRIDE = WMMA_N*BLOCK_COL_TILES;  // how many elements per block row (NOTE: not strictly necessary since this is just TILE_N, including just for clarity)
    static constexpr int SHMEM_OFFSET = WMMA_N*WARP_COL_TILES;   // how many elements per warp row
    
    // load matrix sync requires 256-bit (32-byte) alignment
    // hence the minimum possible padding is 16 half elements, i.e. 32 bytes (256+32=288%32=0)
    // relevant when WMMA interprets W as column major (X@W.T), otherwise massive bank conflicts
    static constexpr int SKEW_HALF = 16; // W{Q,K,V} assumed to always be col-major
    
    static constexpr size_t SHMEM_SZ = const_max(
        ((TILE_M+3*TILE_N)*(CHUNK_K_VAL*WMMA_K+SKEW_HALF))*2, // X+WQ+WK+WV tiles (written to shmem as half)
        TILE_M*TILE_N*4                              // out tiles (written to shmem as float)
    );
};

using Config_QKV_F16_T = QKVConfig<64, 64, 8, 4, 2>;  // X @ W{Q,K,V}.T
using Config_QKV_Quant_T = QKVConfig<64, 32, 16, 4, 2>;  // X @ W{Q,K,V}.T, W{Q,K,V} quantized