#pragma once

#include <cuda_fp16.h>
#include <vector_types.h>
#include <cassert>

#include <cstddef>

namespace minfer::impl {

constexpr __host__ __device__ unsigned int int_log2(unsigned int n) {
    unsigned int result = 0;
    while (n >>= 1) result++;
    return result;
}

// NOTE: recall that A^(A^B) = B
template<
    unsigned int BITMASK,
    unsigned int SHIFT
>
__device__ __forceinline__ constexpr uint32_t swizzle(uint32_t x) { 
    return x ^ ((x & BITMASK) >> SHIFT);
}

template<
    unsigned int STRIDE, 
    unsigned int BITMASK, 
    unsigned int SHIFT,
    unsigned int INCR
>
__device__ __forceinline__ constexpr uint32_t xor_pattern(unsigned int i) {
    return swizzle<BITMASK, SHIFT>((i+INCR)*STRIDE) ^ swizzle<BITMASK, SHIFT>(i*STRIDE);
}

// ldmatrix for shmem -> regs
template <
    bool LOAD_TRANS,
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int MMAS_ROW,
    unsigned int MMAS_COL,
    unsigned int COLS_BLOCK
>
__device__ __forceinline__ void ldmatrix(
    unsigned int lane_idx,
    const half* src,
    uint32_t (&reg)[MMAS_ROW][MMAS_COL][(ROWS_MMA/8)*(COLS_MMA/8)]
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4));
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    // GRID_ROWS x GRID_COLS grid of 8x8 tiles
    constexpr unsigned int GRID_ROWS = ROWS_MMA / 8;
    constexpr unsigned int GRID_COLS = COLS_MMA / 8;
    constexpr unsigned int NUM_TILES = GRID_ROWS * GRID_COLS;

    static_assert(NUM_TILES == 1 || NUM_TILES == 2 || NUM_TILES == 4);

    constexpr unsigned int EFF_ROW_SPAN = (GRID_ROWS == 1 && MMAS_ROW >= 2) ? 2 : 1;
    constexpr unsigned int EFF_COL_SPAN = (GRID_COLS == 1 && MMAS_COL >= 2) ? 2 : 1;
    constexpr unsigned int EFF_GRID_ROWS = GRID_ROWS * EFF_ROW_SPAN;
    constexpr unsigned int EFF_GRID_COLS = GRID_COLS * EFF_COL_SPAN;
    constexpr unsigned int EFF_NUM_TILES = (NUM_TILES*EFF_ROW_SPAN*EFF_COL_SPAN>4) ? 4 : NUM_TILES*EFF_ROW_SPAN*EFF_COL_SPAN;

    constexpr unsigned int ROW_INCR = EFF_ROW_SPAN;
    constexpr unsigned int COL_INCR = EFF_COL_SPAN;

    // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
    unsigned int tile_id = lane_idx >> 3;
    unsigned int row_in_tile = lane_idx & 7;

    unsigned int tile_row = tile_id%EFF_GRID_ROWS;
    unsigned int tile_col = tile_id/EFF_GRID_ROWS;

    unsigned int row = tile_row*8+row_in_tile;
    unsigned int col = tile_col*8;

    unsigned int idx_thr = row*COLS_BLOCK+col;

    // reduce instrs by XOR src pointer with const
    const uint32_t shared_base = __cvta_generic_to_shared(src);
    uint32_t base_addr = shared_base + idx_thr*sizeof(half);
    uint32_t src_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_COL = COLS_MMA * sizeof(half);
    constexpr uint32_t STRIDE_ROW = ROWS_MMA * COLS_BLOCK * sizeof(half);

#pragma unroll
    for (int mma_row = 0; mma_row < MMAS_ROW; mma_row += ROW_INCR) {
        
        const uint32_t row_addr = src_addr;

#pragma unroll
        for (int mma_col = 0; mma_col < MMAS_COL; mma_col += COL_INCR) {

            if constexpr (EFF_NUM_TILES == 1) {
                if constexpr (LOAD_TRANS) {
                    asm(
                        "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 "
                        "{%0}, [%1];\n"
                        : "=r"(reg[mma_row][mma_col][0])
                        : "r"(src_addr)
                    );
                } else {
                    asm(
                        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
                        "{%0}, [%1];\n"
                        : "=r"(reg[mma_row][mma_col][0])
                        : "r"(src_addr)
                    );
                }
            } else if constexpr (EFF_NUM_TILES == 2) {
                if constexpr (LOAD_TRANS) {
                    if constexpr (EFF_ROW_SPAN == 2 && EFF_COL_SPAN == 1) {
                        // spans 2 MMA rows
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row+1][mma_col][0])
                            : "r"(src_addr)
                        );
                    } else if constexpr (EFF_ROW_SPAN == 1 && EFF_COL_SPAN == 2) {
                        // spans 2 MMA cols
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col+1][0])
                            : "r"(src_addr)
                        );
                    } else if constexpr (GRID_ROWS == 2 && GRID_COLS == 1) {
                        // 16x8 w/in 1 MMA tile
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1])
                            : "r"(src_addr)
                        );
                    } else if constexpr (GRID_ROWS == 1 && GRID_COLS == 2) {
                        // 8x16 w/in 1 MMA tile
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1])
                            : "r"(src_addr)
                        );
                    }
                } else {
                    if constexpr (EFF_ROW_SPAN == 2 && EFF_COL_SPAN == 1) {
                        // in terms of args, identical to above
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row+1][mma_col][0])
                            : "r"(src_addr)
                        );
                    } else if constexpr (EFF_ROW_SPAN == 1 && EFF_COL_SPAN == 2) {
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col+1][0])
                            : "r"(src_addr)
                        );
                    } else if constexpr (GRID_ROWS == 2 && GRID_COLS == 1) {
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1])
                            : "r"(src_addr)
                        );
                    } else if constexpr (GRID_ROWS == 1 && GRID_COLS == 2) {
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                            "{%0, %1}, [%2];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1])
                            : "r"(src_addr)
                        );
                    }
                }
            } else if constexpr (EFF_NUM_TILES == 4) {
                if constexpr (LOAD_TRANS) {
                    if constexpr (EFF_ROW_SPAN == 2 && EFF_COL_SPAN == 2) {
                        // 8x8 load spanning 2x2 MMA tiles
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row+1][mma_col][0]),
                              "=r"(reg[mma_row][mma_col+1][0]), "=r"(reg[mma_row+1][mma_col+1][0])
                            : "r"(src_addr)
                        );
                    } else if constexpr (EFF_ROW_SPAN == 2 && EFF_COL_SPAN == 1 && GRID_COLS == 2) {
                        // 8x16 load spanning 2 MMA rows
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row+1][mma_col][0]),
                              "=r"(reg[mma_row][mma_col][1]), "=r"(reg[mma_row+1][mma_col][1])
                            : "r"(src_addr)
                        );
                    } else if constexpr (EFF_ROW_SPAN == 1 && EFF_COL_SPAN == 2 && GRID_ROWS == 2) {
                        // 16x8 load spanning 2 MMA cols
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1]),
                              "=r"(reg[mma_row][mma_col+1][0]), "=r"(reg[mma_row][mma_col+1][1])
                            : "r"(src_addr)
                        );
                    } else if constexpr (GRID_ROWS == 2 && GRID_COLS == 2) {
                        // 16x16 w/in 1 MMA tile
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1]),
                              "=r"(reg[mma_row][mma_col][2]), "=r"(reg[mma_row][mma_col][3])
                            : "r"(src_addr)
                        );
                    }
                } else {
                    if constexpr (EFF_ROW_SPAN == 2 && EFF_COL_SPAN == 2) {
                        // in terms of args, identical to above
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row+1][mma_col][0]),
                              "=r"(reg[mma_row][mma_col+1][0]), "=r"(reg[mma_row+1][mma_col+1][0])
                            : "r"(src_addr)
                        );
                    } else if constexpr (EFF_ROW_SPAN == 2 && EFF_COL_SPAN == 1 && GRID_COLS == 2) {
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row+1][mma_col][0]),
                              "=r"(reg[mma_row][mma_col][1]), "=r"(reg[mma_row+1][mma_col][1])
                            : "r"(src_addr)
                        );
                    } else if constexpr (EFF_ROW_SPAN == 1 && EFF_COL_SPAN == 2 && GRID_ROWS == 2) {
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1]),
                              "=r"(reg[mma_row][mma_col+1][0]), "=r"(reg[mma_row][mma_col+1][1])
                            : "r"(src_addr)
                        );
                    } else if constexpr (GRID_ROWS == 2 && GRID_COLS == 2) {
                        asm(
                            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                            "{%0, %1, %2, %3}, [%4];\n"
                            : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1]),
                              "=r"(reg[mma_row][mma_col][2]), "=r"(reg[mma_row][mma_col][3])
                            : "r"(src_addr)
                        );
                    }
                }
            }

            src_addr ^= xor_pattern<STRIDE_COL, BITMASK, SHIFT, COL_INCR>(mma_col);
        }

        src_addr = row_addr ^ xor_pattern<STRIDE_ROW, BITMASK, SHIFT, ROW_INCR>(mma_row);
    }
}

// configurable K mma_sync for tuning
template <
    unsigned int DIM_MM,
    unsigned int DIM_MK,
    unsigned int DIM_MN,
    unsigned int MMAS_M,
    unsigned int MMAS_K,
    unsigned int MMAS_N
>
__device__ __forceinline__ void mma_sync(
    uint32_t (&x_reg)[MMAS_M][MMAS_K][(DIM_MM/8)*(DIM_MK/8)],
    uint32_t (&w_reg)[MMAS_K][MMAS_N][(DIM_MK/8)*(DIM_MN/8)],
    uint32_t (&acc_reg)[MMAS_M][MMAS_N][(DIM_MM/8)*(DIM_MN/8)]
) {

    static_assert(DIM_MM == 16 && DIM_MN == 8); // based on the instrs we have available

#pragma unroll
    for (int mma_k = 0; mma_k < MMAS_K; ++mma_k) {

#pragma unroll
        for (int mma_m = 0; mma_m < MMAS_M; ++mma_m) {

#pragma unroll
            for (int mma_n = 0; mma_n < MMAS_N; ++mma_n) {

                if constexpr (DIM_MK == 8) {

                    asm(
                        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                        "{%0, %1}, "
                        "{%2, %3}, "
                        "{%4}, "
                        "{%5, %6};\n"
                        : "=r"(acc_reg[mma_m][mma_n][0]), "=r"(acc_reg[mma_m][mma_n][1])
                        : "r"(x_reg[mma_m][mma_k][0]), "r"(x_reg[mma_m][mma_k][1]),
                          "r"(w_reg[mma_k][mma_n][0]),
                          "r"(acc_reg[mma_m][mma_n][0]), "r"(acc_reg[mma_m][mma_n][1])
                    );

                } else if constexpr (DIM_MK == 16) {
                    asm(
                        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                        "{%0, %1}, "
                        "{%2, %3, %4, %5}, "
                        "{%6, %7}, "
                        "{%8, %9};\n"
                        : "=r"(acc_reg[mma_m][mma_n][0]), "=r"(acc_reg[mma_m][mma_n][1])
                        : "r"(x_reg[mma_m][mma_k][0]), "r"(x_reg[mma_m][mma_k][1]), "r"(x_reg[mma_m][mma_k][2]), "r"(x_reg[mma_m][mma_k][3]),
                          "r"(w_reg[mma_k][mma_n][0]), "r"(w_reg[mma_k][mma_n][1]),
                          "r"(acc_reg[mma_m][mma_n][0]), "r"(acc_reg[mma_m][mma_n][1])
                    );
                }
            }
        }
    }
}

// sync/async toShmem
template <
    unsigned int ROWS_MMA,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS,
    unsigned int ELEMS_THR
>
__device__ __forceinline__ void gmemToRegSync(
    size_t stride_src,
    const half* __restrict__ src,
    int4 (&dst)[ELEMS_THR]
) {

    // can issue up to 128-bit load/store instrs (8 half elements)
    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);
    // assert(stride_src % 8 == 0);
    
    size_t eff_stride_src = stride_src >> 3;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;
    static_assert(NUM_ITERS >= 1u);
    static_assert(NUM_ITERS == ELEMS_THR);

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;

#pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        dst[i] = reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
        row_thr += INCR_ROW;
    }
}

template <
    unsigned int ROWS_MMA,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS,
    unsigned int ELEMS_THR
>
__device__ __forceinline__ void regToShmemSync(
    half* __restrict__ dst,
    const int4 (&src)[ELEMS_THR]
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // since ldmatrix reads in with this base
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);

    // can issue up to 128-bit load/store instrs (8 half elements)
    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);

    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;

    static_assert(NUM_ITERS >= 1u);
    static_assert(ELEMS_THR == NUM_ITERS);

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;

    // reduce instrs by XOR dst pointer with const
    uint32_t shared_base = __cvta_generic_to_shared(dst);
    unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t base_addr = shared_base + idx_dst*sizeof(int4);
    uint32_t dst_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_DST = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

#pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        int4* dst_ptr = reinterpret_cast<int4*>(__cvta_shared_to_generic(dst_addr));
        *dst_ptr = src[i];
        dst_addr ^= xor_pattern<STRIDE_DST, BITMASK, SHIFT, 1u>(i); // NOTE: this imposes a constraint on i, i believe it is i <= 31
    }
}

// for prefetching before loop in gemm_sync
template <
    unsigned int ROWS_MMA,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void toShmemSync(
    size_t stride_src,
    const half* __restrict__ src,
    half* __restrict__ dst
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u<<BITS)-1) << (BASE+SHIFT);

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    // can issue up to 128-bit load/store instrs (8 half elements)
    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);
    // assert(stride_src % 8 == 0);
    
    size_t eff_stride_src = stride_src >> 3;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;
    static_assert(NUM_ITERS >= 1u);
    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;
    
    // reduce instrs by XOR dst pointer with const
    uint32_t shared_base = __cvta_generic_to_shared(dst);
    unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t base_addr = shared_base + idx_dst*sizeof(int4);
    uint32_t dst_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_DST = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

#pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        int4* dst_ptr = reinterpret_cast<int4*>(__cvta_shared_to_generic(dst_addr));
        *dst_ptr = reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
        dst_addr ^= xor_pattern<STRIDE_DST, BITMASK, SHIFT, 1u>(i);
        row_thr += INCR_ROW;
    }
}

// (pipelining) async wrappers
template<unsigned int N>
__device__ __forceinline__ void cp_async(
    uint32_t dst_shared,
    const void* src_global
) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-non-bulk-copy
    static_assert(N == 4 || N == 8 || N == 16);

    asm(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(dst_shared), "l"(src_global), "n"(N)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<unsigned int N>
__device__ __forceinline__ void cp_async_wait() {
    static_assert(N <= 7);
    
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

template <
    unsigned int ROWS_MMA,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void toShmemAsync(
    size_t stride_src,
    const half* __restrict__ src,
    half* __restrict__ dst
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u<<BITS)-1) << (BASE+SHIFT);

    // can issue up to 128-bit load/store instrs (8 half elements)
    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);
    // assert(stride_src % 8 == 0);
    
    size_t eff_stride_src = stride_src >> 3;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);

    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;
    static_assert(NUM_ITERS >= 1u);
    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;
    
    // reduce instrs by XOR dst pointer with const
    uint32_t shared_base = __cvta_generic_to_shared(dst);
    unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t base_addr = shared_base + idx_dst*sizeof(int4);
    uint32_t dst_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_DST = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

#pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        const void* src_global = &reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
        cp_async<16>(dst_addr, src_global);
        dst_addr ^= xor_pattern<STRIDE_DST, BITMASK, SHIFT, 1u>(i);
        row_thr += INCR_ROW;
    }
}

// prologue for GEMM (optimize by loading C into registers first if applicable (beta != 0))
template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int MMAS_ROW,
    unsigned int MMAS_COL,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void prologue(
    unsigned int lane_idx,
    size_t stride_src,
    float alpha,
    float beta,
    const half* __restrict__ bias_src,
    half* __restrict__ shmem_bias_base,
    half* __restrict__ shmem_bias_warp,
    uint32_t (&regs_acc)[MMAS_ROW][MMAS_COL][(ROWS_MMA/8)*(COLS_MMA/8)]
) {
    static_assert(ROWS_MMA == 16 && COLS_MMA == 8); // for now DIM_MM = 16 and DIM_MN = 8 are fixed

    if (beta == 0.0f) {
        memset(regs_acc, 0, sizeof(regs_acc));
        return;
    }

    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);

    size_t eff_stride_src = stride_src >> 3;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    // swizzle
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment for toGmem
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;
    static_assert(NUM_ITERS >= 1u);
    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;

    // reduce instrs by XOR dst pointer with const
    uint32_t shared_base = __cvta_generic_to_shared(shmem_bias_base);
    unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t base_addr = shared_base + idx_dst*sizeof(int4);
    uint32_t dst_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_DST = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

    // gmem -> C shmem
#pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        int4* dst_ptr = reinterpret_cast<int4*>(__cvta_shared_to_generic(dst_addr));
        *dst_ptr = reinterpret_cast<const int4*>(bias_src)[row_thr*eff_stride_src+col_thr];
        dst_addr ^= xor_pattern<STRIDE_DST, BITMASK, SHIFT, 1u>(i);
        row_thr += INCR_ROW;
    }

    __syncthreads();

    // C shmem -> regs
    ldmatrix<false, ROWS_MMA, COLS_MMA, MMAS_ROW, MMAS_COL, COLS_BLOCK>(lane_idx, shmem_bias_warp, regs_acc);

    if (beta != alpha) { // alpha = 0 caught on host-side

        half scale_h = __float2half(beta/alpha);
        half2 scale_h2 = __half2half2(scale_h);

#pragma unroll
        for (int mma_row = 0; mma_row < MMAS_ROW; ++mma_row) {

#pragma unroll
            for (int mma_col = 0; mma_col < MMAS_COL; ++mma_col) {

                half2 lo = *reinterpret_cast<half2*>(&regs_acc[mma_row][mma_col][0]);
                half2 hi = *reinterpret_cast<half2*>(&regs_acc[mma_row][mma_col][1]);

                lo = __hmul2(lo, scale_h2);
                hi = __hmul2(hi, scale_h2);

                regs_acc[mma_row][mma_col][0] = *reinterpret_cast<uint32_t*>(&lo);
                regs_acc[mma_row][mma_col][1] = *reinterpret_cast<uint32_t*>(&hi);
            }
        }
    }
}

template <
    unsigned int ROWS_MMA,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void toGmem(
    size_t stride_dst,
    const half* src,
    half* dst
) {
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4));
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    static_assert(COLS_BLOCK % 8 == 0);
    // assert(stride_dst % 8 == 0);

    size_t eff_stride_dst = stride_dst >> 3;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);

    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;
    static_assert(NUM_ITERS >= 1u);
    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;

    // reduce instrs by XOR src pointer with const
    uint32_t shared_base = __cvta_generic_to_shared(src);
    uint32_t idx_src = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t base_addr = shared_base + idx_src*sizeof(int4);
    uint32_t src_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_SRC = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

#pragma unroll
    for (int i = 0; i < NUM_ITERS; ++i) {
        int4* src_ptr = reinterpret_cast<int4*>(__cvta_shared_to_generic(src_addr));
        reinterpret_cast<int4*>(dst)[row_thr*eff_stride_dst+col_thr] = *src_ptr;
        src_addr ^= xor_pattern<STRIDE_SRC, BITMASK, SHIFT, 1u>(i);
        row_thr += INCR_ROW;
    }
}

// effectively computes alpha*AB + beta*C and stores to shmem
template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int MMAS_ROW,
    unsigned int MMAS_COL,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void epilogue(
    unsigned int lane_idx,
    size_t stride_dst,
    float alpha,
    half* shmem_out_warp,
    half* shmem_out_base,
    half* dst,
    uint32_t (&regs_acc)[MMAS_ROW][MMAS_COL][(ROWS_MMA/8)*(COLS_MMA/8)]
) {
    static_assert(ROWS_MMA == 16 && COLS_MMA == 8); // for now DIM_MM = 16 and DIM_MN = 8 are fixed

    half2 alpha_h2 = __half2half2(__float2half(alpha));

    // swizzle
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment for toGmem
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    unsigned int frag_row = lane_idx >> 2;
    unsigned int frag_col = lane_idx & 3;

    // reduce instrs by XOR dst pointer with const
    uint32_t shared_base = __cvta_generic_to_shared(shmem_out_warp);
    uint32_t base_addr = shared_base + frag_row * COLS_BLOCK * sizeof(half) + frag_col * sizeof(uint32_t);
    uint32_t dst_addr = swizzle<BITMASK, SHIFT>(base_addr);
    constexpr uint32_t STRIDE_COL = COLS_MMA * sizeof(half);
    constexpr uint32_t STRIDE_ROW = ROWS_MMA * COLS_BLOCK * sizeof(half);
    constexpr uint32_t STRIDE_8 = 8 * COLS_BLOCK * sizeof(half);
    constexpr uint32_t PATTERN_8 = swizzle<BITMASK, SHIFT>(STRIDE_8);

    // regs -> shmem
#pragma unroll
    for (int mma_row = 0; mma_row < MMAS_ROW; ++mma_row) {

        const uint32_t row_addr = dst_addr;

#pragma unroll
        for (int mma_col = 0; mma_col < MMAS_COL; ++mma_col) {

            if (alpha != 1.0f) { // alpha = 0 caught on host side

                half2 lo = *reinterpret_cast<half2*>(&regs_acc[mma_row][mma_col][0]);
                half2 hi = *reinterpret_cast<half2*>(&regs_acc[mma_row][mma_col][1]);

                lo = __hmul2(lo, alpha_h2);
                hi = __hmul2(hi, alpha_h2);

                regs_acc[mma_row][mma_col][0] = *reinterpret_cast<uint32_t*>(&lo);
                regs_acc[mma_row][mma_col][1] = *reinterpret_cast<uint32_t*>(&hi);
            }

            asm(
                "st.shared.u32 [%0], %1;\n" 
                :: "r"(dst_addr), "r"(regs_acc[mma_row][mma_col][0]) 
                : "memory"
            );
            asm(
                "st.shared.u32 [%0], %1;\n" 
                :: "r"(dst_addr^PATTERN_8), "r"(regs_acc[mma_row][mma_col][1]) 
                : "memory"
            );

            dst_addr ^= xor_pattern<STRIDE_COL, BITMASK, SHIFT, 1u>(mma_col);
        }

        dst_addr = row_addr ^ xor_pattern<STRIDE_ROW, BITMASK, SHIFT, 1u>(mma_row);
    }

    __syncthreads();

    toGmem<ROWS_MMA, ROWS_BLOCK, COLS_BLOCK, NUM_THRS>(stride_dst, shmem_out_base, dst);
}

}
