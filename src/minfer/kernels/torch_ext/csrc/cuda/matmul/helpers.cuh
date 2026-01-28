#pragma once

#include <cuda_fp16.h>
#include <vector_types.h>
#include <cassert>

#include "common/types.hpp"

namespace minfer::impl {

constexpr unsigned int int_log2(unsigned int n) {
    unsigned int result = 0;
    while (n >>= 1) result++;
    return result;
}

// baseline
// template <
//     unsigned int ROWS_MMA,
//     unsigned int COLS_MMA,
//     unsigned int COLS_BLOCK
// >
// __device__ __forceinline__ void ldmatrix_m16n16(
//     int mma_row,
//     int mma_col,
//     unsigned int lane_idx,
//     uint32_t src_ofst,
//     uint32_t (&reg)[4]
// ) {

//     // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
//     uint32_t ofst_mma = (mma_row * ROWS_MMA * COLS_BLOCK + mma_col * COLS_MMA) * sizeof(half);
//     uint32_t ofst_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK * sizeof(half);
//     uint32_t tot_ofst_f1 = src_ofst + ofst_mma + ofst_thr;
//     uint32_t tot_ofst_f2 = tot_ofst_f1 + 8*sizeof(half);

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
//         "{%0, %1}, [%2];"
//         : "=r"(reg[0]), "=r"(reg[1])
//         : "r"(tot_ofst_f1)
//     );

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
//         "{%0, %1}, [%2];"
//         : "=r"(reg[2]), "=r"(reg[3])
//         : "r"(tot_ofst_f2)
//     );
// }

// improvement 2: swizzling
template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int COLS_BLOCK
>
__device__ __forceinline__ void ldmatrix_m16n16(
    int mma_row,
    int mma_col,
    unsigned int lane_idx,
    uint32_t src_ofst,
    uint32_t (&reg)[4]
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);
    
    // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
    uint32_t ofst_mma = (mma_row * ROWS_MMA * COLS_BLOCK + mma_col * COLS_MMA) * sizeof(half);
    uint32_t ofst_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK * sizeof(half);
    uint32_t tot_ofst_f1 = src_ofst + ofst_mma + ofst_thr;
    uint32_t tot_ofst_f2 = tot_ofst_f1 + 8*sizeof(half);

    tot_ofst_f1 ^= (tot_ofst_f1 & BITMASK) >> SHIFT;
    tot_ofst_f2 ^= (tot_ofst_f2 & BITMASK) >> SHIFT;

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(tot_ofst_f1)
    );

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(reg[2]), "=r"(reg[3])
        : "r"(tot_ofst_f2)
    );
}

// baseline
// template <
//     unsigned int ROWS_MMA,
//     unsigned int COLS_MMA,
//     unsigned int COLS_BLOCK
// >
// __device__ __forceinline__ void ldmatrix_m16n8(
//     int mma_row,
//     int mma_col,
//     unsigned int lane_idx,
//     uint32_t src_ofst,
//     uint32_t (&reg)[2]
// ) {

//     // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
//     uint32_t ofst_mma = (mma_row * ROWS_MMA * COLS_BLOCK + mma_col * COLS_MMA) * sizeof(half);
//     uint32_t ofst_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK * sizeof(half);
//     uint32_t tot_ofst = src_ofst + ofst_mma + ofst_thr;

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
//         "{%0, %1}, [%2];"
//         : "=r"(reg[0]), "=r"(reg[1])
//         : "r"(tot_ofst)
//     );
// }

// improvement 2: swizzling
template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int COLS_BLOCK
>
__device__ __forceinline__ void ldmatrix_m16n8(
    int mma_row,
    int mma_col,
    unsigned int lane_idx,
    uint32_t src_ofst,
    uint32_t (&reg)[2]
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);

    // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
    uint32_t ofst_mma = (mma_row * ROWS_MMA * COLS_BLOCK + mma_col * COLS_MMA) * sizeof(half);
    uint32_t ofst_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK * sizeof(half);
    uint32_t tot_ofst = src_ofst + ofst_mma + ofst_thr;

    tot_ofst ^= (tot_ofst & BITMASK) >> SHIFT;

    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(reg[0]), "=r"(reg[1])
        : "r"(tot_ofst)
    );
}

// baseline
__device__ __forceinline__ void mma_sync_m16n8k16(
    int mma_m,
    int mma_k,
    int mma_n,
    uint32_t (&x_reg)[4],
    uint32_t (&w_reg)[2],
    uint32_t (&acc_reg)[2]
) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};"
        : "=r"(acc_reg[0]), "=r"(acc_reg[1])
        : "r"(x_reg[0]), "r"(x_reg[1]), "r"(x_reg[2]), "r"(x_reg[3]),
          "r"(w_reg[0]), "r"(w_reg[1]),
          "r"(acc_reg[0]), "r"(acc_reg[1])
    );
}

// baseline (basic tiling)
// __device__ __forceinline__ void toShmem(
//     unsigned int rows_block,
//     unsigned int cols_block,
//     size_t stride_src,
//     const half* __restrict__ src,
//     half* __restrict__ dst
// ) {

//     unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

//     unsigned int num_thrs = blockDim.x * blockDim.y;

//     assert(num_thrs % cols_block == 0);

//     unsigned int incr_row = num_thrs / cols_block;
//     unsigned int row_thr = idx_thr / cols_block;
//     unsigned int col_thr = idx_thr % cols_block;
    
//     for (int r=row_thr; r<rows_block; r+=incr_row) {
//         dst[r*cols_block+col_thr] = src[r*stride_src+col_thr]; // this breaks down into 2 instrs, store to register then shmem
//     }
// }

// improvement 1:
// loop unrolling, vectorization of loads->stores
// template <
//     unsigned int ROWS_BLOCK,
//     unsigned int COLS_BLOCK,
//     unsigned int NUM_THRS
// >
// __device__ __forceinline__ void toShmem(
//     size_t stride_src,
//     const half* __restrict__ src,
//     half* __restrict__ dst
// ) {

//     unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

//     // can issue up to 128-bit load/store instrs (8 half elements)
//     // alignment checks
//     static_assert(COLS_BLOCK % 8 == 0);
//     assert(stride_src % 8 == 0);
    
//     size_t eff_stride_src = stride_src / 8;
//     constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

//     static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
//     constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
//     constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;

//     unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
//     unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;
    
//     #pragma unroll
//     for (int i=0; i<NUM_ITERS; ++i) {
//         reinterpret_cast<int4*>(dst)[row_thr*EFF_COLS_BLOCK+col_thr] = 
//         reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
//         row_thr += INCR_ROW;
//     }
// }

// improvement 2:
// swizzling
// template <
//     unsigned int ROWS_MMA,
//     unsigned int ROWS_BLOCK,
//     unsigned int COLS_BLOCK,
//     unsigned int NUM_THRS
// >
// __device__ __forceinline__ void toShmem(
//     size_t stride_src,
//     const half* __restrict__ src,
//     half* __restrict__ dst
// ) {

//     // swizzling
//     constexpr unsigned int BITS = int_log2(ROWS_MMA);
//     constexpr unsigned int BASE = 0; // no byte alignment to preserve
//     constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
//     constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);

//     unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

//     // can issue up to 128-bit load/store instrs (8 half elements)
//     // alignment checks
//     static_assert(COLS_BLOCK % 8 == 0);
//     assert(stride_src % 8 == 0);
    
//     size_t eff_stride_src = stride_src / 8;
//     constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

//     static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
//     constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
//     constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;

//     unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
//     unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;
    
//     #pragma unroll
//     for (int i=0; i<NUM_ITERS; ++i) {
//         unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
//         idx_dst ^= (idx_dst & BITMASK) >> SHIFT;
//         reinterpret_cast<int4*>(dst)[idx_dst] = 
//         reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
//         row_thr += INCR_ROW;
//     }
// }

// improvement 3: async primitives, async toShmem
template<unsigned int N>
__device__ __forceinline__ void cp_async(
    uint32_t dst_shared,
    const void* src_global
) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-non-bulk-copy
    static_assert(N == 4 || N == 8 || N == 16);

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2;\n"
        :: "r"(dst_shared), "l"(src_global), "n"(N)
    );
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template<unsigned int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template <
    unsigned int ROWS_MMA,
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void toShmem(
    size_t stride_src,
    const half* __restrict__ src,
    half* __restrict__ dst
) {

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = 0; // no byte alignment to preserve
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    // can issue up to 128-bit load/store instrs (8 half elements)
    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);
    assert(stride_src % 8 == 0);
    
    size_t eff_stride_src = stride_src / 8;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;

    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;
    
    uint32_t dst_shared_base = __cvta_generic_to_shared(dst);

    #pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
        idx_dst ^= (idx_dst & BITMASK) >> SHIFT;
        
        uint32_t dst_shared = dst_shared_base + idx_dst*sizeof(int4);
        const void* src_global = &reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
        cp_async<16>(dst_shared, src_global);
        row_thr += INCR_ROW;
    }
}

// baseline
// (named after the Hopper instruction)
__device__ __forceinline__ void stmatrix_m16n8(
    unsigned int lane_idx,
    size_t bytes_stride_dst,
    half* dst,
    uint32_t (&reg)[2]
) {

    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
    bytes_stride_dst /= sizeof(uint32_t);
    
    // (16 byte transactions per row)
    unsigned int frag_col = lane_idx%4;
    unsigned int frag_row = lane_idx/4;

    dst_ptr[frag_row*bytes_stride_dst+frag_col] = reg[0]; // 4 bytes written per thr
    frag_row += 8;
    dst_ptr[frag_row*bytes_stride_dst+frag_col] = reg[1]; // 4 bytes written per thr
}

}
