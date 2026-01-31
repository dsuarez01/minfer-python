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
//         "{%0, %1}, [%2];\n"
//         : "=r"(reg[0]), "=r"(reg[1])
//         : "r"(tot_ofst_f1)
//     );

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
//         "{%0, %1}, [%2];\n"
//         : "=r"(reg[2]), "=r"(reg[3])
//         : "r"(tot_ofst_f2)
//     );
// }

// improvement 2: swizzling
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

//     // swizzling
//     constexpr unsigned int BITS = int_log2(ROWS_MMA);
//     constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
//     constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
//     constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);
    
//     // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
//     uint32_t ofst_mma = (mma_row * ROWS_MMA * COLS_BLOCK + mma_col * COLS_MMA) * sizeof(half);
//     uint32_t ofst_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK * sizeof(half);
//     uint32_t tot_ofst_f1 = src_ofst + ofst_mma + ofst_thr;
//     uint32_t tot_ofst_f2 = tot_ofst_f1 + 8*sizeof(half);

//     tot_ofst_f1 ^= (tot_ofst_f1 & BITMASK) >> SHIFT;
//     tot_ofst_f2 ^= (tot_ofst_f2 & BITMASK) >> SHIFT;

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
//         "{%0, %1}, [%2];\n"
//         : "=r"(reg[0]), "=r"(reg[1])
//         : "r"(tot_ofst_f1)
//     );

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
//         "{%0, %1}, [%2];\n"
//         : "=r"(reg[2]), "=r"(reg[3])
//         : "r"(tot_ofst_f2)
//     );
// }

// improvement 3: (pipelining) reduce instr cnt
// NOTE: recall that A^(A^B) = B
template<unsigned int STRIDE, unsigned int BITMASK, unsigned int SHIFT>
__device__ __forceinline__ constexpr uint32_t xor_pattern(unsigned int i) {
    auto swizzle = [](uint32_t x) constexpr { return x ^ ((x & BITMASK) >> SHIFT); };
    return swizzle((i+1)*STRIDE) ^ swizzle(i*STRIDE);
}

template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int MMAS_ROW,
    unsigned int MMAS_COL,
    unsigned int COLS_BLOCK
>
__device__ __forceinline__ void ldmatrix_a(
    unsigned int lane_idx,
    const half* src,
    uint32_t (&reg)[MMAS_ROW][MMAS_COL][4]
) {
    // load 16x16 frag into regs
    static_assert(ROWS_MMA == 16);
    static_assert(COLS_MMA == 16);

    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4));
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    constexpr uint32_t STRIDE_COL = COLS_MMA * sizeof(half);
    constexpr uint32_t STRIDE_ROW = ROWS_MMA * COLS_BLOCK * sizeof(half);
    constexpr uint32_t STRIDE_8 = 8 * sizeof(half);
    constexpr uint32_t PATTERN_8 = STRIDE_8 ^ ((STRIDE_8 & BITMASK) >> SHIFT);

    const uint32_t shared_base = __cvta_generic_to_shared(src);

    // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
    unsigned int idx_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK;
    uint32_t base_addr = shared_base + idx_thr * sizeof(half);
    uint32_t src_addr = base_addr ^ ((base_addr & BITMASK) >> SHIFT);

    #pragma unroll
    for (int mma_row = 0; mma_row < MMAS_ROW; ++mma_row) {
        
        const uint32_t row_addr = src_addr;

        #pragma unroll
        for (int mma_col = 0; mma_col < MMAS_COL; ++mma_col) {

            asm(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1])
                : "r"(src_addr)
            );

            asm(
                "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(reg[mma_row][mma_col][2]), "=r"(reg[mma_row][mma_col][3])
                : "r"(src_addr^PATTERN_8)
            );

            src_addr ^= xor_pattern<STRIDE_COL, BITMASK, SHIFT>(mma_col);
        }

        src_addr = row_addr ^ xor_pattern<STRIDE_ROW, BITMASK, SHIFT>(mma_row);
    }
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
//         "{%0, %1}, [%2];\n"
//         : "=r"(reg[0]), "=r"(reg[1])
//         : "r"(tot_ofst)
//     );
// }

// improvement 2: swizzling
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

//     // swizzling
//     constexpr unsigned int BITS = int_log2(ROWS_MMA);
//     constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
//     constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
//     constexpr unsigned int BITMASK  = ((1u << BITS) - 1) << (BASE + SHIFT);

//     // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
//     uint32_t ofst_mma = (mma_row * ROWS_MMA * COLS_BLOCK + mma_col * COLS_MMA) * sizeof(half);
//     uint32_t ofst_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK * sizeof(half);
//     uint32_t tot_ofst = src_ofst + ofst_mma + ofst_thr;

//     tot_ofst ^= (tot_ofst & BITMASK) >> SHIFT;

//     asm volatile (
//         "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
//         "{%0, %1}, [%2];\n"
//         : "=r"(reg[0]), "=r"(reg[1])
//         : "r"(tot_ofst)
//     );
// }

// improvement 3: (pipelining) reduce instr cnt
template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int MMAS_ROW,
    unsigned int MMAS_COL,
    unsigned int COLS_BLOCK
>
__device__ __forceinline__ void ldmatrix_b(
    unsigned int lane_idx,
    const half* src,
    uint32_t (&reg)[MMAS_ROW][MMAS_COL][2]
) {
    // load 16x8 frag into regs
    static_assert(ROWS_MMA == 16);
    static_assert(COLS_MMA == 8);

    // swizzling
    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4));
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    constexpr uint32_t STRIDE_COL = COLS_MMA * sizeof(half);
    constexpr uint32_t STRIDE_ROW = ROWS_MMA * COLS_BLOCK * sizeof(half);

    const uint32_t shared_base = __cvta_generic_to_shared(src);

    // see: https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/
    unsigned int idx_thr = (((lane_idx/8)%2)*8 + (lane_idx%8)) * COLS_BLOCK;
    uint32_t base_addr = shared_base + idx_thr * sizeof(half);
    uint32_t src_addr = base_addr ^ ((base_addr & BITMASK) >> SHIFT);

    #pragma unroll
    for (int mma_row = 0; mma_row < MMAS_ROW; ++mma_row) {
        
        const uint32_t row_addr = src_addr;

        #pragma unroll
        for (int mma_col = 0; mma_col < MMAS_COL; ++mma_col) {

            asm(
                "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                "{%0, %1}, [%2];\n"
                : "=r"(reg[mma_row][mma_col][0]), "=r"(reg[mma_row][mma_col][1])
                : "r"(src_addr)
            );

            src_addr ^= xor_pattern<STRIDE_COL, BITMASK, SHIFT>(mma_col);
        }

        src_addr = row_addr ^ xor_pattern<STRIDE_ROW, BITMASK, SHIFT>(mma_row);
    }
}

// baseline
__device__ __forceinline__ void mma_sync_m16n8k16(
    uint32_t (&x_reg)[4],
    uint32_t (&w_reg)[2],
    uint32_t (&acc_reg)[2]
) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
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

// improvement 3: (pipelining) async primitives, async toShmem
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
    asm volatile("cp.async.commit_group;\n" ::);
}

template<unsigned int N>
__device__ __forceinline__ void cp_async_wait() {
    static_assert(N <= 7);
    
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
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK  = ((1u<<BITS)-1) << (BASE+SHIFT);

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    // can issue up to 128-bit load/store instrs (8 half elements)
    // alignment checks
    static_assert(COLS_BLOCK % 8 == 0);
    // assert(stride_src % 8 == 0);
    
    size_t eff_stride_src = stride_src / 8;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);
    
    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;

    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;
    
    // reduce instrs by XOR dst pointer with const
    constexpr uint32_t STRIDE_DST = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

    uint32_t shared_base = __cvta_generic_to_shared(dst);
    unsigned int idx_dst = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t dst_addr = shared_base + idx_dst*sizeof(int4);
    dst_addr ^= (dst_addr & BITMASK) >> SHIFT;

    #pragma unroll
    for (int i=0; i<NUM_ITERS; ++i) {
        const void* src_global = &reinterpret_cast<const int4*>(src)[row_thr*eff_stride_src+col_thr];
        cp_async<16>(dst_addr, src_global);
        dst_addr ^= xor_pattern<STRIDE_DST, BITMASK, SHIFT>(i);
        row_thr += INCR_ROW;
    }
}

// baseline
// (named after the Hopper instruction)
// __device__ __forceinline__ void stmatrix_m16n8(
//     unsigned int lane_idx,
//     size_t stride_dst,
//     half* dst,
//     uint32_t (&reg)[2]
// ) {

//     uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
//     stride_dst /= 2;
    
//     // (16 byte transactions per row)
//     unsigned int frag_col = lane_idx%4;
//     unsigned int frag_row = lane_idx/4;

//     dst_ptr[frag_row*stride_dst+frag_col] = reg[0]; // 4 bytes written per thr
//     frag_row += 8;
//     dst_ptr[frag_row*stride_dst+frag_col] = reg[1]; // 4 bytes written per thr
// }

// improvement 3: (pipelining) mitigating uncoalesced stores to gmem
template <
    unsigned int ROWS_MMA,
    unsigned int COLS_MMA,
    unsigned int MMAS_ROW,
    unsigned int MMAS_COL,
    unsigned int COLS_BLOCK
>
__device__ __forceinline__ void stmatrix(
    unsigned int lane_idx,
    half* dst,
    uint32_t (&reg)[MMAS_ROW][MMAS_COL][2]
) {
    static_assert(ROWS_MMA == 16);
    static_assert(COLS_MMA == 8);

    constexpr unsigned int BITS = int_log2(ROWS_MMA);
    constexpr unsigned int BASE = int_log2(sizeof(int4)); // 16-byte alignment for toGmem
    constexpr unsigned int SHIFT = int_log2(COLS_BLOCK / 8);
    constexpr unsigned int BITMASK = ((1u<<BITS)-1) << (BASE+SHIFT);

    constexpr uint32_t STRIDE_COL = COLS_MMA * sizeof(half);
    constexpr uint32_t STRIDE_ROW = ROWS_MMA * COLS_BLOCK * sizeof(half);
    constexpr uint32_t STRIDE_8 = 8 * COLS_BLOCK * sizeof(half);
    constexpr uint32_t PATTERN_8 = STRIDE_8 ^ ((STRIDE_8 & BITMASK) >> SHIFT);

    uint32_t shared_base = __cvta_generic_to_shared(dst);

    unsigned int frag_col = lane_idx%4;
    unsigned int frag_row = lane_idx/4;

    uint32_t base_addr = shared_base + frag_row * COLS_BLOCK * sizeof(half) + frag_col * sizeof(uint32_t);
    uint32_t dst_addr = base_addr ^ ((base_addr & BITMASK) >> SHIFT);

    #pragma unroll
    for (int mma_row = 0; mma_row < MMAS_ROW; ++mma_row) {

        const uint32_t row_addr = dst_addr;

        #pragma unroll
        for (int mma_col = 0; mma_col < MMAS_COL; ++mma_col) {

            asm(
                "st.shared.u32 [%0], %1;\n" 
                :: "r"(dst_addr), "r"(reg[mma_row][mma_col][0]) 
                : "memory"
            );
            asm(
                "st.shared.u32 [%0], %1;\n" 
                :: "r"(dst_addr^PATTERN_8), "r"(reg[mma_row][mma_col][1]) 
                : "memory"
            );

            dst_addr ^= xor_pattern<STRIDE_COL, BITMASK, SHIFT>(mma_col);
        }

        dst_addr = row_addr ^ xor_pattern<STRIDE_ROW, BITMASK, SHIFT>(mma_row);
    }
}

// improvement 3: (pipelining) mitigating uncoalesced stores to gmem
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

    size_t eff_stride_dst = stride_dst / 8;
    constexpr unsigned int EFF_COLS_BLOCK = COLS_BLOCK / 8;

    static_assert(NUM_THRS % EFF_COLS_BLOCK == 0);

    constexpr unsigned int INCR_ROW = NUM_THRS / EFF_COLS_BLOCK;
    constexpr unsigned int NUM_ITERS = ROWS_BLOCK / INCR_ROW;
    constexpr uint32_t STRIDE_SRC = INCR_ROW * EFF_COLS_BLOCK * sizeof(int4);

    unsigned int row_thr = idx_thr / EFF_COLS_BLOCK;
    unsigned int col_thr = idx_thr % EFF_COLS_BLOCK;

    uint32_t shared_base = __cvta_generic_to_shared(src);
    uint32_t idx_src = row_thr*EFF_COLS_BLOCK+col_thr;
    uint32_t src_addr = shared_base + idx_src*sizeof(int4);
    src_addr ^= (src_addr & BITMASK) >> SHIFT;

    int4 src_vals;

    #pragma unroll
    for (int i = 0; i < NUM_ITERS; ++i) {
        asm(
            "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(src_vals.x), "=r"(src_vals.y), "=r"(src_vals.z), "=r"(src_vals.w)
            : "r"(src_addr)
        );
        reinterpret_cast<int4*>(dst)[row_thr*eff_stride_dst+col_thr] = src_vals;
        src_addr ^= xor_pattern<STRIDE_SRC, BITMASK, SHIFT>(i);
        row_thr += INCR_ROW;
    }
}

}
