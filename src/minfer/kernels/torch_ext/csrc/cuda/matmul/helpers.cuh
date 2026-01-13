#pragma once

#include "common/types.hpp"

template <
    unsigned int ROWS_BLOCK,
    unsigned int COLS_BLOCK,
    unsigned int NUM_THRS
>
__device__ __forceinline__ void toShmem(
    size_t src_stride,
    const half* __restrict__ src,
    half* __restrict__ dst
) {

    unsigned int thr_idx = threadIdx.y * blockDim.x + threadIdx.x;

    assert(NUM_THRS % COLS_BLOCK == 0);

    constexpr unsigned int row_incr = NUM_THRS / COLS_BLOCK;
    unsigned int thr_row = thr_idx / COLS_BLOCK;
    unsigned int thr_col = thr_idx % COLS_BLOCK;
    
    for (int r=thr_row; r<ROWS_BLOCK; r+=row_incr) {
        dst[r*COLS_BLOCK+thr_col] = src[r*src_stride+thr_col];
    }
}

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void *ptr) {
    uint32_t addr;
    asm volatile("{\n\t"
        "  .reg .u64 u64addr;\n\t"
        "  cvta.to.shared.u64 u64addr, %1;\n\t"
        "  cvt.u32.u64 %0, u64addr;\n\t"
        "}"
        : "=r"(addr)
        : "l"(ptr));
    return addr;
  }

