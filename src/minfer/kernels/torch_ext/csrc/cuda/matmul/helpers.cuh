#pragma once

#include "common/types.hpp"

namespace minfer::impl {

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

    static_assert(NUM_THRS % COLS_BLOCK == 0);

    constexpr unsigned int row_incr = NUM_THRS / COLS_BLOCK;
    unsigned int thr_row = thr_idx / COLS_BLOCK;
    unsigned int thr_col = thr_idx % COLS_BLOCK;
    
    for (int r=thr_row; r<ROWS_BLOCK; r+=row_incr) {
        dst[r*COLS_BLOCK+thr_col] = src[r*src_stride+thr_col]; // this breaks down into 2 instrs, store to register then shmem
    }
}

__device__ __forceinline__ void toGmem_m16n8(
    size_t bytes_stride_dst,
    half* dst, 
    half (&reg)[4]
) {
    unsigned int laneIdx = threadIdx.x % 32;
    uint32_t (&reg_)[2] = reinterpret_cast<uint32_t(&)[2]>(reg);
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
    bytes_stride_dst /= sizeof(uint32_t);
    unsigned int frag_row = laneIdx/4;
    unsigned int frag_col = laneIdx%4;

    // 16 byte transactions per row 
    dst_ptr[frag_row*bytes_stride_dst+frag_col] = reg_[0]; // 4 bytes written per thr
    frag_row += 8;
    dst_ptr[frag_row*bytes_stride_dst+frag_col] = reg_[1]; // 4 bytes written per thr
}

}
