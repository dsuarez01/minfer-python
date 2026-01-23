#pragma once

#include <cassert>
#include "common/types.hpp"

namespace minfer::impl {


// baseline
__device__ __forceinline__ void toShmem(
    unsigned int rows_block,
    unsigned int cols_block,
    size_t stride_src,
    const half* __restrict__ src,
    half* __restrict__ dst
) {

    unsigned int idx_thr = threadIdx.y * blockDim.x + threadIdx.x;

    unsigned int num_thrs = blockDim.x * blockDim.y;

    assert(num_thrs % cols_block == 0);

    unsigned int incr_row = num_thrs / cols_block;
    unsigned int row_thr = idx_thr / cols_block;
    unsigned int col_thr = idx_thr % cols_block;
    
    for (int r=row_thr; r<rows_block; r+=incr_row) {
        dst[r*cols_block+col_thr] = src[r*stride_src+col_thr]; // this breaks down into 2 instrs, store to register then shmem
    }
}

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

//     static_assert(NUM_THRS % COLS_BLOCK == 0);

//     constexpr unsigned int incr_row = NUM_THRS / COLS_BLOCK;
//     unsigned int row_thr = idx_thr / COLS_BLOCK;
//     unsigned int col_thr = idx_thr % COLS_BLOCK;
    
//     for (int r=row_thr; r<ROWS_BLOCK; r+=incr_row) {
//         dst[r*COLS_BLOCK+col_thr] = src[r*stride_src+col_thr]; // this breaks down into 2 instrs, store to register then shmem
//     }
// }

// baseline
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
