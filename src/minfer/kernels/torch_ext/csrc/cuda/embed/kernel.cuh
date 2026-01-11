#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

__global__ void embed_f16_cuda_impl(
    int64_t hidden_dim,
    const int64_t* __restrict__ token_ids,
    const half* __restrict__ w,
    half* __restrict__ x
) {
    int64_t L = gridDim.y;

    int64_t iB = blockIdx.x;
    int64_t iL = blockIdx.y;
    int64_t token_id = token_ids[iB*L+iL];

    const half* w_row = w + token_id*hidden_dim;
    half* x_row = x + iB*L*hidden_dim + iL*hidden_dim;

    for (int idx=threadIdx.x; idx<hidden_dim; idx+=blockDim.x) {
        x_row[idx] = w_row[idx];
    }
}

__global__ void embed_quant_cuda_impl(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    int64_t b, // bytes per row
    int64_t k, // dequant elems per row
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ w,
    half* __restrict__ x
) {
    int64_t L = gridDim.y;

    int64_t iB = blockIdx.x;
    int64_t iL = blockIdx.y;
    int64_t block_in_row = blockIdx.z;
    int64_t token_id = token_ids[iB*L+iL];
    
    const uint8_t* w_block = w + token_id*b + block_in_row*qtype_size;
    half* x_block = x + iB*L*k + iL*k + block_in_row*qblock_size;
    dequant_block(qtype_int, threadIdx.x, x_block, w_block);
}