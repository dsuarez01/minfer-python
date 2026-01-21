#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

namespace minfer::impl {

    __global__ void qkv_f16_cuda_impl(
        size_t M, size_t K, size_t N_Q, size_t N_KV,
        const half* __restrict__ x,
        const half* __restrict__ wq,
        const half* __restrict__ wk,
        const half* __restrict__ wv,
        half* __restrict__ q_out,
        half* __restrict__ k_out,
        half* __restrict__ v_out
    ) {
        
    }

    __global__ void qkv_quant_cuda_impl(
        int q_qtype_int, int k_qtype_int, int v_qtype_int,
        int q_qblock_size,  int k_qblock_size, int v_qblock_size,
        size_t q_qtype_size, size_t k_qtype_size, size_t v_qtype_size,
        size_t M, size_t K, size_t N_Q, size_t N_KV,
        const half* __restrict__ x,
        const uint8_t* __restrict__ wq,
        const uint8_t* __restrict__ wk,
        const uint8_t* __restrict__ wv,
        half* __restrict__ q_out,
        half* __restrict__ k_out,
        half* __restrict__ v_out
    ) {
        
    }
}