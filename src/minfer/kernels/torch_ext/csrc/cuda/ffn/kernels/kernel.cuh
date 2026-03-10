#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

namespace minfer::impl {

template <int WARPS_PER_BLOCK>
__global__ void swiglu_f16_cuda_impl(
    size_t M, size_t K_GU, size_t N_GU,
    const half* __restrict__ in,
    const half* __restrict__ ws_up,
    const half* __restrict__ ws_gate,
    half* __restrict__ hb,
    half* __restrict__ hb2
) {

}

template <int WARPS_PER_BLOCK>
__global__ void swiglu_quant_cuda_impl(
    int up_qtype_int, int gate_qtype_int,
    int up_qblock_size, int gate_qblock_size,
    size_t up_qtype_size, size_t gate_qtype_size,
    size_t M, size_t K_GU, size_t N_GU,
    const half* __restrict__ in,
    const uint8_t* __restrict__ ws_up,
    const uint8_t* __restrict__ ws_gate,
    half* __restrict__ hb,
    half* __restrict__ hb2
) {

}

}