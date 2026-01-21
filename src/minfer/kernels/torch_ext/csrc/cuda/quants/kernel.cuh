#pragma once

#include "common/types.hpp"
#include "impls.cuh"

namespace minfer::impl {

template <typename T>
__global__ void dequant_cuda_(
    GGMLQuantizationType qtype,
    int qblock_size,
    int qtype_size,
    const uint8_t* __restrict__ xr,
    T* __restrict__ y,
    size_t b,
    size_t k
) {

    const uint8_t* w = xr + blockIdx.x*b + blockIdx.y*static_cast<size_t>(qtype_size);
    T* out = y + blockIdx.x*k + blockIdx.y*static_cast<size_t>(qblock_size);

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_block_q4_0<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q4_1: dequant_block_q4_1<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q5_0: dequant_block_q5_0<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q5_1: dequant_block_q5_1<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q8_0: dequant_block_q8_0<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::MXFP4: dequant_block_mxfp4<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q2_K: dequant_block_q2_K<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q3_K: dequant_block_q3_K<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q4_K: dequant_block_q4_K<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q5_K: dequant_block_q5_K<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q6_K: dequant_block_q6_K<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::TQ1_0: dequant_block_tq1_0<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::TQ2_0: dequant_block_tq2_0<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_block_iq2_xxs<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ2_XS: dequant_block_iq2_xs<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ2_S: dequant_block_iq2_s<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_block_iq3_xxs<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ3_S: dequant_block_iq3_s<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ1_S: dequant_block_iq1_s<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ1_M: dequant_block_iq1_m<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ4_NL: dequant_block_iq4_nl<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::IQ4_XS: dequant_block_iq4_xs<T>(w, out, threadIdx.x); break;
        case GGMLQuantizationType::Q8_K: dequant_block_q8_K<T>(w, out, threadIdx.x); break;
    }
}

}