#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

template <int WARPS_PER_BLOCK>
__global__ void swiglu_f16_cuda_impl(
    int64_t M, int64_t K_GU, int64_t N_GU,
    const half* __restrict__ in,
    const half* __restrict__ ws_up,
    const half* __restrict__ ws_gate,
    half* __restrict__ hb,
    half* __restrict__ hb2
) {

    __shared__ half w_gu_shared[WMMA_N][WMMA_K];
    __shared__ half hb_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_N];
    __shared__ half hb2_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_N];

    const half* in_row = in + blockIdx.y*WARPS_PER_BLOCK*WMMA_M*K_GU;
    half* hb2_tile = hb2 + blockIdx.x*M*N_GU + blockIdx.y*WARPS_PER_BLOCK*WMMA_M*N_GU + blockIdx.z*WMMA_N;

    const half* ws_gate_row = ws_gate + blockIdx.x*N_GU*K_GU + blockIdx.z*WMMA_N*K_GU;
    const half* ws_up_row = ws_up + blockIdx.x*N_GU*K_GU + blockIdx.z*WMMA_N*K_GU;

    // gate proj stored in hb_row
    // compute_tile_wmma_f16(
    //     M, K_GU, N_GU,
    //     K_GU, K_GU, WMMA_N,
    //     w_gu_shared,
    //     in_row, ws_gate_row, (half*)hb_shared
    // );

    // up proj stored in hb2
    // compute_tile_wmma_f16(
    //     M, K_GU, N_GU,
    //     K_GU, K_GU, WMMA_N,
    //     w_gu_shared,
    //     in_row, ws_up_row, (half*)hb2_shared
    // );

    __syncthreads();

    // apply swish to hb (swish(x) = x*sigmoid(beta*x), beta taken to be 1 here)
    // then element-wise mult of hb_shared and hb2_shared
    for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*WMMA_N; idx+=blockDim.x) {
        int r = idx/WMMA_N;
        int c = idx%WMMA_N;
        float hbv = __half2float(hb_shared[r][c]);
        float hb2v = __half2float(hb2_shared[r][c]);
        hb2_tile[r*N_GU+c] = __float2half(hbv/(1.0f+expf(-hbv))*hb2v);
    }
}

template <int WARPS_PER_BLOCK>
__global__ void swiglu_quant_cuda_impl(
    int up_qtype_int, int gate_qtype_int,
    int up_qblock_size, int gate_qblock_size,
    int up_qtype_size, int gate_qtype_size,
    int64_t M, int64_t K_GU, int64_t N_GU,
    const half* __restrict__ in,
    const uint8_t* __restrict__ ws_up,
    const uint8_t* __restrict__ ws_gate,
    half* __restrict__ hb,
    half* __restrict__ hb2
) {

    __shared__ half w_gate_shared[WMMA_N][256];
    __shared__ half w_up_shared[WMMA_N][256];
    __shared__ half hb_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_N];
    __shared__ half hb2_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_N];

    const half* in_row = in + blockIdx.y*WARPS_PER_BLOCK*WMMA_M*K_GU;
    half* hb2_tile = hb2 + blockIdx.x*M*N_GU + blockIdx.y*WARPS_PER_BLOCK*WMMA_M*N_GU + blockIdx.z*WMMA_N;

    int64_t exp_gate_size = N_GU*(K_GU/gate_qblock_size)*gate_qtype_size;
    int64_t exp_up_size = N_GU*(K_GU/up_qblock_size)*up_qtype_size;

    const uint8_t* ws_gate_row = ws_gate + blockIdx.x*exp_gate_size + blockIdx.z*WMMA_N*(K_GU/gate_qblock_size)*gate_qtype_size;
    const uint8_t* ws_up_row = ws_up + blockIdx.x*exp_up_size + blockIdx.z*WMMA_N*(K_GU/up_qblock_size)*up_qtype_size;

    // gate proj stored in hb_row
    // compute_tile_quant(
    //     gate_qtype_int, gate_qblock_size, gate_qtype_size,
    //     M, K_GU, N_GU,
    //     K_GU, K_GU, WMMA_N,
    //     w_gate_shared,
    //     in_row, ws_gate_row, (half*)hb_shared
    // );
    
    // up proj stored in hb2
    // compute_tile_quant(
    //     up_qtype_int, up_qblock_size, up_qtype_size,
    //     M, K_GU, N_GU,
    //     K_GU, K_GU, WMMA_N,
    //     w_up_shared,
    //     in_row, ws_up_row, (half*)hb2_shared
    // );

    __syncthreads();

    // apply swish to hb (swish(x) = x*sigmoid(beta*x), beta taken to be 1 here)
    // then element-wise mult of hb_shared and hb2_shared
    for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*WMMA_N; idx+=blockDim.x) {
        int r = idx/WMMA_N;
        int c = idx%WMMA_N;
        float hbv = __half2float(hb_shared[r][c]);
        float hb2v = __half2float(hb2_shared[r][c]);
        hb2_tile[r*N_GU+c] = __float2half(hbv/(1.0f+expf(-hbv))*hb2v);
    }
}