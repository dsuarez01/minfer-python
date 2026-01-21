#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"

namespace minfer::impl {
    template <int TILE_M, int N_EXPS, int TOP_K>
    __global__ void moe_scoring_cuda_impl(
        size_t M, size_t K,
        const half* __restrict__ x,
        const half* __restrict__ w,
        uint8_t* __restrict__ act_exps,
        half* __restrict__ act_exps_scores,
        half* __restrict__ scores
    ) {

        // __shared__ half l_shared[TILE_M][N_EXPS];
        // __shared__ float scratch[TILE_M*((N_EXPS+31)/32)];

        // const half* x_row = x + blockIdx.x*TILE_M*K;
        // half* scores_row = scores + blockIdx.x*TILE_M*N_EXPS;
        // uint8_t* act_exps_row = act_exps + blockIdx.x*TILE_M*TOP_K;
        // half* act_exps_scores_row = act_exps_scores + blockIdx.x*TILE_M*TOP_K;

        // int r = threadIdx.x/N_EXPS;
        // int c = threadIdx.x%N_EXPS;

        // // compute logits
        // float acc = 0.0f;
        // for (int k=0; k<K; k+=8) {
        //     float4 xv = *reinterpret_cast<const float4*>(x_row+r*K+k);
        //     float4 wv = *reinterpret_cast<const float4*>(w+c*K+k);
        //     half2* xh = reinterpret_cast<half2*>(&xv);
        //     half2* wh = reinterpret_cast<half2*>(&wv);
        //     #pragma unroll
        //     for (int i=0; i<4; ++i) {
        //         acc += __half2float(xh[i].x) * __half2float(wh[i].x);
        //         acc += __half2float(xh[i].y) * __half2float(wh[i].y);
        //     }
        // }
        // l_shared[r][c] = __float2half(acc);
        // __syncthreads();

        // // softmax
        // float val = __half2float(l_shared[r][c]);
        // float m = row_reduce_max<N_EXPS>(scratch, val, r, c);
        // float exp_val = expf(val-m);
        // float s = row_reduce_sum<N_EXPS>(scratch, exp_val, r, c);
        // float softmax_val = exp_val/s;
        // l_shared[r][c] = __float2half(softmax_val);
        // scores_row[r*N_EXPS+c] = __float2half(softmax_val);
        // __syncthreads();

        // // insertion sort topK, one thread per row
        // if (c == 0) {
        //     float top_vals[TOP_K];
        //     int top_idx[TOP_K];

        //     #pragma unroll
        //     for (int i=0; i<TOP_K; ++i) {
        //         top_vals[i] = -INFINITY;
        //         top_idx[i] = -1;
        //     }

        //     #pragma unroll
        //     for (int e=0; e<N_EXPS; ++e) {
        //         float v = __half2float(l_shared[r][e]);
        //         if (v>top_vals[TOP_K-1]) {
        //             top_vals[TOP_K-1] = v;
        //             top_idx[TOP_K-1] = e;
        //             #pragma unroll
        //             for (int i=TOP_K-1; i>0 && top_vals[i]>top_vals[i-1]; --i) {
        //                 float tv = top_vals[i]; top_vals[i] = top_vals[i-1]; top_vals[i-1] = tv;
        //                 int ti = top_idx[i]; top_idx[i] = top_idx[i-1]; top_idx[i-1] = ti;
        //             }
        //         }
        //     }

        //     #pragma unroll
        //     for (int i=0; i<TOP_K; ++i) {
        //         act_exps_row[r*TOP_K+i] = (uint8_t)top_idx[i];
        //         act_exps_scores_row[r*TOP_K+i] = __float2half(top_vals[i]);
        //     }
        // }
    }
}