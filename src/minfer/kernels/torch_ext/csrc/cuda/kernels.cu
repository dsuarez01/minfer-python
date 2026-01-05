/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Modified by dsuarez01
 * Reworking performant WMMA GEMM kernels to fit this use case.
 */

#include <Python.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>

#include <cassert>
#include <algorithm>
#include <type_traits>
#include <iostream>
#include <float.h>

#include "quants_impl.cuh"
#include "impl_common.hpp"

using namespace nvcuda;

extern "C" {
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

static void __checkKernelErrors(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << file << ":" << line << ": '" << expr << "' failed: " << cudaGetErrorString(result) << "\n";
        std::abort();
    }
}

#define checkKernelErrors(expr) __checkKernelErrors((expr), #expr, __FILE__, __LINE__)

// TODO: complete me!
namespace minfer {

template <typename T>
static __device__ T convert_float(float v) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(v);
    } else {
        return v;
    }
}

// helpers for warp-level and block-level reductions
// refer to: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
template <int WIDTH=32>
static __device__ float warp_reduce_sum(float v) {
    unsigned mask = (1ull << WIDTH) - 1;

    for (int offset = WIDTH/2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, WIDTH);
    }
    return __shfl_sync(mask, v, 0, WIDTH);
}

template <int WIDTH=32>
static __device__ float warp_reduce_max(float v) {
    unsigned mask = (1ull << WIDTH) - 1;

    for (int offset = WIDTH/2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(mask, v, offset, WIDTH));
    }
    return __shfl_sync(mask, v, 0, WIDTH);
}

static __device__ float blockreduce_sum(float* vs, float v, int tid) {
    int warp_id = tid/32;
    int lane_id = tid%32;
    int n_warps = blockDim.x/32;

    v = warp_reduce_sum(v);
    if (lane_id == 0) vs[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        v = (lane_id < n_warps) ? vs[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
    }
    if (tid == 0) vs[0] = v;
    __syncthreads();
    return vs[0];
}

static __device__ float blockreduce_max(float* vs, float v, int tid) {
    int warp_id = tid/32;
    int lane_id = tid%32;
    int n_warps = blockDim.x/32;

    v = warp_reduce_max(v);
    if (lane_id == 0) vs[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        v = (lane_id < n_warps) ? vs[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
    }
    if (tid == 0) vs[0] = v;
    __syncthreads();
    return vs[0];
}

// helper for dispatch in matmul_cuda_impl
// might template this later on to additionally support float
static __device__ void dequant_block(
    int64_t qtype_int,
    int64_t tid,
    half* __restrict__ y,
    const uint8_t* __restrict__ w
) {
    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_block_q4_0<half>(w, y, tid); break;
        case GGMLQuantizationType::Q4_1: dequant_block_q4_1<half>(w, y, tid); break;
        case GGMLQuantizationType::Q5_0: dequant_block_q5_0<half>(w, y, tid); break;
        case GGMLQuantizationType::Q5_1: dequant_block_q5_1<half>(w, y, tid); break;
        case GGMLQuantizationType::Q8_0: dequant_block_q8_0<half>(w, y, tid); break;
        case GGMLQuantizationType::MXFP4: dequant_block_mxfp4<half>(w, y, tid); break;
        case GGMLQuantizationType::Q2_K: dequant_block_q2_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q3_K: dequant_block_q3_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q4_K: dequant_block_q4_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q5_K: dequant_block_q5_K<half>(w, y, tid); break;
        case GGMLQuantizationType::Q6_K: dequant_block_q6_K<half>(w, y, tid); break;
        case GGMLQuantizationType::TQ1_0: dequant_block_tq1_0<half>(w, y, tid); break;
        case GGMLQuantizationType::TQ2_0: dequant_block_tq2_0<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_block_iq2_xxs<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ2_XS: dequant_block_iq2_xs<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ2_S: dequant_block_iq2_s<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_block_iq3_xxs<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ3_S: dequant_block_iq3_s<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ1_S: dequant_block_iq1_s<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ1_M: dequant_block_iq1_m<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ4_NL: dequant_block_iq4_nl<half>(w, y, tid); break;
        case GGMLQuantizationType::IQ4_XS: dequant_block_iq4_xs<half>(w, y, tid); break;
        case GGMLQuantizationType::Q8_K: dequant_block_q8_K<half>(w, y, tid); break;
        case GGMLQuantizationType::BF16: dequant_block_bf16<half>(w, y, tid); break;
        default: assert(false && "Unsupported dtype"); // this gets compiled out in non-debug builds...
    }
}

__global__ void rmsnorm_cuda_impl(
    int64_t dim,
    float eps,
    const half* __restrict__ in,
    const half* __restrict__ w,
    half* __restrict__ out
) {

    const half* vec_in = in + blockIdx.x * dim;
    half* vec_out = out + blockIdx.x * dim;
    
    __shared__ float shared_sum[32];
    
    // one pass for the squared sum (then parallel reduction over block)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i<dim; i += blockDim.x) {
        float val = __half2float(vec_in[i]);
        sum_sq += val*val;
    }

    sum_sq = blockreduce_sum(shared_sum, sum_sq, threadIdx.x);
    float sc = rsqrt(sum_sq / float(dim) + eps);
    
    // one pass to apply weight and scale
    for (int i = threadIdx.x; i<dim; i += blockDim.x) {
        float in_f = __half2float(vec_in[i]);
        float w_f = __half2float(w[i]);
        vec_out[i] = __float2half(in_f*w_f*sc);
    }
}

void rmsnorm_cuda(
    double eps, 
    const at::Tensor& in, 
    const at::Tensor& w,
    at::Tensor& out
) {
    // checks
    TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    
    TORCH_CHECK(in.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(in.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);

    TORCH_CHECK(in.dim() == 3 || in.dim() == 4);


    TORCH_CHECK(in.sizes().equals(out.sizes()));
    TORCH_CHECK(in.size(-1) == w.size(0) || in.size(-1) % w.size(0) == 0); // per-head or per-entire vector
    
    const half* in_ptr = reinterpret_cast<const half*>(in.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());

    // handles both [B,L,D] and [B,L,n_heads,head_dim]
    int dim = w.size(0);
    int n_blocks = in.numel() / dim;

    int block_size = min(1024, max(128, (dim+8-1)/8));
    block_size = (block_size+32-1)/32 * 32;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_cuda_impl<<<n_blocks, block_size, 0, stream>>>(dim, eps, in_ptr, w_ptr, out_ptr);
}

__global__ void il_rope_cuda_impl(
    int64_t n_heads,
    int64_t rotary_dim,
    int64_t head_dim,
    int64_t start_pos,
    float freq_base,
    half* __restrict__ x
) {
    int L = gridDim.z / ((rotary_dim/2+31)/32);
    int head_idx = blockIdx.y;
    int pair_block = blockIdx.z % ((rotary_dim/2+31)/32);
    int seq_idx = blockIdx.z / ((rotary_dim/2+31)/32);
    int pair_idx = pair_block * 32 + threadIdx.x;
    int pos = start_pos + seq_idx;
    
    if (pair_idx >= rotary_dim / 2) return;

    half* x_head = x + (blockIdx.x*n_heads*L + head_idx*L + seq_idx)*head_dim;

    float freq = 1.0f / pow(freq_base, 2.0f * pair_idx / rotary_dim);
    float angle = pos * freq;

    int idx = 2*pair_idx;

    float x_0 = __half2float(x_head[idx]);
    float x_1 = __half2float(x_head[idx+1]);

    x_head[idx] = __float2half(cos(angle)*x_0 - sin(angle)*x_1);
    x_head[idx+1] = __float2half(sin(angle)*x_0 + cos(angle)*x_1);
}

void il_rope_cuda(
    int64_t rotary_dim,
    int64_t start_pos,
    double freq_base,
    at::Tensor& x // [B,L,n_heads,head_dim]
) {
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(rotary_dim <= x.size(3));

    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    int B = x.size(0);
    int n_heads = x.size(1);
    int L = x.size(2);
    int head_dim = x.size(3);

    dim3 grid(B, n_heads, L*((rotary_dim/2+31)/32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    il_rope_cuda_impl<<<grid, 32, 0, stream>>>(n_heads, rotary_dim, head_dim, start_pos, freq_base, x_ptr);
}

__global__ void neox_rope_cuda_impl(
    int64_t n_heads,
    int64_t rotary_dim,
    int64_t head_dim,
    int64_t start_pos,
    float freq_base,
    half* __restrict__ x
) {
    int L = gridDim.z / ((rotary_dim/2+31)/32);
    int head_idx = blockIdx.y;
    int pair_block = blockIdx.z % ((rotary_dim/2+31)/32);
    int seq_idx = blockIdx.z / ((rotary_dim/2+31)/32);
    int pair_idx = pair_block * 32 + threadIdx.x;
    int pos = start_pos + seq_idx;
    
    if (pair_idx >= rotary_dim / 2) return;

    half* x_head = x + ((int64_t)blockIdx.x*n_heads*L + head_idx*L + seq_idx)*head_dim;

    float freq = 1.0f / pow(freq_base, 2.0f * pair_idx / rotary_dim);
    float angle = pos * freq;

    float x_0 = __half2float(x_head[pair_idx]);
    float x_1 = __half2float(x_head[pair_idx+rotary_dim/2]);

    x_head[pair_idx] = __float2half(cos(angle)*x_0 - sin(angle)*x_1);
    x_head[pair_idx+rotary_dim/2] = __float2half(sin(angle)*x_0 + cos(angle)*x_1);
}

void neox_rope_cuda(
    int64_t rotary_dim,
    int64_t start_pos,
    double freq_base,
    at::Tensor& x // [B,L,n_heads,head_dim]
) {
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(rotary_dim <= x.size(-1));

    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    int B = x.size(0);
    int n_heads = x.size(1);
    int L = x.size(2);
    int head_dim = x.size(3);

    dim3 grid(B, n_heads, L*((rotary_dim/2+31)/32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    neox_rope_cuda_impl<<<grid, 32, 0, stream>>>(n_heads, rotary_dim, head_dim, start_pos, freq_base, x_ptr);
}

// constants for WMMA GEMM
constexpr int TILE_SIZE = 128;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK; // 256

constexpr int CHUNK_K = 8; // each warp processes 8 k-tiles at once
constexpr int CHUNK_LINE_BYTES = CHUNK_K * WMMA_K * sizeof(half); // line (row) is 256 bytes
constexpr int WARP_COPY_BYTES = WARP_SIZE * sizeof(int4); // warp responsible for copying 512 bytes 
constexpr int CHUNK_COPY_LINES_PER_WARP = WARP_COPY_BYTES / CHUNK_LINE_BYTES; // 2 rows per warp
constexpr int CHUNK_COPY_LINE_LANES = WARP_SIZE / CHUNK_COPY_LINES_PER_WARP; // 16 lanes per row

constexpr int BLOCK_ROW_WARPS = 4;
constexpr int BLOCK_COL_WARPS = 2;

constexpr int WARP_ROW_TILES = 2;
constexpr int WARP_COL_TILES = 4;

constexpr int BLOCK_ROW_TILES = BLOCK_ROW_WARPS * WARP_ROW_TILES; // 8
constexpr int BLOCK_COL_TILES = BLOCK_COL_WARPS * WARP_COL_TILES; // 8

constexpr int SHMEM_STRIDE = WMMA_N * BLOCK_ROW_TILES; // 128
constexpr int SHMEM_OFFSET = WMMA_N * WARP_ROW_TILES; // 64

// load matrix sync requires 256-bit (32-byte) alignment
// hence the minimum possible padding is 16 half elements, i.e. 32 bytes (256+32=288%32=0)
// relevant when W is interpreted as column major (X@W.T)
constexpr int SKEW_HALF = 16;

constexpr size_t SHMEM_SZ = std::max(
    sizeof(half)*(BLOCK_COL_TILES*WMMA_M)*(CHUNK_K*WMMA_K+SKEW_HALF)*2,  // X+W tiles
    WMMA_M*(BLOCK_ROW_WARPS*WARP_ROW_TILES)*WMMA_N*(BLOCK_COL_WARPS*WARP_COL_TILES)*sizeof(float)  // out tiles
);

constexpr size_t SHMEM_SZ_T = std::max(
    sizeof(half)*(BLOCK_COL_TILES*WMMA_M)*(CHUNK_K*WMMA_K)*2,  // X+W tiles (w/ no SKEW)
    WMMA_M*(BLOCK_ROW_WARPS*WARP_ROW_TILES)*WMMA_N*(BLOCK_COL_WARPS*WARP_COL_TILES)*sizeof(float)  // out tiles
);

// helper to compute TILE_SIZE x TILE_SIZE portion of result (X@W.T)
static __device__ void compute_tile_wmma_f16(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out
) {
    extern __shared__ half shmem[][CHUNK_K*WMMA_K+SKEW_HALF];

    const unsigned int warpId = threadIdx.x/32;
    const unsigned int laneId = threadIdx.x%32;

    const size_t shmem_idx_b_off = WMMA_M*BLOCK_ROW_TILES;
    float* shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId/BLOCK_COL_WARPS)*WARP_ROW_TILES*WMMA_M*SHMEM_STRIDE + (warpId%BLOCK_COL_WARPS)*SHMEM_OFFSET;
    float* shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId*WMMA_M*SHMEM_STRIDE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> x_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> w_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {
        for (int j=0; j<WARP_COL_TILES; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const half* warp_ptr = (warpId<(WARPS_PER_BLOCK/2)) 
                            ? (x+WMMA_M*WMMA_K*(warpId%WARP_COL_TILES)*WARP_ROW_TILES) 
                            : (w+WMMA_N*WMMA_K*(warpId%WARP_COL_TILES)*WARP_ROW_TILES);

    int K_TILES = (K+WMMA_K-1)/WMMA_K;

    // iterating over global K dimension, CHUNK_K 16x16 tiles at a time
    for (int tile_k=0; tile_k<K_TILES; tile_k+=CHUNK_K) {
        
        // load X and W into shared memory
        // each warp loads 32 rows of X (warps 0-3) or W (warps 4-7)
        size_t shmem_idx = (warpId<(WARPS_PER_BLOCK/2)) 
                            ? (WMMA_M * (warpId%WARP_COL_TILES) * WARP_ROW_TILES)
                            : (WMMA_N * (warpId%WARP_COL_TILES) * WARP_ROW_TILES + shmem_idx_b_off);

        int4* lane_ptr = (int4*)(warp_ptr + (laneId/CHUNK_COPY_LINE_LANES)*K + tile_k*WMMA_K + (laneId%CHUNK_COPY_LINE_LANES));

        shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

        #pragma unroll
        for (int i=0; i<((WARP_SIZE/2)/CHUNK_COPY_LINES_PER_WARP)*WARP_ROW_TILES; ++i) {
            *((int4*)&shmem[shmem_idx][0] + (laneId%CHUNK_COPY_LINE_LANES)) = *lane_ptr;

            lane_ptr = (int4*)((half*)lane_ptr + CHUNK_COPY_LINES_PER_WARP*K);
            shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

        // process CHUNK_K 16x16 tiles, accumulate to acc
        #pragma unroll
        for (int k_step=0; k_step<CHUNK_K; ++k_step) {

            // iterate over grid w/in warp, compute tiles for acc
            #pragma unroll
            for (int i=0; i<WARP_ROW_TILES; ++i) {
                size_t  shmem_idx_x = (warpId/WARP_ROW_TILES)*WARP_ROW_TILES*WMMA_M + (i*WMMA_M);
                const half* tile_ptr = &shmem[shmem_idx_x][k_step*WMMA_K];

                wmma::load_matrix_sync(x_frag[i], tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);

                #pragma unroll
                for (int j=0; j<WARP_COL_TILES; ++j) {
                    if (i==0) {
                        size_t shmem_idx_w = shmem_idx_b_off + (warpId%2)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                        const half* tile_ptr = &shmem[shmem_idx_w][k_step*WMMA_K];

                        wmma::load_matrix_sync(w_frag[j], tile_ptr, WMMA_K*CHUNK_K+SKEW_HALF);
                    }

                    wmma::mma_sync(acc[i][j], x_frag[i], w_frag[j], acc[i][j]);
                }

            }

            __syncthreads();
        }
    }


    // acc fragments stored to shmem
    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {

        #pragma unroll
        for (int j=0; j<WARP_COL_TILES; ++j) {
            float* tile_ptr = shmem_warp_tile_ptr + i*WMMA_K*SHMEM_STRIDE + j*WMMA_N; 
            wmma::store_matrix_sync(tile_ptr, acc[i][j], SHMEM_STRIDE, wmma::row_major);
        }
    }

    __syncthreads();

    // stream tiles of out (in shmem) to gmem
    const size_t gmem_idx = (blockIdx.x*TILE_SIZE+warpId*WMMA_M)*N + blockIdx.y*TILE_SIZE;
    half* dst_gmem_warp_stream_ptr = &out[gmem_idx];

    #pragma unroll
    for (int i=0; i<WMMA_K; ++i) {
        *((int2*)(dst_gmem_warp_stream_ptr+i*N)+laneId) = *((int2*)(shmem_warp_stream_ptr+i*SHMEM_STRIDE) + laneId);
    }

}

// helper to compute TILE_SIZE x TILE_SIZE portion of result (X@W)
// no need for SKEW_HALF here since W is not interpreted as col-major
static __device__ void compute_tile_wmma_f16_T(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out
) {
    extern __shared__ half shmem[][CHUNK_K*WMMA_K];

    const unsigned int warpId = threadIdx.x/32;
    const unsigned int laneId = threadIdx.x%32;

    const size_t shmem_idx_b_off = WMMA_M*BLOCK_ROW_TILES;
    float* shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId/BLOCK_COL_WARPS)*WARP_ROW_TILES*WMMA_M*SHMEM_STRIDE + (warpId%BLOCK_COL_WARPS)*SHMEM_OFFSET;
    float* shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId*WMMA_M*SHMEM_STRIDE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> x_frag[WARP_ROW_TILES];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> w_frag[WARP_COL_TILES];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_ROW_TILES][WARP_COL_TILES];

    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {
        for (int j=0; j<WARP_COL_TILES; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const half* warp_ptr = (warpId<(WARPS_PER_BLOCK/2)) 
                            ? (x+WMMA_M*WMMA_K*(warpId%WARP_COL_TILES)*WARP_ROW_TILES) 
                            : (w+WMMA_N*WMMA_K*(warpId%WARP_COL_TILES)*WARP_ROW_TILES);

    int K_TILES = (K+WMMA_K-1)/WMMA_K;

    // iterating over global K dimension, CHUNK_K 16x16 tiles at a time

    for (int tile_k=0; tile_k<K_TILES; tile_k+=CHUNK_K) {
        
        // loading X and W into shared memory
        // each warp loads 32 rows of X (warps 0-3) or W (warps 4-7)
        size_t shmem_idx = (warpId<(WARPS_PER_BLOCK/2)) 
                            ? (WMMA_M * (warpId%WARP_COL_TILES) * WARP_ROW_TILES)
                            : (WMMA_N * (warpId%WARP_COL_TILES) * WARP_ROW_TILES + shmem_idx_b_off);

        int4* lane_ptr = (int4*)(warp_ptr + (laneId/CHUNK_COPY_LINE_LANES)*K + tile_k*WMMA_K + (laneId%CHUNK_COPY_LINE_LANES));

        shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

        // 16 iterations, 2 rows over each iteration covers 32 rows
        #pragma unroll
        for (int i=0; i<((WARP_SIZE/2)/CHUNK_COPY_LINES_PER_WARP)*WARP_ROW_TILES; ++i) {
            *((int4*)&shmem[shmem_idx][0] + (laneId%CHUNK_COPY_LINE_LANES)) = *lane_ptr;

            lane_ptr = (int4*)((half*)lane_ptr + CHUNK_COPY_LINES_PER_WARP*K);
            shmem_idx += CHUNK_COPY_LINES_PER_WARP;
        }

        __syncthreads();

        // process CHUNK_K 16x16 tiles, accumulate to acc
        #pragma unroll
        for (int k_step=0; k_step<CHUNK_K; ++k_step) {

            // iterate over grid w/in warp, compute tiles for acc
            #pragma unroll
            for (int i=0; i<WARP_ROW_TILES; ++i) {
                size_t  shmem_idx_x = (warpId/WARP_ROW_TILES)*WARP_ROW_TILES*WMMA_M + (i*WMMA_M);
                const half* tile_ptr = &shmem[shmem_idx_x][k_step*WMMA_K];

                wmma::load_matrix_sync(x_frag[i], tile_ptr, WMMA_K*CHUNK_K);

                #pragma unroll
                for (int j=0; j<WARP_COL_TILES; ++j) {
                    if (i==0) {
                        size_t shmem_idx_w = shmem_idx_b_off + (warpId%2)*(WARP_COL_TILES*WMMA_N) + j*WMMA_N;
                        const half* tile_ptr = &shmem[shmem_idx_w][k_step*WMMA_K];

                        wmma::load_matrix_sync(w_frag[j], tile_ptr, WMMA_K*CHUNK_K);
                    }

                    wmma::mma_sync(acc[i][j], x_frag[i], w_frag[j], acc[i][j]);
                }

            }

            __syncthreads();
        }
    }


    // acc fragments stored to shmem
    #pragma unroll
    for (int i=0; i<WARP_ROW_TILES; ++i) {

        #pragma unroll
        for (int j=0; j<WARP_COL_TILES; ++j) {
            float* tile_ptr = shmem_warp_tile_ptr + i*WMMA_K*SHMEM_STRIDE + j*WMMA_N; 
            wmma::store_matrix_sync(tile_ptr, acc[i][j], SHMEM_STRIDE, wmma::row_major);
        }
    }

    __syncthreads();

    // stream tiles of out (in shmem) to gmem
    const size_t gmem_idx = (blockIdx.x*TILE_SIZE+warpId*WMMA_M)*N + blockIdx.y*TILE_SIZE;
    half* dst_gmem_warp_stream_ptr = &out[gmem_idx];

    #pragma unroll
    for (int i=0; i<WMMA_K; ++i) {
        *((int2*)(dst_gmem_warp_stream_ptr+i*N)+laneId) = *((int2*)(shmem_warp_stream_ptr+i*SHMEM_STRIDE) + laneId);
    }
}

// helper to compute WMMA_M x WMMA_N portion of result
static __device__ void compute_tile_quant(
    int64_t qtype_int, int64_t qblock_size, int64_t qtype_size,
    int64_t M, int64_t K, int64_t N,
    int64_t x_stride, int64_t w_stride, int64_t out_stride,
    half (&w_shared)[WMMA_N][256],
    const half* __restrict__ x,
    const uint8_t* __restrict__ w,
    half* __restrict__ out
) {
    int warpIdInBlock = threadIdx.x/32;
    // int warpsPerBlock = blockDim.x / 32;

    int localRow = warpIdInBlock*WMMA_M;
    // int globalRowM = blockIdx.x*warpsPerBlock*WMMA_M + localRow;
    // int globalColN = blockIdx.y*WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

    wmma::fill_fragment(acc_frag, __float2half(0.0f));

    for (int k=0; k<K; k+=256) {

        // dequantize 16x256 tile of w into w_shared
        for (int l=threadIdx.x; l<WMMA_N*256; l+=blockDim.x) {
            int r = l / 256;
            int c = l % 256;
            const uint8_t* w_block = w + r*w_stride + ((k+c)/qblock_size)*qtype_size;
            dequant_block(qtype_int, (k+c)%qblock_size, &w_shared[r][c], w_block);
        }

        __syncthreads();

        #pragma unroll
        for (int kk=0; kk<256; kk+=WMMA_K) {
            wmma::load_matrix_sync(a_frag, x + localRow*x_stride + k+kk, x_stride);
            wmma::load_matrix_sync(b_frag, &w_shared[0][kk], 256);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    wmma::store_matrix_sync(out + localRow*out_stride, acc_frag, out_stride, wmma::mem_row_major);
}

__global__ void matmul_wmma_f16_cuda_impl(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out
) {
    int M_TILES = (M+TILE_SIZE-1)/TILE_SIZE;
    int N_TILES = (N+TILE_SIZE-1)/TILE_SIZE;

    for (unsigned int block_pos = blockIdx.x; block_pos < M_TILES * N_TILES; block_pos += gridDim.x) {

        unsigned int tile_i = block_pos/N_TILES;
        unsigned int tile_j = block_pos%N_TILES;

        const half* x_row = x + tile_i*TILE_SIZE*K;
        const half* w_row = w + tile_j*TILE_SIZE*K;
        half* out_tile = out + tile_i*TILE_SIZE*N + tile_j*TILE_SIZE;
        
        compute_tile_wmma_f16(
            M, K, N,
            x_row, w_row, out_tile
        );
    }
}

__global__ void matmul_wmma_f16_T_cuda_impl(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out
) {

    int M_TILES = (M+TILE_SIZE-1)/TILE_SIZE;
    int N_TILES = (N+TILE_SIZE-1)/TILE_SIZE;

    for (unsigned int block_pos = blockIdx.x; block_pos < M_TILES * N_TILES; block_pos += gridDim.x) {

        unsigned int tile_i = block_pos/N_TILES;
        unsigned int tile_j = block_pos%N_TILES;

        const half* x_row = x + tile_i*TILE_SIZE*K;
        const half* w_row = w + tile_j*TILE_SIZE*K;
        half* out_tile = out + tile_i*TILE_SIZE*N + tile_j*TILE_SIZE;
        
        compute_tile_wmma_f16_T(
            M, K, N,
            x_row, w_row, out_tile
        );
    }
}

__global__ void matmul_quant_cuda_impl(
    int64_t qtype_int, int64_t qblock_size, int64_t qtype_size,
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    const uint8_t* __restrict__ w,
    half* __restrict__ out
) {
    __shared__ half w_shared[WMMA_N][256];

    int warpsPerBlock = blockDim.x / 32;
    int rowM = blockIdx.x*warpsPerBlock*WMMA_M;
    int rowN = blockIdx.y*WMMA_N;

    const half* x_row = x + rowM*K;
    const uint8_t* w_row = w + rowN*(K/qblock_size)*qtype_size;
    half* out_tile = out + rowM*N + rowN;

    compute_tile_quant(
        qtype_int, qblock_size, qtype_size,
        M, K, N,
        K, K, N,
        w_shared,
        x_row, w_row, out_tile
    );
}

void matmul_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x, // [B,L,in_dim]
    const at::Tensor& w, // [out_dim, in_dim (possibly bytes)]
    at::Tensor& out // [B,L,out_dim]
) {
    TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.device() == w.device() && x.device() == out.device());

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kByte || w.dtype() == at::kHalf);

    TORCH_CHECK(x.dim() == 3);
    TORCH_CHECK(out.dim() == 3);
    TORCH_CHECK(w.dim() == 2);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(x.size(0) == out.size(0) && x.size(1) == out.size(1));
    TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size) || 
                x.size(-1) == w.size(-1) || x.size(-1) == w.size(0));
    TORCH_CHECK(w.size(0) == out.size(-1) || w.size(1) == out.size(-1));

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    
    int dev = x.device().index();
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.major < 7) {
        TORCH_CHECK(false, "SM 7.0 or higher required to use tensor cores");
    } else if (deviceProp.sharedMemPerMultiprocessor < SHMEM_SZ) {
        TORCH_CHECK(false, "Not enough shared memory for performant kernel");
    }

    dim3 grid(deviceProp.multiProcessorCount);
    dim3 block(THREADS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (qtype == GGMLQuantizationType::F16) {
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
        if (x.size(-1) == w.size(-1)) {
            int64_t N = w.size(0);
            checkCudaErrors(
                cudaFuncSetAttribute(matmul_wmma_f16_cuda_impl, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ)
            );
            matmul_wmma_f16_cuda_impl<<<grid, THREADS_PER_BLOCK, SHMEM_SZ, stream>>>(
                M, K, N, 
                x_ptr, w_ptr, out_ptr
            );
        } else if (x.size(-1) == w.size(0)) {
            int64_t N = w.size(1);
            checkCudaErrors(
                cudaFuncSetAttribute(matmul_wmma_f16_cuda_impl, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ)
            );
            matmul_wmma_f16_T_cuda_impl<<<grid, THREADS_PER_BLOCK, SHMEM_SZ, stream>>>(
                M, K, N,
                x_ptr, w_ptr, out_ptr
            );
        }
    } else { // ignore this case for now, still need to fix
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
        constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
        int64_t N = w.size(0);
        dim3 grid((M+ROWS_M-1)/ROWS_M, (N+WMMA_N-1)/WMMA_N);
        dim3 block(BLOCK_SIZE);

        matmul_quant_cuda_impl<<<grid, block, 0, stream>>>(
            qtype_int, qblock_size, qtype_size, 
            M, K, N, 
            x_ptr, w_ptr, out_ptr
        );
    }

    checkKernelErrors(cudaGetLastError());
}

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

void embed_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& token_ids, // [B,L]
    const at::Tensor& w, // [vocab_size, hidden_dim (possibly bytes)]
    at::Tensor& x // [B,L,D]
) {
    TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(token_ids.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(token_ids.dtype() == at::kLong);
    TORCH_CHECK(w.dtype() == at::kByte || w.dtype() == at::kHalf);

    TORCH_CHECK(x.dim() == 3);
    TORCH_CHECK(token_ids.dim() == 2);
    TORCH_CHECK(w.dim() == 2);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(token_ids.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(x.size(0) == token_ids.size(0) && x.size(1) == token_ids.size(1));

    const int64_t* token_ids_ptr = token_ids.data_ptr<int64_t>();
    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&device_pop, dev));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t hidden_dim = x.size(2);

    if (qtype == GGMLQuantizationType::F16) {
        TORCH_CHECK(w.size(-1) == x.size(-1));
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
        dim3 grid(B, L);
        embed_f16_cuda_impl<<<grid, 1024, 0, stream>>>(
            hidden_dim, token_ids_ptr, w_ptr, x_ptr
        );
    } else {
        TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size));
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
        int64_t n_qblocks_per_row = hidden_dim / qblock_size;
        int64_t b = w.size(-1); // bytes per row
        int64_t k = hidden_dim; // dequant elems per row
        dim3 grid(B, L, n_qblocks_per_row);
        embed_quant_cuda_impl<<<grid, qblock_size, 0, stream>>>(
            qtype_int, qblock_size, qtype_size, b, k, token_ids_ptr, w_ptr, x_ptr
        );
    }
}

// helper to compute WMMA_M x WMMA_N portion of result for QKV projections
// reduces loads from global mem of input x by x3 factor
template <int WARPS_PER_BLOCK>
static __device__ void compute_qkv_tile_f16(
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    half (&x_shared) [WARPS_PER_BLOCK*WMMA_M][WMMA_K],
    half (&wq_shared) [WMMA_N][WMMA_K],
    half (&wk_shared) [WMMA_N][WMMA_K],
    half (&wv_shared) [WMMA_N][WMMA_K],
    const half* __restrict__ x,
    const half* __restrict__ wq,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {
    int warpIdInBlock = threadIdx.x / 32;
    // int warpsPerBlock = blockDim.x / 32;

    int localRow = warpIdInBlock*WMMA_M;

    // int globalRowM = blockIdx.x*warpsPerBlock*WMMA_M + localRow;
    int globalColN = blockIdx.y*WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> q_acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> k_acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> v_acc_frag;

    wmma::fill_fragment(q_acc_frag, __float2half(0.0f));
    if (globalColN < N_KV) {
        wmma::fill_fragment(k_acc_frag, __float2half(0.0f));
        wmma::fill_fragment(v_acc_frag, __float2half(0.0f));
    }
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;    

    for (int k=0; k<K; k+=WMMA_K) {

        for (int l=threadIdx.x; l<WARPS_PER_BLOCK*WMMA_M*WMMA_K; l+=blockDim.x) {
            int r = l/WMMA_K;
            int c = l%WMMA_K;            
            x_shared[r][c] = x[r*K+k+c];
        }

        for (int l=threadIdx.x; l<WMMA_N*WMMA_K; l+=blockDim.x) {
            int r = l/WMMA_K;
            int c = l%WMMA_K;
            wq_shared[r][c] = wq[r*K+k+c];
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, &x_shared[localRow][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &wq_shared[0][0], WMMA_K);
        wmma::mma_sync(q_acc_frag, a_frag, b_frag, q_acc_frag);

        if (globalColN < N_KV) {
            for (int l=threadIdx.x; l<WMMA_N*WMMA_K; l+=blockDim.x) {
                int r = l/WMMA_K;
                int c = l%WMMA_K;
                wk_shared[r][c] = wk[r*K+k+c];
            }
        }

        __syncthreads();

        if (globalColN < N_KV) {
            wmma::load_matrix_sync(b_frag, &wk_shared[0][0], WMMA_K);
            wmma::mma_sync(k_acc_frag, a_frag, b_frag, k_acc_frag);

            for (int l=threadIdx.x; l<WMMA_N*WMMA_K; l+=blockDim.x) {
                int r = l/WMMA_K;
                int c = l%WMMA_K;
                wv_shared[r][c] = wv[r*K+k+c];
            }
        }
        
        __syncthreads();

        if (globalColN < N_KV) {
            wmma::load_matrix_sync(b_frag, &wv_shared[0][0], WMMA_K);
            wmma::mma_sync(v_acc_frag, a_frag, b_frag, v_acc_frag);
        }

        __syncthreads();
    }

    wmma::store_matrix_sync(q_out + localRow*N_Q, q_acc_frag, N_Q, wmma::mem_row_major);

    if (globalColN < N_KV) {
        wmma::store_matrix_sync(k_out + localRow*N_KV, k_acc_frag, N_KV, wmma::mem_row_major);
        wmma::store_matrix_sync(v_out + localRow*N_KV, v_acc_frag, N_KV, wmma::mem_row_major);
    }
}

// helper to compute TILE_M x TILE_N portion of result for QKV projections
// reduces loads from global mem of input x by x3 factor
template <int WARPS_PER_BLOCK>
static __device__ void compute_qkv_tile_quant(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size,  int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    half* x_shared,   // [WARPS_PER_BLOCK*WMMA_M][256]
    half* wq_shared,  // [WMMA_N][256]
    half* wk_shared,  // [WMMA_N][256]
    half* wv_shared,  // [WMMA_N][256]
    const half* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {
    int warpIdInBlock = threadIdx.x / 32;
    // int warpsPerBlock = blockDim.x / 32;

    int localRow = warpIdInBlock*WMMA_M;

    // int globalRowM = blockIdx.x*warpsPerBlock*WMMA_M + localRow;
    int globalColN = blockIdx.y*WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> q_acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> k_acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> v_acc_frag;

    wmma::fill_fragment(q_acc_frag, __float2half(0.0f));
    if (globalColN < N_KV) {
        wmma::fill_fragment(k_acc_frag, __float2half(0.0f));
        wmma::fill_fragment(v_acc_frag, __float2half(0.0f));
    }
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;  

    // in this approach we try to hide dequant latency behind compute
    // hopefully this helps
    for (int k=0; k<K; k+=256) {

        for (int l=threadIdx.x; l<WARPS_PER_BLOCK*WMMA_M*256; l+=blockDim.x) {
            int r = l / 256;
            int c = l % 256;            
            x_shared[r*256+c] = x[r*K+k+c];
        }

        // dequantize 16x256 tile of wq into wq_shared
        for (int l=threadIdx.x; l<WMMA_N*256; l+=blockDim.x) {
            int r = l / 256;
            int c = l % 256;
            const uint8_t* wq_block = wq + r*(K/q_qblock_size)*q_qtype_size + ((k+c)/q_qblock_size)*q_qtype_size;
            dequant_block(q_qtype_int, (k+c)%q_qblock_size, &wq_shared[r*256+c], wq_block);
        }

        __syncthreads();

        #pragma unroll
        for (int kk=0; kk<256; kk+=WMMA_K) {
            wmma::load_matrix_sync(a_frag, x_shared+localRow*256+kk, 256);
            wmma::load_matrix_sync(b_frag, wq_shared+kk, 256);
            wmma::mma_sync(q_acc_frag, a_frag, b_frag, q_acc_frag);
        }
        
        if (globalColN < N_KV) {
            // dequantize 16x256 tile of wk into wk_shared
            for (int l=threadIdx.x; l<WMMA_N*256; l+=blockDim.x) {
                int r = l / 256;
                int c = l % 256;
                const uint8_t* wk_block = wk + r*(K/k_qblock_size)*k_qtype_size + ((k+c)/k_qblock_size)*k_qtype_size;
                dequant_block(k_qtype_int, (k+c)%k_qblock_size, &wk_shared[r*256+c], wk_block);
            }
        }
        
        __syncthreads();

        if (globalColN < N_KV) {
            #pragma unroll
            for (int kk=0; kk<256; kk+=WMMA_K) {
                wmma::load_matrix_sync(a_frag, x_shared+localRow*256+kk, 256);
                wmma::load_matrix_sync(b_frag, wk_shared+kk, 256);
                wmma::mma_sync(k_acc_frag, a_frag, b_frag, k_acc_frag);
            }
        }
        
        if (globalColN < N_KV) {
            // dequantize 16x256 tile of wv into wv_shared
            for (int l=threadIdx.x; l<WMMA_N*256; l+=blockDim.x) {
                int r = l / 256;
                int c = l % 256;
                const uint8_t* wv_block = wv + r*(K/v_qblock_size)*v_qtype_size + ((k+c)/v_qblock_size)*v_qtype_size;
                dequant_block(v_qtype_int, (k+c)%v_qblock_size, &wv_shared[r*256+c], wv_block);
            }
        }
        

        __syncthreads();

        if (globalColN < N_KV) {
            #pragma unroll
            for (int kk=0; kk<256; kk+=WMMA_K) {
                wmma::load_matrix_sync(a_frag, x_shared+localRow*256+kk, 256);
                wmma::load_matrix_sync(b_frag, wv_shared+kk, 256);
                wmma::mma_sync(v_acc_frag, a_frag, b_frag, v_acc_frag);
            }
        }
        
        __syncthreads();
    }

    wmma::store_matrix_sync(q_out + localRow*N_Q, q_acc_frag, N_Q, wmma::mem_row_major);

    if (globalColN < N_KV) {
        wmma::store_matrix_sync(k_out + localRow*N_KV, k_acc_frag, N_KV, wmma::mem_row_major);
        wmma::store_matrix_sync(v_out + localRow*N_KV, v_acc_frag, N_KV, wmma::mem_row_major);
    }
}

template <int WARPS_PER_BLOCK>
__global__ void qkv_f16_cuda_impl(
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    const half* __restrict__ x,
    const half* __restrict__ wq,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {

    __shared__ half x_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_K];
    __shared__ half wq_shared[WMMA_N][WMMA_K];
    __shared__ half wk_shared[WMMA_N][WMMA_K];
    __shared__ half wv_shared[WMMA_N][WMMA_K];

    int warpsPerBlock = blockDim.x / 32;
    int rowM = blockIdx.x*warpsPerBlock*WMMA_M;
    int rowN = blockIdx.y*WMMA_N;

    const half* x_row = x + rowM*K;

    const half* wq_row = wq + rowN*K;
    const half* wk_row = wk + rowN*K;
    const half* wv_row = wv + rowN*K;

    half* q_out_tile = q_out + rowM*N_Q + rowN;
    half* k_out_tile = k_out + rowM*N_KV + rowN;
    half* v_out_tile = v_out + rowM*N_KV + rowN;
    
    compute_qkv_tile_f16<WARPS_PER_BLOCK>(
        M, K, N_Q, N_KV,
        x_shared, wq_shared, wk_shared, wv_shared,
        x_row, wq_row, wk_row, wv_row,
        q_out_tile, k_out_tile, v_out_tile
    );
}

template <int WARPS_PER_BLOCK>
__global__ void qkv_quant_cuda_impl(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size,  int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    const half* __restrict__ x,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out
) {

    // smem requirements exceed 48KB static limit on V100
    extern __shared__ char smem[];
    half* x_shared = (half*)smem;
    half* wq_shared = x_shared + WARPS_PER_BLOCK*WMMA_M*256;
    half* wk_shared = wq_shared + WMMA_N*256;
    half* wv_shared = wk_shared + WMMA_N*256;

    int warpsPerBlock = blockDim.x / 32;
    int rowM = blockIdx.x*warpsPerBlock*WMMA_M;
    int rowN = blockIdx.y*WMMA_N;

    const half* x_row = x + rowM*K;

    const uint8_t* wq_row = wq + rowN*(K/q_qblock_size)*q_qtype_size;
    const uint8_t* wk_row = wk + rowN*(K/k_qblock_size)*k_qtype_size;
    const uint8_t* wv_row = wv + rowN*(K/v_qblock_size)*v_qtype_size;

    half* q_out_tile = q_out + rowM*N_Q + rowN;
    half* k_out_tile = k_out + rowM*N_KV + rowN;
    half* v_out_tile = v_out + rowM*N_KV + rowN;
    
    compute_qkv_tile_quant<WARPS_PER_BLOCK>(
        q_qtype_int, q_qblock_size, q_qtype_size,
        k_qtype_int, k_qblock_size, k_qtype_size,
        v_qtype_int, v_qblock_size, v_qtype_size,
        M, K, N_Q, N_KV,
        x_shared, wq_shared, wk_shared, wv_shared,
        x_row, wq_row, wk_row, wv_row,
        q_out_tile, k_out_tile, v_out_tile
    );
}

void qkv_cuda(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size, int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    const at::Tensor& x, // [B, L, hidden_dim]
    const at::Tensor& wq, // [hidden_dim, q_dim (possibly in bytes)]
    const at::Tensor& wk, // [hidden_dim, kv_dim (possibly in bytes)]
    const at::Tensor& wv, // [hidden_dim, kv_dim (possibly in bytes)]
    at::Tensor& q_out, // [B, L, q_dim]
    at::Tensor& k_out, // [B, L, kv_dim]
    at::Tensor& v_out // [B, L, kv_dim]
) {
    // validation
    TORCH_CHECK(is_valid_qtype(q_qtype_int) && is_valid_qtype(k_qtype_int) && is_valid_qtype(v_qtype_int));
    auto q_qtype = static_cast<GGMLQuantizationType>(q_qtype_int);
    auto k_qtype = static_cast<GGMLQuantizationType>(k_qtype_int);
    auto v_qtype = static_cast<GGMLQuantizationType>(v_qtype_int);

    TORCH_INTERNAL_ASSERT(
        (x.device().type() == at::DeviceType::CUDA) &&
        (q_out.device().type() == at::DeviceType::CUDA) &&
        (k_out.device().type() == at::DeviceType::CUDA) && 
        (v_out.device().type() == at::DeviceType::CUDA) &&
        (wq.device().type() == at::DeviceType::CUDA) &&
        (wk.device().type() == at::DeviceType::CUDA) &&
        (wv.device().type() == at::DeviceType::CUDA)
    );

    TORCH_CHECK(
        x.is_contiguous() &&
        q_out.is_contiguous() && k_out.is_contiguous() && v_out.is_contiguous() &&
        wq.is_contiguous() && wk.is_contiguous() && wv.is_contiguous()
    );

    //dtypes
    TORCH_CHECK(
        (x.dtype() == at::kHalf) && 
        (q_out.dtype() == at::kHalf) && (k_out.dtype() == at::kHalf) && (v_out.dtype() == at::kHalf) &&
        (wq.dtype() == at::kByte || wq.dtype() == at::kHalf) && 
        (wk.dtype() == at::kByte || wk.dtype() == at::kHalf) && 
        (wv.dtype() == at::kByte || wv.dtype() == at::kHalf)
    );

    // dim
    TORCH_CHECK(
        (x.dim() == 3) && 
        (q_out.dim() == 3) && (k_out.dim() == 3) && (v_out.dim() == 3) &&
        (wq.dim() == 2) && (wk.dim() == 2) && (wv.dim() == 2)
    );

    //size
    TORCH_CHECK(
        (wq.size(0) == q_out.size(-1)) &&
        (wk.size(0) == k_out.size(-1)) &&
        (wv.size(0) == v_out.size(-1)) &&
        (wk.size(-1) == wv.size(-1))
    );

    TORCH_CHECK(
        (x.numel() / x.size(-1) == q_out.numel() / q_out.size(-1)) &&
        (x.numel() / x.size(-1) == k_out.numel() / k_out.size(-1)) &&
        (x.numel() / x.size(-1) == v_out.numel() / v_out.size(-1)) &&
        (k_out.size(-1) == v_out.size(-1))
    );

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    
    half* q_out_ptr = reinterpret_cast<half*>(q_out.data_ptr<at::Half>());
    half* k_out_ptr = reinterpret_cast<half*>(k_out.data_ptr<at::Half>());
    half* v_out_ptr = reinterpret_cast<half*>(v_out.data_ptr<at::Half>());

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t N_Q = wq.size(0);
    int64_t N_KV = wk.size(0);
    int64_t N = std::max(N_Q, N_KV);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // assume: either all of QKV are quantized (not F16) or dequantized (all of QKV are FP16)
    if (q_qtype == GGMLQuantizationType::F16 && k_qtype == GGMLQuantizationType::F16 && v_qtype == GGMLQuantizationType::F16) {

        TORCH_CHECK(
            (wq.size(1) == x.size(-1)) &&
            (wk.size(1) == x.size(-1)) &&
            (wv.size(1) == x.size(-1))
        );

        const half* wq_ptr = reinterpret_cast<const half*>(wq.data_ptr<at::Half>());
        const half* wk_ptr = reinterpret_cast<const half*>(wk.data_ptr<at::Half>());
        const half* wv_ptr = reinterpret_cast<const half*>(wv.data_ptr<at::Half>());

        // will probably need to adjust these
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
        constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
        dim3 grid((M+ROWS_M-1)/ROWS_M, (N+WMMA_N-1)/WMMA_N);
        dim3 block(BLOCK_SIZE);

        qkv_f16_cuda_impl<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
            M, K, N_Q, N_KV,
            x_ptr, wq_ptr, wk_ptr, wv_ptr,
            q_out_ptr, k_out_ptr, v_out_ptr
        );
    } else {

        TORCH_CHECK(
            (wq.size(1) == (x.size(-1) / q_qblock_size) * q_qtype_size) &&
            (wk.size(1) == (x.size(-1) / k_qblock_size) * k_qtype_size) &&
            (wv.size(1) == (x.size(-1) / v_qblock_size) * v_qtype_size)
        );

        const uint8_t* wq_ptr = wq.data_ptr<uint8_t>();
        const uint8_t* wk_ptr = wk.data_ptr<uint8_t>();
        const uint8_t* wv_ptr = wv.data_ptr<uint8_t>();

        // will probably need to adjust these
        constexpr int WARPS_PER_BLOCK = 4;
        constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
        constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
        dim3 grid((M+ROWS_M-1)/ROWS_M, (N+WMMA_N-1)/WMMA_N);
        dim3 block(BLOCK_SIZE);

        size_t smem_size = (WARPS_PER_BLOCK*WMMA_M*256 + 3*WMMA_N*256)*sizeof(half);

        qkv_quant_cuda_impl<WARPS_PER_BLOCK><<<grid, block, smem_size, stream>>>(
            q_qtype_int, k_qtype_int, v_qtype_int,
            q_qblock_size, k_qblock_size, v_qblock_size,
            q_qtype_size, k_qtype_size, v_qtype_size,
            M, K, N_Q, N_KV,
            x_ptr, wq_ptr, wk_ptr, wv_ptr,
            q_out_ptr, k_out_ptr, v_out_ptr
        );
    }
}

// helper (TODO: complete me once you know exactly where this will be used)
template <int WARPS_PER_BLOCK>
static __device__ void update_dm(
    float* __restrict__ ds,
    float* __restrict__ ms,
    const half (&tile)[WARPS_PER_BLOCK*WMMA_M][WMMA_N]
) {
    int warpId = threadIdx.x/32;
    int lane = threadIdx.x%32;
    int rowInWarp = lane/2;
    int thrInRow = lane%2;
    int globalRow = warpId*WMMA_M+rowInWarp;

    // row max
    float localMax = -INFINITY;
    for (int c=thrInRow; c<WMMA_N; c+=2) {
        localMax = fmaxf(localMax, __half2float(tile[globalRow][c]));
    }
    float partnerMax = __shfl_xor_sync(0xffffffff, localMax, 1);
    float row_m = fmaxf(localMax, partnerMax);

    // row sum
    float localSum = 0.0f;
    for (int c=thrInRow; c<WMMA_N; c+=2) {
        localSum += expf(__half2float(tile[globalRow][c])-row_m);
    }
    float partnerSum = __shfl_xor_sync(0xffffffff, localSum, 1);
    float row_d = localSum + partnerSum;

    if (thrInRow == 0) {
        float m_old = ms[globalRow];
        float d_old = ds[globalRow];
        float m_new = fmaxf(m_old, row_m);
        float d_new = d_old*expf(m_old-m_new) + row_d*expf(row_m-m_new);
        ms[globalRow] = m_new;
        ds[globalRow] = d_new;
    }
}

// FlashAttention-2 forward pass impl.
// Refer to: https://arxiv.org/abs/2307.08691
constexpr float HALF_MIN = -65504.0f;
struct QKVStrides {
    int64_t q0, q1, q2;
    int64_t kv0, kv1, kv2;
};
template <int WARPS_PER_BLOCK, int HEAD_DIM>
__global__ void flash_attn_cuda_impl(
    QKVStrides strides,
    int64_t L, int64_t n_kv_heads,
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const uint8_t* __restrict__ mask,
    half* __restrict__ out
) {
    
    __shared__ half l_shared[WARPS_PER_BLOCK*WMMA_M][WMMA_N];
    __shared__ half k_shared[WMMA_N][WMMA_K];
    __shared__ half pv_shared[WARPS_PER_BLOCK*WMMA_M][HEAD_DIM];
    __shared__ half out_shared[WARPS_PER_BLOCK*WMMA_M][HEAD_DIM];
    
    __shared__ float ds[WARPS_PER_BLOCK*WMMA_M];
    __shared__ float ms[WARPS_PER_BLOCK*WMMA_M];

    __shared__ float ms_old[WARPS_PER_BLOCK*WMMA_M];
    __shared__ float ds_old[WARPS_PER_BLOCK*WMMA_M];

    int64_t n_heads = gridDim.y;
    int64_t batch_idx = blockIdx.x;
    int64_t q_head_idx = blockIdx.y;
    int64_t kv_head_idx = q_head_idx / (n_heads/n_kv_heads);
    int64_t seq_tile = blockIdx.z;

    const half* q_row = q + batch_idx*strides.q0 + q_head_idx*strides.q1 + seq_tile*WARPS_PER_BLOCK*WMMA_M*strides.q2;
    const half* k_head = k + batch_idx*strides.kv0 + kv_head_idx*strides.kv1;
    const half* v_head = v + batch_idx*strides.kv0 + kv_head_idx*strides.kv1;
    
    half* out_row = out + batch_idx*n_heads*L*HEAD_DIM + q_head_idx*L*HEAD_DIM + seq_tile*WARPS_PER_BLOCK*WMMA_M*HEAD_DIM;
    const uint8_t* mask_row = mask + batch_idx*L*L + seq_tile*WARPS_PER_BLOCK*WMMA_M*L; // [B, 1, L, L]

    for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M; idx+=blockDim.x) {
        ms[idx] = HALF_MIN;
        ds[idx] = 0.0f;
    }

    for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
        out_shared[idx/HEAD_DIM][idx%HEAD_DIM] = __float2half(0.0f);
    }

    __syncthreads();

    for (int n_tile=0; n_tile<L; n_tile+=WMMA_N) {
        const half* k_row = k_head + n_tile*strides.kv2;
        const half* v_row = v_head + n_tile*strides.kv2;
        const uint8_t* mask_tile = mask_row + n_tile;

        compute_tile_wmma_f16(
            L, HEAD_DIM, L,
            strides.q2, strides.kv2, WMMA_N,
            k_shared,
            q_row, k_row, (half*)l_shared
        );

        // apply mask and scale before computing ms and ds
        for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*WMMA_N; idx+=blockDim.x) {
            int r = idx / WMMA_N;
            int c = idx % WMMA_N;

            l_shared[r][c] = mask_tile[r*L+c] ? __float2half(__half2float(l_shared[r][c]) / sqrtf((float)HEAD_DIM)) : __float2half(HALF_MIN);
        }

        for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M; idx+=blockDim.x) {
            ms_old[idx] = ms[idx];
            ds_old[idx] = ds[idx];
        }

        __syncthreads();

        update_dm<WARPS_PER_BLOCK>(ds, ms, l_shared);

        __syncthreads();

        for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*WMMA_N; idx+=blockDim.x) {
            int r = idx/WMMA_N;
            int c = idx%WMMA_N;
            l_shared[r][c] = __float2half(expf(__half2float(l_shared[r][c])-ms[r]));
        }

        for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
            int r = idx/HEAD_DIM;
            int c = idx%HEAD_DIM;

            out_shared[r][c] = __float2half(__half2float(out_shared[r][c])*expf(ms_old[r]-ms[r]));
        }

        __syncthreads();


        #pragma unroll
        for (int n=0; n<HEAD_DIM; n+=WMMA_N) {
            compute_tile_wmma_f16_T(
                WARPS_PER_BLOCK*WMMA_M, WMMA_N, HEAD_DIM,
                WMMA_N, strides.kv2, HEAD_DIM,
                (half*)l_shared, v_row + n,  &pv_shared[0][n]
            );
        }

        __syncthreads();

        for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
            int r = idx/HEAD_DIM;
            int c = idx%HEAD_DIM;
            out_shared[r][c] = __float2half(__half2float(out_shared[r][c]) + __half2float(pv_shared[r][c]));
        }
    }

    for (int idx=threadIdx.x; idx<WARPS_PER_BLOCK*WMMA_M*HEAD_DIM; idx+=blockDim.x) {
        int r = idx/HEAD_DIM;
        int c = idx%HEAD_DIM;
        out_row[r*HEAD_DIM+c] = (ds[r] > 0.0f) ? __float2half(__half2float(out_shared[r][c]) / ds[r]) : __float2half(0.0f);
    }
}

void flash_attn_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& q, // [B, n_heads, L, HEAD_DIM]
    const at::Tensor& k, // [B, n_kv_heads, L, HEAD_DIM]
    const at::Tensor& v,  // [B, n_kv_heads, L, HEAD_DIM]
    at::Tensor& mask, // [B, 1, L, L]
    at::Tensor& out     // [B, n_heads, L, HEAD_DIM]
) {
    // validation
    TORCH_INTERNAL_ASSERT(
        (q.device().type() == at::DeviceType::CUDA) &&
        (k.device().type() == at::DeviceType::CUDA) && 
        (v.device().type() == at::DeviceType::CUDA) &&
        (mask.device().type() == at::DeviceType::CUDA) &&
        (out.device().type() == at::DeviceType::CUDA)
    );

    // contiguity in the last dimension
    // except mask, is fully contiguous
    TORCH_CHECK(
        (q.stride(-1) == 1) &&
        (k.stride(-1) == 1) &&
        (v.stride(-1) == 1) &&
        (out.stride(-1) == 1) &&
        (k.stride(0) == v.stride(0)) &&
        (k.stride(1) == v.stride(1)) &&
        (k.stride(2) == v.stride(2)) &&
        (mask.is_contiguous())
    );

    // dtypes
    TORCH_CHECK(
        (q.dtype() == at::kHalf) &&
        (k.dtype() == at::kHalf) &&
        (v.dtype() == at::kHalf) &&
        (out.dtype() == at::kHalf) && 
        (mask.dtype() == at::kBool)
    );

    // dims
    TORCH_CHECK(
        (q.dim() == 4) &&
        (k.dim() == 4) &&
        (v.dim() == 4) &&
        (out.dim() == 4) &&
        (mask.dim() == 4)
    );

    // checks btwn different dims
    TORCH_CHECK(q.sizes().equals(out.sizes()));
    TORCH_CHECK(k.sizes().equals(v.sizes()));
    TORCH_CHECK(
        (q.size(0) == k.size(0)) && (q.size(0) == mask.size(0)) &&
        (q.size(2) == k.size(2)) && (q.size(2) == mask.size(2)) && (q.size(2) == mask.size(3)) &&
        (q.size(3) == k.size(3))
    );

    QKVStrides strides = {
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2)
    };

    const half* q_ptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half* k_ptr = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    const half* v_ptr = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const uint8_t* mask_ptr = reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());

    int64_t B = q.size(0);
    int64_t n_heads = q.size(1);
    int64_t n_kv_heads = k.size(1);
    int64_t head_dim = q.size(-1);
    TORCH_CHECK(head_dim == 128);
    int64_t L = q.size(2);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int WARPS_PER_BLOCK = 4; // TODO: tune this or adjust as needed
    constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
    constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
    constexpr int HEAD_DIM = 128;
    dim3 grid(B, n_heads, (L+ROWS_M-1)/ROWS_M);
    dim3 block(BLOCK_SIZE);

    flash_attn_cuda_impl<WARPS_PER_BLOCK, HEAD_DIM><<<grid, block, 0, stream>>>(
        strides,
        L, n_kv_heads,
        q_ptr, k_ptr, v_ptr,
        mask_ptr,
        out_ptr
    );
}

template <int THR_PER_ROW>
static __device__ float row_reduce_max(float* vs, float v, int row_in_tile, int thr_in_row) {
    constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;
    constexpr int WARP_WIDTH = WARPS_PER_ROW > 1 ? 32 : THR_PER_ROW;

    v = warp_reduce_max<WARP_WIDTH>(v);

    int warp_in_row = thr_in_row / 32;
    if (thr_in_row % 32 == 0) vs[row_in_tile*WARPS_PER_ROW+warp_in_row]=v;
    __syncthreads();
    
    if (WARPS_PER_ROW > 1 && warp_in_row == 0 && thr_in_row < WARPS_PER_ROW) {
        v = vs[row_in_tile*WARPS_PER_ROW + thr_in_row];
        v = warp_reduce_max<WARPS_PER_ROW>(v);
    }

    if (thr_in_row == 0) vs[row_in_tile*WARPS_PER_ROW] = v;
    __syncthreads();
    return vs[row_in_tile*WARPS_PER_ROW];
}

template <int THR_PER_ROW>
static __device__ float row_reduce_sum(float* vs, float v, int row_in_tile, int thr_in_row) {
    constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;
    constexpr int WARP_WIDTH = WARPS_PER_ROW > 1 ? 32 : THR_PER_ROW;

    v = warp_reduce_sum<WARP_WIDTH>(v);

    int warp_in_row = thr_in_row / 32;
    if (thr_in_row % 32 == 0) vs[row_in_tile*WARPS_PER_ROW+warp_in_row]=v;
    __syncthreads();
    
    if (WARPS_PER_ROW > 1 && warp_in_row == 0 && thr_in_row < WARPS_PER_ROW) {
        v = vs[row_in_tile*WARPS_PER_ROW + thr_in_row];
        v = warp_reduce_sum<WARPS_PER_ROW>(v);
    }

    if (thr_in_row == 0) vs[row_in_tile*WARPS_PER_ROW] = v;
    __syncthreads();
    return vs[row_in_tile*WARPS_PER_ROW];
}

template <int TILE_M, int N_EXPS, int TOP_K>
__global__ void moe_scoring_cuda_impl(
    int64_t M, int64_t K,
    const half* __restrict__ x,
    const half* __restrict__ w,
    uint8_t* __restrict__ act_exps,
    half* __restrict__ act_exps_scores,
    half* __restrict__ scores
) {

    __shared__ half l_shared[TILE_M][N_EXPS];
    __shared__ float scratch[TILE_M*((N_EXPS+31)/32)];

    const half* x_row = x + blockIdx.x*TILE_M*K;
    half* scores_row = scores + blockIdx.x*TILE_M*N_EXPS;
    uint8_t* act_exps_row = act_exps + blockIdx.x*TILE_M*TOP_K;
    half* act_exps_scores_row = act_exps_scores + blockIdx.x*TILE_M*TOP_K;

    int r = threadIdx.x/N_EXPS;
    int c = threadIdx.x%N_EXPS;

    // compute logits
    float acc = 0.0f;
    for (int k=0; k<K; k+=8) {
        float4 xv = *reinterpret_cast<const float4*>(x_row+r*K+k);
        float4 wv = *reinterpret_cast<const float4*>(w+c*K+k);
        half2* xh = reinterpret_cast<half2*>(&xv);
        half2* wh = reinterpret_cast<half2*>(&wv);
        #pragma unroll
        for (int i=0; i<4; ++i) {
            acc += __half2float(xh[i].x) * __half2float(wh[i].x);
            acc += __half2float(xh[i].y) * __half2float(wh[i].y);
        }
    }
    l_shared[r][c] = __float2half(acc);
    __syncthreads();

    // softmax
    float val = __half2float(l_shared[r][c]);
    float m = row_reduce_max<N_EXPS>(scratch, val, r, c);
    float exp_val = expf(val-m);
    float s = row_reduce_sum<N_EXPS>(scratch, exp_val, r, c);
    float softmax_val = exp_val/s;
    l_shared[r][c] = __float2half(softmax_val);
    scores_row[r*N_EXPS+c] = __float2half(softmax_val);
    __syncthreads();

    // insertion sort topK, one thread per row
    if (c == 0) {
        float top_vals[TOP_K];
        int top_idx[TOP_K];

        #pragma unroll
        for (int i=0; i<TOP_K; ++i) {
            top_vals[i] = -INFINITY;
            top_idx[i] = -1;
        }

        #pragma unroll
        for (int e=0; e<N_EXPS; ++e) {
            float v = __half2float(l_shared[r][e]);
            if (v>top_vals[TOP_K-1]) {
                top_vals[TOP_K-1] = v;
                top_idx[TOP_K-1] = e;
                #pragma unroll
                for (int i=TOP_K-1; i>0 && top_vals[i]>top_vals[i-1]; --i) {
                    float tv = top_vals[i]; top_vals[i] = top_vals[i-1]; top_vals[i-1] = tv;
                    int ti = top_idx[i]; top_idx[i] = top_idx[i-1]; top_idx[i-1] = ti;
                }
            }
        }

        #pragma unroll
        for (int i=0; i<TOP_K; ++i) {
            act_exps_row[r*TOP_K+i] = (uint8_t)top_idx[i];
            act_exps_scores_row[r*TOP_K+i] = __float2half(top_vals[i]);
        }
    }
}

template <int TILE_M, int N_EXPS>
void dispatch_top_k(
    int64_t top_k, int64_t M, int64_t K,
    const half* x_ptr, const half* w_ptr,
    uint8_t* act_exps_ptr, half* act_exps_scores_ptr, half* scores_ptr,
    cudaStream_t stream
) {
    dim3 grid((M+TILE_M-1)/TILE_M);
    dim3 block(TILE_M*N_EXPS);
    
    switch (top_k) {
        case 1: moe_scoring_cuda_impl<TILE_M, N_EXPS, 1><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        case 2: moe_scoring_cuda_impl<TILE_M, N_EXPS, 2><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        case 4: moe_scoring_cuda_impl<TILE_M, N_EXPS, 4><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        case 8: moe_scoring_cuda_impl<TILE_M, N_EXPS, 8><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        default: TORCH_CHECK(false, "unsupported top_k: ", top_k);
    }
}

// NOTE: ffn_gate_inp seems to be kept in half precision.
// need to handle in compute_tiles and elsewhere properly
void moe_scoring_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x, // [B,L,hidden_dim]
    const at::Tensor& w, // [n_experts, hidden_dim]
    at::Tensor& act_exps, // [B,L,n_act_exps]
    at::Tensor& act_exps_scores, // [B,L,n_act_exps]
    at::Tensor& scores // [B,L,n_experts]
) {
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(scores.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(act_exps.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(act_exps_scores.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(scores.is_contiguous());
    TORCH_CHECK(w.is_contiguous());
    TORCH_CHECK(act_exps.is_contiguous());
    TORCH_CHECK(act_exps_scores.is_contiguous());

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(scores.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);
    TORCH_CHECK(act_exps.dtype() == at::kByte);
    TORCH_CHECK(act_exps_scores.dtype() == at::kHalf);

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
    uint8_t* act_exps_ptr = act_exps.data_ptr<uint8_t>();
    half* act_exps_scores_ptr = reinterpret_cast<half*>(act_exps_scores.data_ptr<at::Half>());
    half* scores_ptr = reinterpret_cast<half*>(scores.data_ptr<at::Half>());

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t n_exps = scores.size(-1);
    int64_t top_k = act_exps.size(-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (n_exps) {
        case 8:   dispatch_top_k<128, 8>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 16:  dispatch_top_k<64, 16>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 32:  dispatch_top_k<32, 32>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 64:  dispatch_top_k<16, 64>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 128: dispatch_top_k<8, 128>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 256: dispatch_top_k<4, 256>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        default: TORCH_CHECK(false, "unsupported n_exps: ", n_exps);
    }
}

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
    compute_tile_wmma_f16(
        M, K_GU, N_GU,
        K_GU, K_GU, WMMA_N,
        w_gu_shared,
        in_row, ws_gate_row, (half*)hb_shared
    );

    // up proj stored in hb2
    compute_tile_wmma_f16(
        M, K_GU, N_GU,
        K_GU, K_GU, WMMA_N,
        w_gu_shared,
        in_row, ws_up_row, (half*)hb2_shared
    );

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
    compute_tile_quant(
        gate_qtype_int, gate_qblock_size, gate_qtype_size,
        M, K_GU, N_GU,
        K_GU, K_GU, WMMA_N,
        w_gate_shared,
        in_row, ws_gate_row, (half*)hb_shared
    );
    
    // up proj stored in hb2
    compute_tile_quant(
        up_qtype_int, up_qblock_size, up_qtype_size,
        M, K_GU, N_GU,
        K_GU, K_GU, WMMA_N,
        w_up_shared,
        in_row, ws_up_row, (half*)hb2_shared
    );

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

// elemwise multiplication of up proj tile with swiglu (tile of swish(gate(x))), then apply downproj
void ffn_cuda(
    int64_t up_qtype_int, int64_t gate_qtype_int, int64_t down_qtype_int,
    int64_t up_qblock_size, int64_t gate_qblock_size, int64_t down_qblock_size,
    int64_t up_qtype_size, int64_t gate_qtype_size, int64_t down_qtype_size,
    const at::Tensor& in, // [B,L,hidden_dim]
    const at::Tensor& ws_up, // [n_local_exps, mlp_dim, hidden_dim]
    const at::Tensor& ws_gate, // // [n_local_exps, mlp_dim, hidden_dim]
    const at::Tensor& ws_down, // [n_local_exps, hidden_dim, mlp_dim]
    at::Tensor& hb, // [n_local_exps,B,L,mlp_dim]
    at::Tensor& hb2, // [n_local_exps,B,L,mlp_dim]
    at::Tensor& out // [n_local_exps,B,L,hidden_dim]
) {
    TORCH_CHECK(is_valid_qtype(up_qtype_int) && is_valid_qtype(gate_qtype_int) && is_valid_qtype(down_qtype_int));
    auto up_qtype = static_cast<GGMLQuantizationType>(up_qtype_int);
    auto gate_qtype = static_cast<GGMLQuantizationType>(gate_qtype_int);
    auto down_qtype = static_cast<GGMLQuantizationType>(down_qtype_int);

    //validation
    TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(hb.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(hb2.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws_up.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws_gate.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws_down.device().type() == at::DeviceType::CUDA);

    //contiguity
    TORCH_CHECK(
        in.is_contiguous() && out.is_contiguous() && hb.is_contiguous() && hb2.is_contiguous() && 
        ws_up.is_contiguous() && ws_gate.is_contiguous() && ws_down.is_contiguous()
    );

    //dtypes
    TORCH_CHECK(
        (in.dtype() == at::kHalf) && (out.dtype() == at::kHalf) && 
        (hb.dtype() == at::kHalf) && (hb2.dtype() == at::kHalf) &&
        (ws_up.dtype() == at::kByte || ws_up.dtype() == at::kHalf) && 
        (ws_gate.dtype() == at::kByte || ws_gate.dtype() == at::kHalf) && 
        (ws_down.dtype() == at::kByte || ws_down.dtype() == at::kHalf)
    );

    //dims
    TORCH_CHECK((in.dim() == 3) && (out.dim() == 4));
    TORCH_CHECK((hb.dim() == 4) && (hb2.dim() == 4));
    TORCH_CHECK((ws_up.dim() == 3) && (ws_gate.dim() == 3) && (ws_down.dim() == 3));

    //checks between dimensions
    TORCH_CHECK(in.numel() == out.numel() / out.size(0));
    TORCH_CHECK(hb.sizes().equals(hb2.sizes()));
    TORCH_CHECK(
        ws_up.size(0) == ws_gate.size(0) && ws_gate.size(0) == ws_down.size(0) && ws_up.size(1) == ws_gate.size(1)
    );

    const half* in_ptr = reinterpret_cast<const half*>(in.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    half* hb_ptr = reinterpret_cast<half*>(hb.data_ptr<at::Half>());
    half* hb2_ptr = reinterpret_cast<half*>(hb2.data_ptr<at::Half>());

    int64_t B = in.size(0);
    int64_t L = in.size(1);
    int64_t hidden_dim = in.size(-1);
    int64_t n_local_exps = out.size(0);
    int64_t mlp_dim = hb.size(3);

    int64_t M = B*L;
    int64_t K_GU = hidden_dim;
    int64_t N_GU = mlp_dim;
    int64_t K_DOWN = mlp_dim;
    int64_t N_DOWN = hidden_dim;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // assume: all of gate, up, down are dequantized (F16) or all are quantized i.e. not FP16
    if (
        up_qtype == GGMLQuantizationType::F16 && 
        gate_qtype == GGMLQuantizationType::F16 && 
        down_qtype == GGMLQuantizationType::F16
    ) {
        const half* ws_up_ptr = reinterpret_cast<const half*>(ws_up.data_ptr<at::Half>());
        const half* ws_gate_ptr = reinterpret_cast<const half*>(ws_gate.data_ptr<at::Half>());
        const half* ws_down_ptr = reinterpret_cast<const half*>(ws_down.data_ptr<at::Half>());

        constexpr int WARPS_PER_BLOCK = 4; // tune this as needed
        constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
        constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
        dim3 grid_swiglu(n_local_exps, (M+ROWS_M-1)/ROWS_M, (N_GU+WMMA_N-1)/WMMA_N);
        dim3 grid_down_exp((M+ROWS_M-1)/ROWS_M, (N_DOWN+WMMA_N-1)/WMMA_N);
        dim3 block(BLOCK_SIZE);

        // swiglu
        swiglu_f16_cuda_impl<WARPS_PER_BLOCK><<<grid_swiglu, block, 0, stream>>>(
            M, K_GU, N_GU,
            in_ptr, ws_up_ptr, ws_gate_ptr,
            hb_ptr, hb2_ptr
        );

        // launch down-proj kernel per exp
        for (int e=0; e<n_local_exps; e++) {
            half* hb2_exp = hb2_ptr + e*M*N_GU;
            half* out_exp = out_ptr + e*M*N_DOWN;
            const half* ws_down_exp = ws_down_ptr + e*N_DOWN*K_DOWN;
            
            matmul_wmma_f16_cuda_impl<<<grid_down_exp, block, 0, stream>>>(
                M, K_DOWN, N_DOWN,
                hb2_exp, ws_down_exp, out_exp
            );
        }
    } else {
        const uint8_t* ws_up_ptr = ws_up.data_ptr<uint8_t>();
        const uint8_t* ws_gate_ptr = ws_gate.data_ptr<uint8_t>();
        const uint8_t* ws_down_ptr = ws_down.data_ptr<uint8_t>();

        constexpr int WARPS_PER_BLOCK = 4; // tune this as needed
        constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
        constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
        dim3 grid_swiglu(n_local_exps, (M+ROWS_M-1)/ROWS_M, (N_GU+WMMA_N-1)/WMMA_N);
        dim3 grid_down_exp((M+ROWS_M-1)/ROWS_M, (N_DOWN+WMMA_N-1)/WMMA_N);
        dim3 block(BLOCK_SIZE);

        // swiglu
        swiglu_quant_cuda_impl<WARPS_PER_BLOCK><<<grid_swiglu, block, 0, stream>>>(
            up_qtype_int, gate_qtype_int,
            up_qblock_size, gate_qblock_size,
            up_qtype_size, gate_qtype_size,
            M, K_GU, N_GU,
            in_ptr, ws_up_ptr, ws_gate_ptr,
            hb_ptr, hb2_ptr
        );

        // launch down-proj kernel per expert
        for (int e=0; e<n_local_exps; e++) {
            half* hb2_exp = hb2_ptr + e*M*N_GU;
            half* out_exp = out_ptr + e*M*N_DOWN;
            const uint8_t* ws_down_exp = ws_down_ptr + e*N_DOWN*(K_DOWN/down_qblock_size)*down_qtype_size;
            
            matmul_quant_cuda_impl<<<grid_down_exp, block, 0, stream>>>(
                down_qtype_int, down_qblock_size, down_qtype_size,
                M, K_DOWN, N_DOWN,
                hb2_exp, ws_down_exp, out_exp
            );
        }
    }
}

TORCH_LIBRARY(minfer, m) {
    m.def("dequant(int qtype, int qblock_size, int qtype_size, Tensor x, Tensor(a!) y) -> ()");
    m.def("quant(int qtype, int qblock_size, int qtype_size, Tensor x, Tensor(a!) y) -> ()");
    m.def("rmsnorm(float eps, Tensor input, Tensor w, Tensor(a!) out) -> ()");
    m.def("il_rope(int rotary_dim, int start_pos, float freq_base, Tensor(a!) x) -> ()");
    m.def("neox_rope(int rotary_dim, int start_pos, float freq_base, Tensor(a!) x) -> ()");
    m.def("matmul(int qtype_int, int qblock_size, int qtype_size, Tensor x, Tensor w, Tensor(a!) out) -> ()");
    m.def("embed(int qtype_int, int qblock_size, int qtype_size, Tensor token_ids, Tensor w, Tensor(a!) x) -> ()");
    m.def("qkv(int q_qtype_int, int k_qtype_int, int v_qtype_int, int q_qblock_size, int k_qblock_size, int v_qblock_size, int q_qtype_size, int k_qtype_size, int v_qtype_size, Tensor x, Tensor wq, Tensor wk, Tensor wv, Tensor(a!) q_out, Tensor(a!) k_out, Tensor(a!) v_out) -> ()");
    m.def("flash_attn(int qtype_int, int qblock_size, int qtype_size, Tensor q, Tensor k, Tensor v, Tensor(a!) mask, Tensor(a!) out) -> ()");
    m.def("moe_scoring(int qtype_int, int qblock_size, int qtype_size, Tensor x, Tensor w, Tensor(a!) act_exps, Tensor(a!) act_exps_scores, Tensor(a!) scores) -> ()");
    m.def("ffn(int up_qtype_int, int gate_qtype_int, int down_qtype_int, int up_qblock_size, int gate_qblock_size, int down_qblock_size, int up_qtype_size, int gate_qtype_size, int down_qtype_size, Tensor input, Tensor ws_up, Tensor ws_gate, Tensor ws_down, Tensor(a!) hb, Tensor(a!) hb2, Tensor(a!) out) -> ()");
}

// TODO: add this back in once finished
TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
    m.impl("rmsnorm", &rmsnorm_cuda);
    m.impl("il_rope", &il_rope_cuda);
    m.impl("neox_rope", &neox_rope_cuda);
    m.impl("matmul", &matmul_cuda);
    m.impl("embed", &embed_cuda);
    m.impl("qkv", &qkv_cuda);
    m.impl("flash_attn", &flash_attn_cuda);
    m.impl("moe_scoring", &moe_scoring_cuda);
    m.impl("ffn", &ffn_cuda);
}

}