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

    v = warp_reduce_sum(v);
    if (lane_id == 0) vs[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) v = warp_reduce_sum(vs[lane_id]);
    if (tid == 0) vs[0] = v;
    __syncthreads();
    return vs[0];
}

static __device__ float blockreduce_max(float* vs, float v, int tid) {
    int warp_id = tid/32;
    int lane_id = tid%32;

    v = warp_reduce_max(v);
    if (lane_id == 0) vs[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) v = warp_reduce_max(vs[lane_id]);
    if (tid == 0) vs[0] = v;
    __syncthreads();
    return vs[0];
}

// helper for dispatch in matmul_cuda_impl
// might template this later on to additionally support float
static __device__ void dequant_block(
    int64_t qtype_int,
    int64_t stride,
    int64_t tid,
    half* __restrict__ y,
    const uint8_t* __restrict__ w
) {
    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_block_q4_0<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q4_1: dequant_block_q4_1<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q5_0: dequant_block_q5_0<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q5_1: dequant_block_q5_1<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q8_0: dequant_block_q8_0<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::MXFP4: dequant_block_mxfp4<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q2_K: dequant_block_q2_K<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q3_K: dequant_block_q3_K<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q4_K: dequant_block_q4_K<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q5_K: dequant_block_q5_K<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q6_K: dequant_block_q6_K<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::TQ1_0: dequant_block_tq1_0<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::TQ2_0: dequant_block_tq2_0<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_block_iq2_xxs<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ2_XS: dequant_block_iq2_xs<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ2_S: dequant_block_iq2_s<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_block_iq3_xxs<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ3_S: dequant_block_iq3_s<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ1_S: dequant_block_iq1_s<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ1_M: dequant_block_iq1_m<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ4_NL: dequant_block_iq4_nl<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ4_XS: dequant_block_iq4_xs<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q8_K: dequant_block_q8_K<half>(w, y, stride, tid); break;
        case GGMLQuantizationType::BF16: dequant_block_bf16<half>(w, y, stride, tid); break;
        default: assert(false && "Unsupported dtype"); // this gets compiled out in non-debug builds...
    }
}

__global__ void rmsnorm_cuda_impl(
    int64_t dim,
    float eps,
    const half* __restrict__ in,
    half* __restrict__ out,
    const half* __restrict__ w
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
    at::Tensor& out, 
    const at::Tensor& w
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
    rmsnorm_cuda_impl<<<n_blocks, block_size, 0, stream>>>(dim, eps, in_ptr, out_ptr, w_ptr);
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

template <int TILE_DIM>
static __device__ void compute_tile_f16(
    int64_t M, int64_t K, int64_t N,
    int64_t x_stride,
    int64_t out_stride,
    int64_t w_stride,
    half (&x_shared)[TILE_DIM][TILE_DIM],
    half (&w_shared)[TILE_DIM][TILE_DIM+2], // avoid bank conflict
    const half* __restrict__ x,
    half* __restrict__ out,
    const half* __restrict__ w
) {
    int r = threadIdx.y;
    int c = threadIdx.x;
    int global_r = blockIdx.x*TILE_DIM+r;
    int global_c = blockIdx.y*TILE_DIM+c;

    float acc = 0.0f;

    for (int64_t k=0; k<K; k+=TILE_DIM) {
        
        x_shared[r][c] = (global_r<M && k+c<K) ? x[r*x_stride+k+c] : __float2half(0.0f);
        w_shared[c][r] = (blockIdx.y*TILE_DIM+r<N && k+c<K) ? w[r*w_stride+k+c] : __float2half(0.0f);

        __syncthreads();
        
        for (int kk=0; kk<TILE_DIM; ++kk) {
            acc += __half2float(x_shared[r][kk]) * __half2float(w_shared[kk][c]);
        }
        
        __syncthreads();
    }

    if (global_r<M && global_c<N) {
        out[r*out_stride+c] = __float2half(acc);
    }
}

// helper to compute TILE_M x TILE_N portion of result
template <int TILE_M, int TILE_N, int TILE_K, bool row_major_x = true, bool row_major_w = false>
static __device__ void compute_tile_wmma_f16(
    int64_t K,
    int64_t x_stride,
    int64_t out_stride,
    int64_t w_stride,
    half* __restrict__ x_shared, // [TILE_M, TILE_K]
    float* __restrict__ out_shared, // [TILE_M, TILE_N]
    half* __restrict__ w_shared, // [TILE_N, TILE_K]
    const half* __restrict__ x,
    half* __restrict__ out,
    const half* __restrict__ w
) {
    constexpr int NUM_SUBTILES_M = (TILE_M/16);
    constexpr int NUM_SUBTILES_N = (TILE_N/16);

    // wmma operates on 16x16 subtiles
    constexpr int NUM_SUBTILES = NUM_SUBTILES_M*NUM_SUBTILES_N;


    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    // split work over the subtiles (arranged in a grid of TILE_SIZE/16xTILE_SIZE/16), each of size 16x16.
    for (int i=0; i<tiles_per_warp; ++i) {
        int tile_id = warp_id * tiles_per_warp + i;
        int m_subtile = (tile_id / NUM_SUBTILES_N)*16;
        int n_subtile = (tile_id % NUM_SUBTILES_N)*16;

        wmma::fill_fragment(acc_frag, 0.0f);

        // K-dimension processed in TILE_K chunks
        for (int64_t k_tile=0; k_tile<K; k_tile+=TILE_K) {
            
            // coop load x tile and w tile to shared mem
            for (int idx=threadIdx.x; idx<TILE_M*TILE_K; idx+=blockDim.x) {
                int ml = idx / TILE_K;
                int kl = idx % TILE_K;
                if constexpr (row_major_x) {
                    x_shared[ml*TILE_K+kl] = x[ml*x_stride + k_tile + kl];
                } else {
                    x_shared[ml*TILE_K+kl] = x[(k_tile+kl)*x_stride + ml];
                }
            }

            // coop load of weight to shared mem
            for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                int nl = idx / TILE_K;
                int kl = idx % TILE_K;
                if constexpr (row_major_w) {
                    w_shared[nl*TILE_K + kl] = w[(k_tile+kl)*w_stride + nl];
                } else {
                    w_shared[nl*TILE_K + kl] = w[nl*w_stride + k_tile + kl];
                }
            }

            __syncthreads();

            // process TILE_K with WMMA in chunks of 16
            for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            __syncthreads();
        }

        wmma::store_matrix_sync(
            out_shared + m_subtile*TILE_N + n_subtile,
            acc_frag,
            TILE_N,
            wmma::mem_row_major
        );
    }

    __syncthreads();

    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        out[r*out_stride+c] = __float2half(out_shared[r*TILE_N+c]);
    }
}

// helper to compute TILE_M x TILE_N portion of result
template <int TILE_M, int TILE_N, int TILE_K, bool row_major_x = true, bool row_major_w = false>
static __device__ void compute_tile_quant(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    int64_t K,
    int64_t x_stride,
    int64_t out_stride,
    int64_t w_stride,
    half* __restrict__ x_shared, // [TILE_M, TILE_K]
    float* __restrict__ out_shared, // [TILE_M, TILE_N]
    half* __restrict__ w_shared, // [TILE_N, TILE_K]
    const half* __restrict__ x,
    half* __restrict__ out,
    const uint8_t* __restrict__ w
) {
    constexpr int NUM_SUBTILES_M = (TILE_M/16);
    constexpr int NUM_SUBTILES_N = (TILE_N/16);

    // wmma operates on 16x16 subtiles
    constexpr int NUM_SUBTILES = NUM_SUBTILES_M*NUM_SUBTILES_N;

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    // split work over the subtiles (arranged in a grid of TILE_SIZE/16xTILE_SIZE/16), each of size 16x16.
    for (int i=0; i<tiles_per_warp; ++i) {
        int tile_id = warp_id * tiles_per_warp + i;
        int m_subtile = (tile_id / NUM_SUBTILES_N)*16;
        int n_subtile = (tile_id % NUM_SUBTILES_N)*16;

        wmma::fill_fragment(acc_frag, 0.0f);

        // K-dimension processed in TILE_K chunks
        for (int64_t k_tile=0; k_tile<K; k_tile+=TILE_K) {
            
            // coop load x tile and w tile to shared mem
            for (int idx=threadIdx.x; idx<TILE_M*TILE_K; idx+=blockDim.x) {
                int ml = idx / TILE_K;
                int kl = idx % TILE_K;
                if constexpr (row_major_x) {
                    x_shared[ml*TILE_K+kl] = x[ml*x_stride + k_tile + kl];
                } else {
                    x_shared[ml*TILE_K+kl] = x[(k_tile+kl)*x_stride + ml];
                }
            }

            // coop load of weight to shared mem
            for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                int nl = idx / TILE_K;
                int kb = (idx % TILE_K) / qblock_size;
                int kl = (idx % TILE_K) % qblock_size;
                const uint8_t* w_block;
                if constexpr (row_major_w) {
                    w_block = w + ((k_tile/qblock_size + kb)*(w_stride/qblock_size) + nl)*qtype_size;
                } else {
                    w_block = w + (nl*(w_stride/qblock_size) + (k_tile/qblock_size + kb))*qtype_size;
                }
                dequant_block(qtype_int, 1, kl, w_shared + nl*TILE_K + kb*qblock_size, w_block);
            }

            __syncthreads();

            // process TILE_K with WMMA in chunks of 16
            for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            __syncthreads();
        }

        wmma::store_matrix_sync(
            out_shared + m_subtile*TILE_N + n_subtile,
            acc_frag,
            TILE_N,
            wmma::mem_row_major
        );
    }

    __syncthreads();

    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
        int r = idx/TILE_N;
        int c = idx%TILE_N;
        out[r*out_stride+c] = __float2half(out_shared[r*TILE_N+c]);
    }
}

template <int TILE_DIM>
__global__ void matmul_f16_cuda_impl(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    half* __restrict__ out,
    const half* __restrict__ w
) {
    __shared__ half x_shared[TILE_DIM][TILE_DIM];
    __shared__ half w_shared[TILE_DIM][TILE_DIM+2];

    const half* x_row = x + blockIdx.x*TILE_DIM*K;
    half* out_tile = out + blockIdx.x*TILE_DIM*N + blockIdx.y*TILE_DIM;
    const half* w_row = w + blockIdx.y*TILE_DIM*K;

    compute_tile_f16<TILE_DIM>(
        M, K, N,
        K, N, K,
        x_shared, w_shared,
        x_row, out_tile, w_row
    );
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_wmma_f16_cuda_impl(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    half* __restrict__ out,
    const half* __restrict__ w
) {
    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ float out_shared[TILE_M][TILE_N];
    __shared__ half w_shared[TILE_N][TILE_K];

    const half* x_row = x + blockIdx.x*TILE_M*K;
    half* out_tile = out + blockIdx.x*TILE_M*N + blockIdx.y*TILE_N;
    const half* w_row = w + blockIdx.y*TILE_N*K;

    compute_tile_wmma_f16<TILE_M, TILE_N, TILE_K>(
        K, K, N, K,
        (half*)x_shared, (float*)out_shared, (half*)w_shared,
        x_row, out_tile, w_row
    );
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_quant_cuda_impl(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    half* __restrict__ out,
    const uint8_t* __restrict__ w
) {
    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ float out_shared[TILE_M][TILE_N];
    __shared__ half w_shared[TILE_N][TILE_K];

    const half* x_row = x + blockIdx.x*TILE_M*K;
    half* out_tile = out + blockIdx.x*TILE_M*N + blockIdx.y*TILE_N;
    const uint8_t* w_row = w + blockIdx.y*TILE_N*(K/qblock_size)*qtype_size;

    compute_tile_quant<TILE_M, TILE_N, TILE_K>(
        qtype_int, qblock_size, qtype_size,
        K, K, N, K,
        (half*)x_shared, (float*)out_shared, (half*)w_shared,
        x_row, out_tile, w_row
    );
}

void matmul_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x, // [B,L,in_dim]
    at::Tensor& out, // [B,L,out_dim]
    const at::Tensor& w // [out_dim, in_dim (possibly bytes)]
) {
    TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

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
    TORCH_CHECK(w.size(0) == out.size(-1));

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t N = w.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (qtype == GGMLQuantizationType::F16) {
        TORCH_CHECK(w.size(-1) == x.size(-1));
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
        
        if (N >= 16) {
            constexpr int TILE_SIZE = 64;
            constexpr int BLOCK_SIZE = 512; // i think this is 1 subtile per warp (512/32=16 warps, (64/16)*(64/16)=16 subtiles)
            dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
            dim3 block(BLOCK_SIZE);
            matmul_wmma_f16_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_SIZE><<<grid, block, 0, stream>>>(
                M, K, N, x_ptr, out_ptr, w_ptr
            );
        } else {
            constexpr int TILE_DIM = 8;
            dim3 grid((M+TILE_DIM-1)/TILE_DIM, (N+TILE_DIM-1)/TILE_DIM);
            dim3 block(TILE_DIM, TILE_DIM);
            matmul_f16_cuda_impl<TILE_DIM><<<grid, block, 0, stream>>>(
                M, K, N, x_ptr, out_ptr, w_ptr
            );
        }
    } else {
        TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size));
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
        constexpr int TILE_SIZE = 32;
        constexpr int TILE_K = 256;
        constexpr int BLOCK_SIZE = 128; // also 1 subtile per warp
        dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
        dim3 block(BLOCK_SIZE);

        matmul_quant_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K><<<grid, block, 0, stream>>>(
            qtype_int, qblock_size, qtype_size, M, K, N, x_ptr, out_ptr, w_ptr
        );
    }
}

__global__ void embed_f16_cuda_impl(
    int64_t hidden_dim,
    half* __restrict__ x,
    const int64_t* __restrict__ token_ids,
    const half* __restrict__ w
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
    half* __restrict__ x,
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ w
) {
    int64_t L = gridDim.y;

    int64_t iB = blockIdx.x;
    int64_t iL = blockIdx.y;
    int64_t block_in_row = blockIdx.z;
    int64_t token_id = token_ids[iB*L+iL];
    
    const uint8_t* w_block = w + token_id*b + block_in_row*qtype_size;
    half* x_block = x + iB*L*k + iL*k + block_in_row*qblock_size;
    dequant_block(qtype_int, 1, threadIdx.x, x_block, w_block);
}

// NOTE: for now this assumes that the embedding matrix is quantized
// (it usually is)
void embed_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    at::Tensor& x, // [B,L,D]
    const at::Tensor& token_ids, // [B,L]
    const at::Tensor& w // [vocab_size, hidden_dim (possibly bytes)]
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

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t hidden_dim = x.size(2);

    if (qtype == GGMLQuantizationType::F16) {
        TORCH_CHECK(w.size(-1) == x.size(-1));
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
        dim3 grid(B, L);
        embed_f16_cuda_impl<<<grid, 1024, 0, stream>>>(
            hidden_dim, x_ptr, token_ids_ptr, w_ptr
        );
    } else {
        TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size));
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
        int64_t n_qblocks_per_row = hidden_dim / qblock_size;
        int64_t b = w.size(-1); // bytes per row
        int64_t k = hidden_dim; // dequant elems per row
        dim3 grid(B, L, n_qblocks_per_row);
        embed_quant_cuda_impl<<<grid, qblock_size, 0, stream>>>(
            qtype_int, qblock_size, qtype_size, b, k, x_ptr, token_ids_ptr, w_ptr
        );
    }
}

// helper to compute TILE_M x TILE_N portion of result for QKV projections
// reduces loads from global mem of input x by x3 factor
template <int TILE_M, int TILE_N, int TILE_K>
static __device__ void compute_qkv_tile_f16(
    int64_t N_Q, int64_t N_KV, 
    int64_t K, 
    int64_t x_stride, 
    int64_t q_out_stride, int64_t kv_out_stride, 
    int64_t wq_stride, int64_t wkv_stride,
    half* __restrict__ x_shared, // [TILE_M, TILE_K]
    float* __restrict__ q_out_shared, // [TILE_M, TILE_N]
    float* __restrict__ k_out_shared, // [TILE_M, TILE_N]
    float* __restrict__ v_out_shared, // [TILE_M, TILE_N]
    half* __restrict__ w_shared, // [TILE_N, TILE_K]
    const half* __restrict__ x,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    const half* __restrict__ wq,
    const half* __restrict__ wk,
    const half* __restrict__ wv

) {
    constexpr int NUM_SUBTILES_M = (TILE_M/16);
    constexpr int NUM_SUBTILES_N = (TILE_N/16);

    // wmma operates on 16x16 subtiles
    constexpr int NUM_SUBTILES = NUM_SUBTILES_M*NUM_SUBTILES_N;

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> q_acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> k_acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_acc_frag;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    // split work over the subtiles (arranged in a grid of TILE_SIZE/16xTILE_SIZE/16), each of size 16x16.
    for (int i=0; i<tiles_per_warp; ++i) {
        int tile_id = warp_id * tiles_per_warp + i;
        int m_subtile = (tile_id / NUM_SUBTILES_N)*16;
        int n_subtile = (tile_id % NUM_SUBTILES_N)*16;

        wmma::fill_fragment(q_acc_frag, 0.0f);
        wmma::fill_fragment(k_acc_frag, 0.0f);
        wmma::fill_fragment(v_acc_frag, 0.0f);

        // K-dimension processed in TILE_K chunks
        for (int64_t k_tile=0; k_tile<K; k_tile+=TILE_K) {
            
            // coop load x tile to shared mem
            for (int idx=threadIdx.x; idx<TILE_M*TILE_K; idx+=blockDim.x) {
                int ml = idx / TILE_K;
                int kl = idx % TILE_K;
                x_shared[ml*TILE_K+kl] = x[ml*x_stride + k_tile + kl];
            }

            if (blockIdx.y * TILE_N < N_Q) {
                // coop load of wq to shared mem
                for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                    int nl = idx / TILE_K;
                    int kl = idx % TILE_K;
                    w_shared[nl*TILE_K + kl] = wq[nl*wq_stride + k_tile + kl];
                }

                __syncthreads();

                // process TILE_K with WMMA in chunks of 16
                for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                    wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::mma_sync(q_acc_frag, a_frag, b_frag, q_acc_frag);
                }
            }

            __syncthreads();

            if (blockIdx.y * TILE_N < N_KV) {
                
                // coop load of wk to shared mem
                for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                    int nl = idx / TILE_K;
                    int kl = idx % TILE_K;
                    w_shared[nl*TILE_K + kl] = wk[nl*wkv_stride + k_tile + kl];
                }

                __syncthreads();

                // process TILE_K with WMMA in chunks of 16
                for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                    wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::mma_sync(k_acc_frag, a_frag, b_frag, k_acc_frag);
                }

                __syncthreads();

                // coop load of wv to shared mem
                for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                    int nl = idx / TILE_K;
                    int kl = idx % TILE_K;
                    w_shared[nl*TILE_K + kl] = wv[nl*wkv_stride + k_tile + kl];
                }

                __syncthreads();

                // process TILE_K with WMMA in chunks of 16
                for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                    wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::mma_sync(v_acc_frag, a_frag, b_frag, v_acc_frag);
                }
            }

            __syncthreads();
        }

        if (blockIdx.y * TILE_N < N_Q) {
            wmma::store_matrix_sync(
                q_out_shared + m_subtile*TILE_N + n_subtile,
                q_acc_frag,
                TILE_N,
                wmma::mem_row_major
            );
        }
        
        if (blockIdx.y * TILE_N < N_KV) {
            wmma::store_matrix_sync(
                k_out_shared + m_subtile*TILE_N + n_subtile,
                k_acc_frag,
                TILE_N,
                wmma::mem_row_major
            );

            wmma::store_matrix_sync(
                v_out_shared + m_subtile*TILE_N + n_subtile,
                v_acc_frag,
                TILE_N,
                wmma::mem_row_major
            );
        }
    }

    __syncthreads();

    if (blockIdx.y * TILE_N < N_Q) {
        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            q_out[r*q_out_stride+c] = __float2half(q_out_shared[r*TILE_N+c]);
        }
    }

    if (blockIdx.y * TILE_N < N_KV) {
        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            k_out[r*kv_out_stride+c] = __float2half(k_out_shared[r*TILE_N+c]);
        }

        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            v_out[r*kv_out_stride+c] = __float2half(v_out_shared[r*TILE_N+c]);
        }
    }
}

// helper to compute TILE_M x TILE_N portion of result for QKV projections
// reduces loads from global mem of input x by x3 factor
template <int TILE_M, int TILE_N, int TILE_K>
static __device__ void compute_qkv_tile_quant(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size,  int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    int64_t N_Q, int64_t N_KV,
    int64_t K, 
    int64_t x_stride,
    int64_t q_out_stride, int64_t kv_out_stride,
    int64_t wq_stride, int64_t wkv_stride,
    half* __restrict__ x_shared, // [TILE_M, TILE_K]
    float* __restrict__ q_out_shared, // [TILE_M, TILE_N]
    float* __restrict__ k_out_shared, // [TILE_M, TILE_N]
    float* __restrict__ v_out_shared, // [TILE_M, TILE_N]
    half* __restrict__ w_shared, // [TILE_N, TILE_K]
    const half* __restrict__ x,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv

) {
    constexpr int NUM_SUBTILES_M = (TILE_M/16);
    constexpr int NUM_SUBTILES_N = (TILE_N/16);

    // wmma operates on 16x16 subtiles
    constexpr int NUM_SUBTILES = NUM_SUBTILES_M*NUM_SUBTILES_N;

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> q_acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> k_acc_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> v_acc_frag;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

    // split work over the subtiles (arranged in a grid of TILE_SIZE/16xTILE_SIZE/16), each of size 16x16.
    for (int i=0; i<tiles_per_warp; ++i) {
        int tile_id = warp_id * tiles_per_warp + i;
        int m_subtile = (tile_id / NUM_SUBTILES_N)*16;
        int n_subtile = (tile_id % NUM_SUBTILES_N)*16;

        wmma::fill_fragment(q_acc_frag, 0.0f);
        wmma::fill_fragment(k_acc_frag, 0.0f);
        wmma::fill_fragment(v_acc_frag, 0.0f);

        // K-dimension processed in TILE_K chunks
        for (int64_t k_tile=0; k_tile<K; k_tile+=TILE_K) {
            
            // coop load x tile to shared mem
            for (int idx=threadIdx.x; idx<TILE_M*TILE_K; idx+=blockDim.x) {
                int ml = idx / TILE_K;
                int kl = idx % TILE_K;
                x_shared[ml*TILE_K+kl] = x[ml*x_stride + k_tile + kl];
            }

            if (blockIdx.y * TILE_N < N_Q) {
                // coop load of wq to shared mem
                for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                    int nl = idx / TILE_K;
                    int kb = (idx % TILE_K) / q_qblock_size;
                    int kl = (idx % TILE_K) % q_qblock_size;
                    const uint8_t* wq_block = wq + (nl*(wq_stride/q_qblock_size) + (k_tile/q_qblock_size + kb))*q_qtype_size;
                    dequant_block(q_qtype_int, 1, kl, w_shared + nl*TILE_K + kb*q_qblock_size, wq_block);
                }

                __syncthreads();

                // process TILE_K with WMMA in chunks of 16
                for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                    wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::mma_sync(q_acc_frag, a_frag, b_frag, q_acc_frag);
                }
            }

            __syncthreads();

            if (blockIdx.y * TILE_N < N_KV) {
                
                // coop load of wk to shared mem
                for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                    int nl = idx / TILE_K;
                    int kb = (idx % TILE_K) / k_qblock_size;
                    int kl = (idx % TILE_K) % k_qblock_size;
                    const uint8_t* wk_block = wk + (nl*(wkv_stride/k_qblock_size) + (k_tile/k_qblock_size + kb))*k_qtype_size;
                    dequant_block(k_qtype_int, 1, kl, w_shared + nl*TILE_K + kb*k_qblock_size, wk_block);
                }

                __syncthreads();

                // process TILE_K with WMMA in chunks of 16
                for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                    wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::mma_sync(k_acc_frag, a_frag, b_frag, k_acc_frag);
                }

                __syncthreads();

                // coop load of wv to shared mem
                for (int idx=threadIdx.x; idx<TILE_N*TILE_K; idx+=blockDim.x) {
                    int nl = idx / TILE_K;
                    int kb = (idx % TILE_K) / v_qblock_size;
                    int kl = (idx % TILE_K) % v_qblock_size;
                    const uint8_t* wv_block = wv + (nl*(wkv_stride/v_qblock_size) + (k_tile/v_qblock_size + kb))*v_qtype_size;
                    dequant_block(v_qtype_int, 1, kl, w_shared + nl*TILE_K + kb*v_qblock_size, wv_block);
                }

                __syncthreads();

                // process TILE_K with WMMA in chunks of 16
                for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                    wmma::load_matrix_sync(a_frag, x_shared+m_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::load_matrix_sync(b_frag, w_shared+n_subtile*TILE_K+k_subtile, TILE_K);
                    wmma::mma_sync(v_acc_frag, a_frag, b_frag, v_acc_frag);
                }
            }

            __syncthreads();
        }

        if (blockIdx.y * TILE_N < N_Q) {
            wmma::store_matrix_sync(
                q_out_shared + m_subtile*TILE_N + n_subtile,
                q_acc_frag,
                TILE_N,
                wmma::mem_row_major
            );
        }
        
        if (blockIdx.y * TILE_N < N_KV) {
            wmma::store_matrix_sync(
                k_out_shared + m_subtile*TILE_N + n_subtile,
                k_acc_frag,
                TILE_N,
                wmma::mem_row_major
            );

            wmma::store_matrix_sync(
                v_out_shared + m_subtile*TILE_N + n_subtile,
                v_acc_frag,
                TILE_N,
                wmma::mem_row_major
            );
        }
    }

    __syncthreads();

    if (blockIdx.y * TILE_N < N_Q) {
        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            q_out[r*q_out_stride+c] = __float2half(q_out_shared[r*TILE_N+c]);
        }
    }

    if (blockIdx.y * TILE_N < N_KV) {
        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            k_out[r*kv_out_stride+c] = __float2half(k_out_shared[r*TILE_N+c]);
        }

        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            v_out[r*kv_out_stride+c] = __float2half(v_out_shared[r*TILE_N+c]);
        }
    }
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void qkv_f16_cuda_impl(
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    const half* __restrict__ x,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    const half* __restrict__ wq,
    const half* __restrict__ wk,
    const half* __restrict__ wv
) {

    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ float q_out_shared[TILE_M][TILE_N];
    __shared__ float k_out_shared[TILE_M][TILE_N];
    __shared__ float v_out_shared[TILE_M][TILE_N];
    __shared__ half w_shared[TILE_K][TILE_N];

    const half* x_row = x + blockIdx.x*TILE_M*K;
    half* q_out_tile = q_out + blockIdx.x*TILE_M*N_Q + blockIdx.y*TILE_N;
    half* k_out_tile = k_out + blockIdx.x*TILE_M*N_KV + blockIdx.y*TILE_N;
    half* v_out_tile = v_out + blockIdx.x*TILE_M*N_KV + blockIdx.y*TILE_N;
    
    const half* wq_row = wq + blockIdx.y*TILE_N*K;
    const half* wk_row = wk + blockIdx.y*TILE_N*K;
    const half* wv_row = wv + blockIdx.y*TILE_N*K;

    compute_qkv_tile_f16<TILE_M, TILE_N, TILE_K>(
        N_Q, N_KV,
        K, 
        K,
        N_Q, N_KV,
        K, K,
        (half*)x_shared, 
        (float*)q_out_shared, (float*)k_out_shared, (float*)v_out_shared, 
        (half*)w_shared,
        x_row,
        q_out_tile, k_out_tile, v_out_tile,
        wq_row, wk_row, wv_row
    );
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void qkv_quant_cuda_impl(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size,  int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    int64_t M, int64_t K, int64_t N_Q, int64_t N_KV,
    const half* __restrict__ x,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ wk,
    const uint8_t* __restrict__ wv
) {

    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ float q_out_shared[TILE_M][TILE_N];
    __shared__ float k_out_shared[TILE_M][TILE_N];
    __shared__ float v_out_shared[TILE_M][TILE_N];
    __shared__ half w_shared[TILE_K][TILE_N];

    const half* x_row = x + blockIdx.x*TILE_M*K;
    half* q_out_tile = q_out + blockIdx.x*TILE_M*N_Q + blockIdx.y*TILE_N;
    half* k_out_tile = k_out + blockIdx.x*TILE_M*N_KV + blockIdx.y*TILE_N;
    half* v_out_tile = v_out + blockIdx.x*TILE_M*N_KV + blockIdx.y*TILE_N;
    
    const uint8_t* wq_row = wq + blockIdx.y*TILE_N*(K/q_qblock_size)*q_qtype_size;
    const uint8_t* wk_row = wk + blockIdx.y*TILE_N*(K/k_qblock_size)*k_qtype_size;
    const uint8_t* wv_row = wv + blockIdx.y*TILE_N*(K/v_qblock_size)*v_qtype_size;

    compute_qkv_tile_quant<TILE_M, TILE_N, TILE_K>(
        q_qtype_int, q_qblock_size, q_qtype_size,
        k_qtype_int, k_qblock_size, k_qtype_size,
        v_qtype_int, v_qblock_size, v_qtype_size,
        N_Q, N_KV,
        K, 
        K,
        N_Q, N_KV,
        K, K,
        (half*)x_shared, 
        (float*)q_out_shared, (float*)k_out_shared, (float*)v_out_shared,
        (half*)w_shared,
        x_row,
        q_out_tile, k_out_tile, v_out_tile,
        wq_row, wk_row, wv_row
    );
}

void qkv_cuda(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size, int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    const at::Tensor& x, // [B, L, hidden_dim]
    at::Tensor& q_out, // [B, L, q_dim]
    at::Tensor& k_out, // [B, L, kv_dim]
    at::Tensor& v_out, // [B, L, kv_dim]
    const at::Tensor& wq, // [hidden_dim, q_dim (possibly in bytes)]
    const at::Tensor& wk, // [hidden_dim, kv_dim (possibly in bytes)]
    const at::Tensor& wv // [hidden_dim, kv_dim (possibly in bytes)]
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
        constexpr int TILE_SIZE = 32;
        constexpr int TILE_K = 256;
        constexpr int BLOCK_SIZE = 128; // 1 subtile per warp
    
        dim3 block(BLOCK_SIZE); 
        dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

        qkv_f16_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K><<<grid, block, 0, stream>>>(
            M, K, N_Q, N_KV,
            x_ptr,
            q_out_ptr, k_out_ptr, v_out_ptr,
            wq_ptr, wk_ptr, wv_ptr
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
        constexpr int TILE_SIZE = 32;
        constexpr int TILE_K = 256;
        constexpr int BLOCK_SIZE = 128; // 1 subtile per warp
    
        dim3 block(BLOCK_SIZE); 
        dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

        qkv_quant_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K><<<grid, block, 0, stream>>>(
            q_qtype_int, k_qtype_int, v_qtype_int,
            q_qblock_size, k_qblock_size, v_qblock_size,
            q_qtype_size, k_qtype_size, v_qtype_size,
            M, K, N_Q, N_KV,
            x_ptr,
            q_out_ptr, k_out_ptr, v_out_ptr,
            wq_ptr, wk_ptr, wv_ptr
        );
    }
}

// helper for moe scoring and flash attn, where (online) softmax is fused and taken per-tile
// first thread in warps responsible for writing warp vals (d and m) [per-row]
// threads in first warp responsible for reducing over warp vals [per-row]
template <int THR_PER_ROW>
static __device__ float row_reduce_max(float* vs, float v, int row_in_tile, int thr_in_row) {

    v = warp_reduce_max<THR_PER_ROW>(v);

    constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;

    int warp_in_row = thr_in_row / 32;
    if (thr_in_row % 32 == 0) vs[row_in_tile*WARPS_PER_ROW+warp_in_row]=v;
    __syncthreads();
    
    if (THR_PER_ROW/32 > 0 && warp_in_row == 0 && thr_in_row < WARPS_PER_ROW) {
        v = vs[row_in_tile*WARPS_PER_ROW + thr_in_row];
        v = warp_reduce_max(v);
    }

    if (thr_in_row == 0) vs[row_in_tile*WARPS_PER_ROW] = v;
    __syncthreads();
    return vs[row_in_tile*WARPS_PER_ROW];
}

template <int THR_PER_ROW>
static __device__ float row_reduce_sum(float* vs, float v, int row_in_tile, int thr_in_row) {

    v = warp_reduce_sum<THR_PER_ROW>(v);

    constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;

    int warp_in_row = thr_in_row / 32;
    if (thr_in_row % 32 == 0) vs[row_in_tile*WARPS_PER_ROW+warp_in_row]=v;
    __syncthreads();
    
    if (THR_PER_ROW/32 > 0 && warp_in_row == 0 && thr_in_row < WARPS_PER_ROW) {
        v = vs[row_in_tile*WARPS_PER_ROW + thr_in_row];
        v = warp_reduce_sum(v);
    }

    if (thr_in_row == 0) vs[row_in_tile*WARPS_PER_ROW] = v;
    __syncthreads();
    return vs[row_in_tile*WARPS_PER_ROW];
}

template <int TILE_M, int TILE_N, int THR_PER_ROW>
static __device__ void update_dm(
    float* __restrict__ ds,
    float* __restrict__ ms,
    float* __restrict__ scratch_d,
    float* __restrict__ scratch_m,
    const half* __restrict__ tile
) {

    int row_in_tile = threadIdx.x/THR_PER_ROW;
    int thr_in_row = threadIdx.x%THR_PER_ROW;

    // compute tile's m vals per row, write to tiles_m
    float ml = -INFINITY;
    for (int col=thr_in_row; col<TILE_N; col+=THR_PER_ROW) {
        ml = fmaxf(ml, __half2float(tile[row_in_tile*TILE_N+col])); 
    }

    float row_m = row_reduce_max<THR_PER_ROW>(scratch_m, ml, row_in_tile, thr_in_row);

    // compute tile's d vals per row, write to tiles_d
    float dl = 0.0f;
    for (int col=thr_in_row; col<TILE_N; col+=THR_PER_ROW) {
        dl += expf(__half2float(tile[row_in_tile*TILE_N+col])-row_m);
    }

    float row_d = row_reduce_sum<THR_PER_ROW>(scratch_d, dl, row_in_tile, thr_in_row);

    // update m and d
    if (thr_in_row == 0) {
        float row_m_old = ms[row_in_tile];
        float row_d_old = ds[row_in_tile];
        float row_m_new = fmaxf(row_m_old, row_m);
        float row_d_new = row_d_old*expf(row_m_old-row_m_new)+row_d*expf(row_m-row_m_new);
        ms[row_in_tile] = row_m_new;
        ds[row_in_tile] = row_d_new;
    }
}

// FlashAttention-2 forward pass impl.
// Refer to: https://arxiv.org/abs/2307.08691
constexpr float HALF_MIN = -65504.0f;
struct QKVStrides {
    int64_t q0, q1, q2;
    int64_t kv0, kv1, kv2;
};
template <int TILE_M, int TILE_N, int HEAD_DIM, int BLOCK_SIZE>
__global__ void flash_attn_cuda_impl(
    QKVStrides strides,
    int64_t L, int64_t n_kv_heads,
    const uint8_t* __restrict__ mask,
    half* __restrict__ out,
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v
) {
    
    constexpr int THR_PER_ROW = BLOCK_SIZE / TILE_M;
    constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;

    __shared__ half in_shared[TILE_M][HEAD_DIM];
    __shared__ half kv_shared[TILE_N][HEAD_DIM];
    __shared__ half l_shared[TILE_M][TILE_N];
    
    __shared__ float qk_acc_shared[TILE_M][TILE_N];
    __shared__ float out_acc_shared[TILE_M][HEAD_DIM];
    __shared__ half out_shared[TILE_M][HEAD_DIM];
    __shared__ half pv_shared[TILE_M][HEAD_DIM];
    
    __shared__ float ds[TILE_M];
    __shared__ float ms[TILE_M];

    __shared__ float ms_old[TILE_M];
    __shared__ float ds_old[TILE_M];

    __shared__ float scratch_d[TILE_M*WARPS_PER_ROW];
    __shared__ float scratch_m[TILE_M*WARPS_PER_ROW];

    int64_t n_heads = gridDim.y;
    int64_t batch_idx = blockIdx.x;
    int64_t q_head_idx = blockIdx.y;
    int64_t kv_head_idx = q_head_idx / (n_heads/n_kv_heads);
    int64_t seq_tile = blockIdx.z;

    const half* q_row = q + batch_idx*strides.q0 + q_head_idx*strides.q1 + seq_tile*TILE_M*strides.q2;
    const half* k_head = k + batch_idx*strides.kv0 + kv_head_idx*strides.kv1;
    const half* v_head = v + batch_idx*strides.kv0 + kv_head_idx*strides.kv1;
    
    half* out_row = out + batch_idx*n_heads*L*HEAD_DIM + q_head_idx*L*HEAD_DIM + seq_tile*TILE_M*HEAD_DIM;
    const uint8_t* mask_row = mask + batch_idx*L*L + seq_tile*TILE_M*L; // [B, 1, L, L]

    for (int idx=threadIdx.x; idx<TILE_M; idx+=blockDim.x) {
        ms[idx] = HALF_MIN;
        ds[idx] = 0.0f;
    }

    for (int idx=threadIdx.x; idx<TILE_M*HEAD_DIM; idx+=blockDim.x) {
        out_shared[idx/HEAD_DIM][idx%HEAD_DIM] = __float2half(0.0f);
    }

    __syncthreads();

    for (int n_tile=0; n_tile<L; n_tile+=TILE_N) {
        const half* k_row = k_head + n_tile*strides.kv2;
        const half* v_row = v_head + n_tile*strides.kv2;
        const uint8_t* mask_tile = mask_row + n_tile;

        compute_tile_wmma_f16<TILE_M, TILE_N, HEAD_DIM>(
            HEAD_DIM, 
            strides.q2,
            TILE_N,
            strides.kv2,
            (half*)in_shared, (float*)qk_acc_shared, (half*)kv_shared,
            q_row, (half*)l_shared, k_row
        );

        // apply mask and scale before computing ms and ds
        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;

            l_shared[r][c] = mask_tile[r*L+c] ? __float2half(__half2float(l_shared[r][c]) / sqrtf((float)HEAD_DIM)) : __float2half(HALF_MIN);
        }

        __syncthreads();

        for (int idx=threadIdx.x; idx<TILE_M; idx+=blockDim.x) {
            ms_old[idx] = ms[idx];
            ds_old[idx] = ds[idx];
        }

        __syncthreads();

        update_dm<TILE_M, TILE_N, THR_PER_ROW>(ds, ms, scratch_d, scratch_m, (half*)l_shared);

        __syncthreads();

        for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
            int r = idx/TILE_N;
            int c = idx%TILE_N;
            l_shared[r][c] = __float2half(expf(__half2float(l_shared[r][c])-ms[r]));
        }

        __syncthreads();

        for (int idx=threadIdx.x; idx<TILE_M*HEAD_DIM; idx+=blockDim.x) {
            int r = idx/HEAD_DIM;
            int c = idx%HEAD_DIM;

            out_shared[r][c] = __float2half(__half2float(out_shared[r][c])*expf(ms_old[r]-ms[r]));
        }

        __syncthreads();

        // weird but essentially utilizing V^T^T = V since we are computing l_shared @ V, not V^T
        // can optimize this later by writing a separate impl that handles not writing shared mem to itself (l_shared), 
        // just wanted to reuse some code for now and deal w it later during profiling
        compute_tile_wmma_f16<TILE_M, HEAD_DIM, TILE_N, true, true>(
            TILE_N,
            TILE_N,
            HEAD_DIM,
            strides.kv2,
            (half*)l_shared, (float*)out_acc_shared, (half*)kv_shared,
            (half*)l_shared, (half*)pv_shared, v_row
        );

        __syncthreads();

        for (int idx=threadIdx.x; idx<TILE_M*HEAD_DIM; idx+=blockDim.x) {
            int r = idx/HEAD_DIM;
            int c = idx%HEAD_DIM;
            out_shared[r][c] = __float2half(__half2float(out_shared[r][c]) + __half2float(pv_shared[r][c]));
        }
    }

    for (int idx=threadIdx.x; idx<TILE_M*HEAD_DIM; idx+=blockDim.x) {
        int r = idx/HEAD_DIM;
        int c = idx%HEAD_DIM;
        out_row[r*HEAD_DIM+c] = (ds[r] > 0.0f) ? __float2half(__half2float(out_shared[r][c]) / ds[r]) : __float2half(0.0f);
    }
}

void flash_attn_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    at::Tensor& mask, // [B, 1, L, L]
    at::Tensor& out,     // [B, n_heads, L, HEAD_DIM]
    const at::Tensor& q, // [B, n_heads, L, HEAD_DIM]
    const at::Tensor& k, // [B, n_kv_heads, L, HEAD_DIM]
    const at::Tensor& v  // [B, n_kv_heads, L, HEAD_DIM]
) {
    // validation
    TORCH_INTERNAL_ASSERT(
        (q.device().type() == at::DeviceType::CUDA) &&
        (k.device().type() == at::DeviceType::CUDA) && 
        (v.device().type() == at::DeviceType::CUDA) &&
        (out.device().type() == at::DeviceType::CUDA) &&
        (mask.device().type() == at::DeviceType::CUDA)
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

    constexpr int TILE_M = 16;
    constexpr int HEAD_DIM = 128;
    constexpr int TILE_N = 64;
    constexpr int BLOCK_SIZE = 128;

    dim3 block(BLOCK_SIZE); // TODO: tune this or adjust as needed
    dim3 grid(B, n_heads, (L+TILE_M-1)/TILE_M);

    flash_attn_cuda_impl<TILE_M, TILE_N, HEAD_DIM, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        strides,
        L, n_kv_heads,
        mask_ptr,
        out_ptr, q_ptr, k_ptr, v_ptr
    );
}

template <int TILE_DIM, int BLOCK_SIZE>
__global__ void moe_scoring_cuda_impl(
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    half* __restrict__ out,
    const half* __restrict__ w
) {
    constexpr int THR_PER_ROW = BLOCK_SIZE / TILE_DIM;
    constexpr int WARPS_PER_ROW = (THR_PER_ROW+32-1)/32;

    __shared__ half x_shared[TILE_DIM][TILE_DIM];
    __shared__ half w_shared[TILE_DIM][TILE_DIM+2];
    __shared__ half l_shared[TILE_DIM][TILE_DIM];
    
    __shared__ float ds[TILE_DIM];
    __shared__ float ms[TILE_DIM];
    __shared__ float scratch_d[TILE_DIM*WARPS_PER_ROW];
    __shared__ float scratch_m[TILE_DIM*WARPS_PER_ROW];

    const half* x_row = x + blockIdx.x*TILE_DIM*K;
    half* out_row = out + blockIdx.x*TILE_DIM*N;

    if (threadIdx.x<TILE_DIM) {
        ds[threadIdx.x] = 0.0f;
        ms[threadIdx.x] = -INFINITY;
    }

    __syncthreads();

    // first pass: compute logits, update running d and m, write to out
    for (int n_tile=0; n_tile<N; n_tile+=TILE_DIM) {
        const half* w_row = w + n_tile*K;
        half* out_tile = out_row + n_tile;

        compute_tile_f16<TILE_DIM>(
            M, K, N, 
            K, TILE_DIM, K,
            x_shared, w_shared,
            x_row, (half*)l_shared, w_row
        );

        __syncthreads();

        update_dm<TILE_DIM, TILE_DIM, THR_PER_ROW>(ds, ms, scratch_d, scratch_m, (half*)l_shared);

        for (int idx=threadIdx.x; idx<TILE_DIM*TILE_DIM; idx+=BLOCK_SIZE) {
            int r = idx/TILE_DIM;
            int c = idx%TILE_DIM;
            out_tile[r*N+c] = l_shared[r][c];
        }

        __syncthreads();
    }

    // second pass: apply softmax to each tile w/ global d and m
    for (int64_t n_tile=0; n_tile<N; n_tile+=TILE_DIM) {
        half* out_tile = out_row + n_tile;
        for (int idx=threadIdx.x; idx<TILE_DIM*TILE_DIM; idx+=BLOCK_SIZE) {
            int r = idx/TILE_DIM;
            int c = idx%TILE_DIM;
            l_shared[r][c] = out_tile[r*N+c];
        }
        __syncthreads();

        for (int idx=threadIdx.x; idx<TILE_DIM*TILE_DIM; idx+=BLOCK_SIZE) {
            int r = idx/TILE_DIM;
            int c = idx%TILE_DIM;
            float val = __half2float(l_shared[r][c]);
            val = expf(val-ms[r])/ds[r];
            out_tile[r*N+c] = __float2half(val);
        }
        __syncthreads();
    }
}

// NOTE: ffn_gate_inp seems to be kept in half precision.
// need to handle in compute_tiles and elsewhere properly
void moe_scoring_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x, // [B,L,hidden_dim]
    at::Tensor& out, // [B,L,n_experts]
    const at::Tensor& w // [n_experts, hidden_dim]
) {


    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);

    TORCH_CHECK(x.numel() / x.size(-1) == out.numel() / out.size(-1));
    TORCH_CHECK(w.size(0) == out.size(-1));
    TORCH_CHECK(w.size(1) == x.size(-1));

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());

    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t hidden_dim = x.size(-1);
    int64_t n_exps = out.size(-1);

    int64_t M = B*L;
    int64_t K = hidden_dim;
    int64_t N = n_exps;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    constexpr int TILE_DIM=32;
    constexpr int BLOCK_SIZE=1024;

    dim3 grid((M+TILE_DIM-1)/TILE_DIM);
    dim3 block(BLOCK_SIZE);

    moe_scoring_cuda_impl<TILE_DIM, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        M, K, N,
        x_ptr, out_ptr, w_ptr
    );
}

template <int TILE_M, int TILE_N, int TILE_K, int BLOCK_SIZE>
__global__ void swiglu_f16_cuda_impl(
    int64_t M, int64_t K_GU, int64_t N_GU,
    const half* __restrict__ in,
    half* __restrict__ hb,
    half* __restrict__ hb2,
    const half* __restrict__ ws_up,
    const half* __restrict__ ws_gate
) {

    __shared__ half in_shared[TILE_M][TILE_K];
    __shared__ float out_shared[TILE_M][TILE_N];
    __shared__ half w_shared[TILE_N][TILE_K];
    __shared__ half hb_shared[TILE_M][TILE_N];
    __shared__ half hb2_shared[TILE_M][TILE_N];

    const half* in_row = in + blockIdx.y*TILE_M*K_GU;
    half* hb2_tile = hb2 + blockIdx.x*M*N_GU + blockIdx.y*TILE_M*N_GU + blockIdx.z*TILE_N;

    const half* ws_gate_row = ws_gate + blockIdx.x*N_GU*K_GU + blockIdx.z*TILE_N*K_GU;
    const half* ws_up_row = ws_up + blockIdx.x*N_GU*K_GU + blockIdx.z*TILE_N*K_GU;

    // gate proj stored in hb_row
    compute_tile_wmma_f16<TILE_M, TILE_N, TILE_K>(
        K_GU, 
        K_GU,
        TILE_N,
        K_GU,
        (half*)in_shared, (float*)out_shared, (half*)w_shared, 
        in_row, (half*)hb_shared, ws_gate_row
    );
    
    __syncthreads();

    // up proj stored in hb2
    compute_tile_wmma_f16<TILE_M, TILE_N, TILE_K>(
        K_GU, 
        K_GU,
        TILE_N,
        K_GU,
        (half*)in_shared, (float*)out_shared, (half*)w_shared, 
        in_row, (half*)hb2_shared, ws_up_row
    );

    __syncthreads();

    // apply swish to hb (swish(x) = x*sigmoid(beta*x), beta taken to be 1 here)
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=BLOCK_SIZE) {
        int r = idx/TILE_N;
        int c = idx%TILE_N;
        float val = __half2float(hb_shared[r][c]);
        val = val/(1.0f+expf(-val));
        hb_shared[r][c] = __float2half(val);
    }
    
    __syncthreads();

    // element-wise mult of hb_shared and hb2_shared
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=BLOCK_SIZE) {
        int r = idx/TILE_N;
        int c = idx%TILE_N;
        float hbv = __half2float(hb_shared[r][c]);
        float hb2v = __half2float(hb2_shared[r][c]);
        hb2_shared[r][c] = __float2half(hbv*hb2v);
    }

    __syncthreads();

    // write tile to global hb2
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
        int r = idx/TILE_N;
        int c = idx%TILE_N;
        hb2_tile[r*N_GU+c] = hb2_shared[r][c];
    }
}

template <int TILE_M, int TILE_N, int TILE_K, int BLOCK_SIZE>
__global__ void swiglu_quant_cuda_impl(
    int up_qtype_int, int gate_qtype_int,
    int up_qblock_size, int gate_qblock_size,
    int up_qtype_size, int gate_qtype_size,
    int64_t M, int64_t K_GU, int64_t N_GU,
    const half* __restrict__ in,
    half* __restrict__ hb,
    half* __restrict__ hb2,
    const uint8_t* __restrict__ ws_up,
    const uint8_t* __restrict__ ws_gate
) {

    __shared__ half in_shared[TILE_M][TILE_K];
    __shared__ float out_shared[TILE_M][TILE_N];
    __shared__ half w_shared[TILE_N][TILE_K];
    __shared__ half hb_shared[TILE_M][TILE_N];
    __shared__ half hb2_shared[TILE_M][TILE_N];

    const half* in_row = in + blockIdx.y*TILE_M*K_GU;
    half* hb2_tile = hb2 + blockIdx.x*M*N_GU + blockIdx.y*TILE_M*N_GU + blockIdx.z*TILE_N;

    int64_t exp_gate_size = N_GU*(K_GU/gate_qblock_size)*gate_qtype_size;
    int64_t exp_up_size = N_GU*(K_GU/up_qblock_size)*up_qtype_size;

    const uint8_t* ws_gate_row = ws_gate + blockIdx.x*exp_gate_size + blockIdx.z*TILE_N*(K_GU/gate_qblock_size)*gate_qtype_size;
    const uint8_t* ws_up_row = ws_up + blockIdx.x*exp_up_size + blockIdx.z*TILE_N*(K_GU/up_qblock_size)*up_qtype_size;

    // gate proj stored in hb_row
    compute_tile_quant<TILE_M, TILE_N, TILE_K>(
        gate_qtype_int, gate_qblock_size, gate_qtype_size,
        K_GU, 
        K_GU,
        TILE_N,
        K_GU,
        (half*)in_shared, (float*)out_shared, (half*)w_shared, 
        in_row, (half*)hb_shared, ws_gate_row
    );
    
    // up proj stored in hb2
    compute_tile_quant<TILE_M, TILE_N, TILE_K>(
        up_qtype_int, up_qblock_size, up_qtype_size,
        K_GU, 
        K_GU,
        TILE_N,
        K_GU,
        (half*)in_shared, (float*)out_shared, (half*)w_shared, 
        in_row, (half*)hb2_shared, ws_up_row
    );

    // apply swish to hb (swish(x) = x*sigmoid(beta*x), beta taken to be 1 here)
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=BLOCK_SIZE) {
        int r = idx/TILE_N;
        int c = idx%TILE_N;
        float val = __half2float(hb_shared[r][c]);
        val = val/(1.0f+expf(-val));
        hb_shared[r][c] = __float2half(val);
    }
    
    __syncthreads();

    // element-wise multiplication of hb_shared and hb2_shared
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=BLOCK_SIZE) {
        int r = idx/TILE_N;
        int c = idx%TILE_N;
        float hbv = __half2float(hb_shared[r][c]);
        float hb2v = __half2float(hb2_shared[r][c]);
        hb2_shared[r][c] = __float2half(hbv*hb2v);
    }

    __syncthreads();

    // write tile to global hb2
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=blockDim.x) {
        int64_t r = idx/TILE_N;
        int64_t c = idx%TILE_N;
        hb2_tile[r*N_GU+c] = hb2_shared[r][c];
    }
}

// elementwise multiplication of up proj tile with swiglu (tile of swish(gate(x))), then apply downproj
void ffn_cuda(
    int64_t up_qtype_int, int64_t gate_qtype_int, int64_t down_qtype_int,
    int64_t up_qblock_size, int64_t gate_qblock_size, int64_t down_qblock_size,
    int64_t up_qtype_size, int64_t gate_qtype_size, int64_t down_qtype_size,
    const at::Tensor& in, // [B,L,hidden_dim]
    at::Tensor& out, // [n_local_exps,B,L,hidden_dim]
    at::Tensor& hb, // [n_local_exps,B,L,mlp_dim]
    at::Tensor& hb2, // [n_local_exps,B,L,mlp_dim]
    const at::Tensor& ws_up, // [n_local_exps, mlp_dim, hidden_dim]
    const at::Tensor& ws_gate, // // [n_local_exps, mlp_dim, hidden_dim]
    const at::Tensor& ws_down // [n_local_exps, hidden_dim, mlp_dim]
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

        constexpr int TILE_DIM=32;
        constexpr int BLOCK_SIZE=1024;
        
        dim3 block(BLOCK_SIZE); // TODO: tune this or adjust as needed
        dim3 grid_swiglu(n_local_exps, (M+TILE_DIM-1)/TILE_DIM, (N_GU+TILE_DIM-1)/TILE_DIM);
        dim3 grid_down_exp((M+TILE_DIM-1)/TILE_DIM, (N_DOWN+TILE_DIM-1)/TILE_DIM);

        // swiglu
        swiglu_f16_cuda_impl<TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_SIZE><<<grid_swiglu, block, 0, stream>>>(
            M, K_GU, N_GU,
            in_ptr, hb_ptr, hb2_ptr,
            ws_up_ptr, ws_gate_ptr
        );

        // launch down-proj kernel per expert
        for (int e=0; e<n_local_exps; e++) {
            half* hb2_exp = hb2_ptr + e*M*N_GU;
            half* out_exp = out_ptr + e*M*N_DOWN;
            const half* ws_down_exp = ws_down_ptr + e*N_DOWN*K_DOWN;
            
            matmul_f16_cuda_impl<TILE_DIM><<<grid_down_exp, block, 0, stream>>>(
                M, K_DOWN, N_DOWN,
                hb2_exp, out_exp, ws_down_exp
            );
        }
    } else {
        const uint8_t* ws_up_ptr = ws_up.data_ptr<uint8_t>();
        const uint8_t* ws_gate_ptr = ws_gate.data_ptr<uint8_t>();
        const uint8_t* ws_down_ptr = ws_down.data_ptr<uint8_t>();

        constexpr int TILE_SIZE=32;
        constexpr int TILE_K=256;
        constexpr int BLOCK_SIZE=128;
        
        dim3 block(BLOCK_SIZE); // TODO: tune this or adjust as needed
        dim3 grid_swiglu(n_local_exps, (M+TILE_SIZE-1)/TILE_SIZE, (N_GU+TILE_SIZE-1)/TILE_SIZE);
        dim3 grid_down_exp((M+TILE_SIZE-1)/TILE_SIZE, (N_DOWN+TILE_SIZE-1)/TILE_SIZE);

        // swiglu
        swiglu_quant_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K, BLOCK_SIZE><<<grid_swiglu, block, 0, stream>>>(
            up_qtype_int, gate_qtype_int,
            up_qblock_size, gate_qblock_size,
            up_qtype_size, gate_qtype_size,
            M, K_GU, N_GU,
            in_ptr, hb_ptr, hb2_ptr,
            ws_up_ptr, ws_gate_ptr
        );

        // launch down-proj kernel per expert
        for (int e=0; e<n_local_exps; e++) {
            half* hb2_exp = hb2_ptr + e*M*N_GU;
            half* out_exp = out_ptr + e*M*N_DOWN;
            const uint8_t* ws_down_exp = ws_down_ptr + e*N_DOWN*(K_DOWN/down_qblock_size)*down_qtype_size;
            
            matmul_quant_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K><<<grid_down_exp, block, 0, stream>>>(
                down_qtype_int, down_qblock_size, down_qtype_size,
                M, K_DOWN, N_DOWN,
                hb2_exp, out_exp, ws_down_exp
            );
        }
    }
}

TORCH_LIBRARY(minfer, m) {
    m.def("dequant(int qtype, Tensor x, Tensor(a!) y, int qblock_size, int qtype_size) -> ()");
    m.def("quant(int qtype, Tensor x, Tensor(a!) y, int qblock_size, int qtype_size) -> ()");
    m.def("rmsnorm(float eps, Tensor input, Tensor(a!) out, Tensor w) -> ()");
    m.def("il_rope(int rotary_dim, int start_pos, float freq_base, Tensor(a!) x) -> ()");
    m.def("neox_rope(int rotary_dim, int start_pos, float freq_base, Tensor(a!) x) -> ()");
    m.def("matmul(int qtype_int, int qblock_size, int qtype_size, Tensor x, Tensor(a!) out, Tensor w) -> ()");
    m.def("embed(int qtype_int, int qblock_size, int qtype_size, Tensor(a!) x, Tensor token_ids, Tensor w) -> ()");
    m.def("qkv(int q_qtype_int, int k_qtype_int, int v_qtype_int, int q_qblock_size, int k_qblock_size, int v_qblock_size, int q_qtype_size, int k_qtype_size, int v_qtype_size, Tensor x, Tensor(a!) q_out, Tensor(a!) k_out, Tensor(a!) v_out, Tensor wq, Tensor wk, Tensor wv) -> ()");
    m.def("flash_attn(int qtype_int, int qblock_size, int qtype_size, Tensor mask, Tensor(a!) out, Tensor q, Tensor k, Tensor v) -> ()");
    m.def("moe_scoring(int qtype_int, int qblock_size, int qtype_size, Tensor x, Tensor(a!) out, Tensor w) -> ()");
    m.def("ffn(int up_qtype_int, int gate_qtype_int, int down_qtype_int, int up_qblock_size, int gate_qblock_size, int down_qblock_size, int up_qtype_size, int gate_qtype_size, int down_qtype_size, Tensor input, Tensor(a!) out, Tensor(a!) hb, Tensor(a!) hb2, Tensor ws_up, Tensor ws_gate, Tensor ws_down) -> ()");
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