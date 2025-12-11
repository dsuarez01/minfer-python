#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cstdint>

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>

#include <cassert>
#include <algorithm>

#include "quants_impl.cuh"
#include "impl_common.hpp"

using namespace nvcuda;

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
static __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

static __device__ float blockreduce_sum(float* vs, float v, int tid) {
    v = warp_reduce_sum(v);
    if (tid%32 == 0) vs[tid/32] = v;
    __syncthreads();
    v = vs[tid%32];
    return warp_reduce_sum(v);
}

static __device__ float blockreduce_max(float* vs, float v, int tid) {
    v = warp_reduce_max(v);
    if (tid%32 == 0) vs[tid/32] = v;
    __syncthreads();
    v = vs[tid%32];
    return warp_reduce_max(v);
}

// helper for moe scoring and flash attn, where (online) softmax is fused and taken per-tile
// first thread in warps responsible for writing warp vals (d and m) [per-row]
// threads in first warp responsible for reducing over warp vals [per-row]
template <int THR_PER_ROW>
static __device__ float row_reduce_max(float* vs, float v, int row_in_tile, int thr_in_row) {
    v = warp_reduce_max(v);

    if constexpr(THR_PER_ROW/32 > 1) {
        int warp_in_row = thr_in_row / 32;
        if (thr_in_row % 32 == 0) vs[row_in_tile*(THR_PER_ROW/32)+warp_in_row]=v;
        __syncthreads();

        if (warp_in_row == 0 && thr_in_row < THR_PER_ROW/32) {
            v = vs[row_in_tile*(THR_PER_ROW/32) + thr_in_row];
            v = warp_reduce_max(v);
        }
    }
    return v;
}

template <int THR_PER_ROW>
static __device__ float row_reduce_sum(float* vs, float v, int row_in_tile, int thr_in_row) {
    v = warp_reduce_sum(v);

    if constexpr(THR_PER_ROW/32 > 1) {
        int warp_in_row = thr_in_row / 32;
        if (thr_in_row % 32 == 0) vs[row_in_tile*(THR_PER_ROW/32)+warp_in_row]=v;
        __syncthreads();

        if (warp_in_row == 0 && thr_in_row < THR_PER_ROW/32) {
            v = vs[row_in_tile*(THR_PER_ROW/32) + thr_in_row];
            v = warp_reduce_sum(v);
        }
    }
    return v;
}

template <int TILE_M, int TILE_N, int THR_PER_ROW>
static __device__ int compute_dm_tile(
    float* __restrict__ tiles_d,
    float* __restrict__ tiles_m,
    int* __restrict__ row_visit_cnt,
    const half* __restrict__ tile,
    int M
) {
    constexpr int TILE_SIZE = TILE_M;
    int row_in_tile = threadIdx.x / THR_PER_ROW;
    int thr_in_row = threadIdx.x % THR_PER_ROW;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (row_in_tile >= TILE_M) return -1;

    int global_row = blockIdx.x*TILE_M+row_in_tile;

    if (global_row >= M) return -1;

    __shared__ float tile_rows_m[TILE_M*(THR_PER_ROW/32)];
    __shared__ float tile_rows_d[TILE_M*(THR_PER_ROW/32)];

    // compute tile's m vals per row, write to tiles_m
    float ml = -INFINITY;
    for (int col=thr_in_row; col<TILE_N; col+=THR_PER_ROW) {
        ml = fmaxf(ml, __half2float(tile[row_in_tile*TILE_N+col])); 
    }

    tiles_m[blockIdx.x*gridDim.y*TILE_SIZE + blockIdx.y*TILE_SIZE + row_in_tile] = row_reduce_max<THR_PER_ROW>(tile_rows_m, ml, row_in_tile, thr_in_row);

    // compute tile's d vals per row, write to tiles_d
    float dl = 0.0f;
    for (int col=thr_in_row; col<TILE_N; col+=THR_PER_ROW) {
        dl += expf(__half2float(tile[row_in_tile*TILE_N+col] - ml));
    }

    tiles_d[blockIdx.x*gridDim.y*TILE_SIZE + blockIdx.y*TILE_SIZE + row_in_tile] = row_reduce_sum<THR_PER_ROW>(tile_rows_d, dl, row_in_tile, thr_in_row);

    int cnt = -1;
    if (threadIdx.x == 0) {
        cnt = atomicAdd(&row_visit_cnt[global_row],1);
    }
    return cnt;
}

template <int TILE_M>
static __device__ void reduce_dm_tiles (
    float* __restrict__ tiles_d,
    float* __restrict__ tiles_m,
    float* scratch,
    float* rows_d,
    float* rows_m
 ) {
    int thr_per_row = blockDim.x / TILE_M;

    float* scratch_d = scratch;
    float* scratch_m = scratch + gridDim.y;

    int row_in_tile = threadIdx.x / thr_per_row;

    // load tiles d and m vals to shared mem
    for (int tile_col = threadIdx.x; tile_col<gridDim.y; tile_col += blockDim.x) {
        int idx = blockIdx.x*gridDim.y*TILE_M + tile_col * TILE_M + row_in_tile;
        scratch_d[tile_col] = tiles_d[idx];
        scratch_m[tile_col] = tiles_m[idx];
    }
    __syncthreads();

    // tree reduction
    for (int stride = gridDim.y/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float m1 = scratch_m[threadIdx.x], m2 = scratch_m[threadIdx.x+stride];
            float d1 = scratch_d[threadIdx.x], d2 = scratch_d[threadIdx.x+stride];
            float m = fmaxf(m1,m2);
            scratch_d[threadIdx.x] = d1*expf(m1-m) + d2*expf(m2-m);
            scratch_m[threadIdx.x] = m;
        }
        __syncthreads();
    }

    if (threadIdx.x < TILE_M) {
        rows_d[row_in_tile] = scratch_d[0];
        rows_m[row_in_tile] = scratch_m[0];
    }
}

// only one block in row of tiles should call this
template <int TILE_M>
static __device__ void apply_dm(
    half* __restrict__ out,
    float* __restrict__ rows_d,
    float* __restrict__ rows_m,
    int N
) {
    int thr_per_row = blockDim.x / TILE_M; 
    int row_in_tile = threadIdx.x / thr_per_row;

    float d = rows_d[row_in_tile];
    float m = rows_m[row_in_tile];

    int global_row = blockIdx.x*TILE_M + row_in_tile;

    half* row = out + global_row*N;
    
    for (int col=threadIdx.x; col<N; col+=blockDim.x) {
        row[col] = __float2half(expf(__half2float(row[col])-m)/d);
    }
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
        default: assert(false && "Unsupported dtype"); // this gets compiled out in non-debug builds...
    }
}

__global__ void rmsnorm_cuda_impl(
    int dim,
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

void rmsnorm_cuda(double eps, const at::Tensor& in, at::Tensor& out, const at::Tensor& w) {
    // checks
    TORCH_CHECK(in.sizes().equals(out.sizes()));
    TORCH_CHECK(in.dim() == 3 || in.dim() == 4);
    TORCH_CHECK(in.size(-1) == w.size(0) || in.size(-1) % w.size(0) == 0); // per-head or per-entire vector
    
    TORCH_CHECK(in.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);
    
    TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    
    TORCH_CHECK(in.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    const half* in_ptr = reinterpret_cast<const half*>(in.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());

    // handles both [B,L,D] and [B,L,n_heads,head_dim]
    int dim = w.size(0);
    int n_blocks = in.numel() / dim;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_cuda_impl<<<n_blocks, 1024, 0, stream>>>(dim, eps, in_ptr, out_ptr, w_ptr);
}

__global__ void il_rope_cuda_impl(
    int n_heads,
    int rotary_dim,
    int head_dim,
    int64_t start_pos,
    float freq_base,
    half* __restrict__ x
) {
    int head_idx = blockIdx.z % n_heads;
    int pair_idx = (blockIdx.z / n_heads) * 32 + threadIdx.x;
    int pos = start_pos + blockIdx.y;
    
    if (pair_idx >= rotary_dim / 2) return;

    half* x_head = x + (blockIdx.x * gridDim.y * n_heads + blockIdx.y * n_heads + head_idx)*head_dim;

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
    TORCH_CHECK(rotary_dim <= x.size(3));
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(x.is_contiguous());

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    int B = x.size(0);
    int L = x.size(1);
    int n_heads = x.size(2);
    int head_dim = x.size(3);

    dim3 grid(B, L, n_heads*(rotary_dim/2+31)/32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    il_rope_cuda_impl<<<grid, 32, 0, stream>>>(n_heads, rotary_dim, head_dim, start_pos, freq_base, x_ptr);
}

__global__ void neox_rope_cuda_impl(
    int n_heads,
    int rotary_dim,
    int head_dim,
    int64_t start_pos,
    float freq_base,
    half* __restrict__ x
) {
    int head_idx = blockIdx.z % n_heads;
    int pair_idx = (blockIdx.z / n_heads) * 32 + threadIdx.x;
    int pos = start_pos + blockIdx.y;
    
    if (pair_idx >= rotary_dim / 2) return;

    half* x_head = x + (blockIdx.x * gridDim.y * n_heads + blockIdx.y * n_heads + head_idx)*head_dim;

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
    TORCH_CHECK(rotary_dim <= x.size(3));
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(x.is_contiguous());

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);

    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    int B = x.size(0);
    int L = x.size(1);
    int n_heads = x.size(2);
    int head_dim = x.size(3);

    dim3 grid(B, L, n_heads*(rotary_dim/2+31)/32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    neox_rope_cuda_impl<<<grid, 32, 0, stream>>>(n_heads, rotary_dim, head_dim, start_pos, freq_base, x_ptr);
}

// helper to compute TILE_M x TILE_N portion of result
template <int TILE_M, int TILE_N, int TILE_K, bool IS_HALF>
static __device__ void compute_tile(
    int qtype_int,
    int qblock_size,
    int qtype_size,
    int64_t K,
    int64_t N,
    int warp_id,
    int tiles_per_warp,
    half (&x_shared)[TILE_M][TILE_K],
    half (&w_shared)[TILE_K][TILE_N],
    half* __restrict__ x,
    half* __restrict__ out,
    const uint8_t* __restrict__ w
) {

    constexpr int SUBTILE_DIM = TILE_M / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;

    // split work over the subtiles (arranged in a grid of TILE_SIZE/16xTILE_SIZE/16), each of size 16x16.
    for (int i=0; i<tiles_per_warp; ++i) {
        int tile_id = warp_id * tiles_per_warp + i;
        
        // K-dimension processed in TILE_K chunks
        for (int64_t k_tile=0; k_tile<K; k_tile+=TILE_K) {
            // coop koad x tile to shared mem
            for (int idx = threadIdx.x; idx < TILE_M*TILE_K; idx += blockDim.x) {
                int ml = idx / TILE_K;
                int kl = idx % TILE_K;
                x_shared[ml][kl] = x[(blockIdx.x * TILE_M + ml) * K + (k_tile + kl)];
            }

            if constexpr(IS_HALF) { // not actual branching
                const half* w_half = reinterpret_cast<const half*>(w);

                for (int idx = threadIdx.x; idx < TILE_K*TILE_N; idx += blockDim.x) {
                    int kl = idx / TILE_N;
                    int nl = idx % TILE_N;
                    w_shared[kl][nl] = w_half[(blockIdx.y*TILE_N+nl)*K + (k_tile+kl)];
                }
            } else {

                for (int idx = threadIdx.x; idx < (TILE_K/qblock_size)*TILE_N; idx += blockDim.x) {
                    int kb = idx / TILE_N;
                    int nl = idx % TILE_N;
                    const uint8_t* w_block = w + ((blockIdx.y*TILE_N+nl)*(K/qblock_size) + (k_tile/qblock_size + kb))*qtype_size;
                    dequant_block(qtype_int, TILE_N, threadIdx.x % qblock_size, w_shared[kb*qblock_size]+nl, w_block);
                }
            }
            
            __syncthreads();

            int m_subtile = (tile_id/SUBTILE_DIM)*16;
            int n_subtile = (tile_id%SUBTILE_DIM)*16;

            // process TILE_K with WMMA in chunks of 16
            for (int k_subtile=0; k_subtile<TILE_K; k_subtile+=16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, x_shared[m_subtile]+k_subtile, TILE_K);
                wmma::load_matrix_sync(b_frag, w_shared[k_subtile]+n_subtile, TILE_N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            __syncthreads();
        }

        wmma::store_matrix_sync(
            out + (blockIdx.x*TILE_M+m_subtile)*N + (blockIdx.y*TILE_N+n_subtile),
            acc_frag,
            N,
            wmma::mem_row_major
        );
        wmma::fill_fragment(acc_frag, __float2half(0.0f));
    }
}

// with the current settings there should be (64/16)^2 = 16 subtiles, 
// 512 thrs / 32 = 16 warps, so 16/16=1 warp per subtile
// note that there is 48 KB shared mem constraint on Volta
// (anyone else using this can adjust params as needed)
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void matmul_cuda_impl(
    int qtype_int,
    int qblock_size,
    int qtype_size,
    int64_t M, int64_t K, int64_t N,
    const half* __restrict__ x,
    half* __restrict__ out,
    const uint8_t* __restrict__ w
) {
    auto qtype = static_cast<GGMLQuantizationType(qtype_int);

    bool is_half = qtype == GGMLQuantizationType::F16;

    constexpr int NUM_SUBTILES = (TILE_M/16)*(TILE_N/16); // wmma operates on 16x16 subtiles

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ half w_shared[TILE_K][TILE_N];

    if (is_half) {
        compute_tile<TILE_M, TILE_N, TILE_K, true>(
            qtype_int,
            qblock_size,
            qtype_size,
            K,
            N,
            warp_id,
            tiles_per_warp,
            x_shared,
            w_shared,
            x,
            out,
            w
        );
    } else {
        compute_tile<TILE_M, TILE_N, TILE_K, false>(
            qtype_int,
            qblock_size,
            qtype_size,
            K,
            N,
            warp_id,
            tiles_per_warp,
            x_shared,
            w_shared,
            x,
            out,
            w
        );
    }
}

void matmul_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& w
) {

    TORCH_CHECK(is_valid_qtype(qtype_int));

    TORCH_CHECK(w.size(1) == (x.size(-1) / qblock_size) * qtype_size);
    TORCH_CHECK(w.size(0) == out.size(-1));
    TORCH_CHECK(x.numel() / x.size(-1) == out.numel() / out.size(-1));

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kByte);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const uint8_t* w_ptr = w.data_ptr<uint8_t>();

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t N = w.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int TILE_SIZE = 64;
    constexpr int TILE_K = 256;
    dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    dim3 block(512);

    matmul_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K><<<grid, block, 0, stream>>>(qtype_int, qblock_size, qtype_size, M, K, N, x_ptr, out_ptr, w_ptr);
}

__global__ void embed_cuda_impl(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    int64_t b, // bytes per row
    int64_t k, // dequant elems per row
    half* __restrict__ x,
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ w
) {
    int64_t B = gridDim.x;
    int64_t L = gridDim.y;

    int64_t iB = blockIdx.x;
    int64_t iL = blockIdx.y;
    int64_t block_in_row = blockIdx.z;
    int64_t token_id = token_ids[iB*L+iL];

    const uint8_t* w_block = w + token_id*b + block_in_row*qtype_size;
    T* x_block = x + iB*L*k + iL*k + block_in_row*qblock_size;

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
    const at::Tensor& w // [vocab_size,bytes_per_row]
) {
    
    TORCH_CHECK(token_ids.dim() == 2);
    TORCH_CHECK(w.dim() == 2);
    TORCH_CHECK(x.dim() == 3);

    TORCH_CHECK(token_ids.dtype() == at::kLong);
    TORCH_CHECK(w.dtype() == at::kByte);
    TORCH_CHECK(x.dtype() == at::kHalf);

    TORCH_CHECK(x.size(0) == token_ids.size(0) && x.size(1) == token_ids.size(1));

    TORCH_CHECK(token_ids.is_contiguous());
    TORCH_CHECK(w.is_contiguous());
    TORCH_CHECK(x.is_contiguous());

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    const int64_t* token_ids_ptr = token_ids.data_ptr<int64_t>();
    const uint8_t* w_ptr = w.data_ptr<uint8_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t n_qblocks_per_row = x.size(-1) / qblock_size;
    dim3 grid(x.size(0), x.size(1), n_qblocks_per_row);

    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());
    embed_cuda_impl<<<grid, qblock_size, 0, stream>>>(qtype_int, qblock_size, qtype_size, w.size(-1), x.size(-1), x_ptr, token_ids_ptr, w_ptr);
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void qkv_cuda_impl(
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

    auto q_qtype = static_cast<GGMLQuantizationType>(q_qtype_int);
    auto k_qtype = static_cast<GGMLQuantizationType>(k_qtype_int);
    auto v_qtype = static_cast<GGMLQuantizationType>(v_qtype_int);

    bool q_is_half = q_qtype == GGMLQuantizationType::F16;
    bool k_is_half = k_qtype == GGMLQuantizationType::F16;
    bool v_is_half = v_qtype == GGMLQuantizationType::F16;

    constexpr int NUM_SUBTILES = (TILE_M/16)*(TILE_N/16); // wmma operates on 16x16 subtiles

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ half w_shared[TILE_K][TILE_N];


    if (blockIdx.y*TILE_N < N_Q) {
        if (q_is_half) {
            compute_tile<TILE_M, TILE_N, TILE_K, true>(
                q_qtype_int, q_qblock_size, q_qtype_size,
                K, N_Q,
                warp_id, tiles_per_warp,
                x_shared, w_shared,
                x, q_out, wq
            );
        } else {
            compute_tile<TILE_M, TILE_N, TILE_K, false>(
                q_qtype_int, q_qblock_size, q_qtype_size,
                K, N_Q,
                warp_id, tiles_per_warp,
                x_shared, w_shared,
                x, q_out, wq
            );
        }
    }

    if (blockIdx.y*TILE_N < N_KV) {
        if (k_is_half) {
            compute_tile<TILE_M, TILE_N, TILE_K, true>(
                k_qtype_int, k_qblock_size, k_qtype_size,
                K, N_KV,
                warp_id, tiles_per_warp,
                x_shared, w_shared,
                x, k_out, wk
            );
        } else {
            compute_tile<TILE_M, TILE_N, TILE_K, false>(
                k_qtype_int, k_qblock_size, k_qtype_size,
                K, N_KV,
                warp_id, tiles_per_warp,
                x_shared, w_shared,
                x, k_out, wk
            );
        }

        if (v_is_half) {
            compute_tile<TILE_M, TILE_N, TILE_K, true>(
                v_qtype_int, v_qblock_size, v_qtype_size,
                K, N_KV,
                warp_id, tiles_per_warp,
                x_shared, w_shared,
                x, v_out, wv
            );
        } else {
            compute_tile<TILE_M, TILE_N, TILE_K, false>(
                v_qtype_int, v_qblock_size, v_qtype_size,
                K, N_KV,
                warp_id, tiles_per_warp,
                x_shared, w_shared,
                x, v_out, wv
            );
        }
    }
}

// NOTE: this kernel assumes that wq,wk,wv are actually quantized and not half
void qkv_cuda(
    int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
    int64_t q_qblock_size, int64_t k_qblock_size, int64_t v_qblock_size,
    int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
    const at::Tensor& x, // [B, L, hidden_dim]
    at::Tensor& q_out, // [B, L, q_dim]
    at::Tensor& k_out, // [B, L, kv_dim]
    at::Tensor& v_out, // [B, L, kv_dim]
    const at::Tensor& wq, // [hidden_dim, q_dim]
    const at::Tensor& wk, // [hidden_dim, kv_dim]
    const at::Tensor& wv // [hidden_dim, kv_dim]
) {
    // validation

    //dim
    TORCH_CHECK(
        (x.dim() == 3) && 
        (q_out.dim() == 3) && (k_out.dim() == 3) && (v_out.dim() == 3) &&
        (wq.dim() == 2) && (wk.dim() == 2) && (wv.dim() == 2)
    );

    //size
    TORCH_CHECK(
        (wq.size(1) == (x.size(-1) / q_qblock_size) * q_qtype_size) &&
        (wk.size(1) == (x.size(-1) / k_qblock_size) * k_qtype_size) &&
        (wv.size(1) == (x.size(-1) / v_qblock_size) * v_qtype_size) &&
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


    //dtypes
    TORCH_CHECK(
        (x.dtype() == at::kHalf) && 
        (q_out.dtype() == at::kHalf) && (k_out.dtype() == at::kHalf) && (v_out.dtype() == at::kHalf) &&
        (wq.dtype() == at::kByte) && (wk.dtype() == at::kByte) && (wv.dtype() == at::kByte)
    );

    TORCH_CHECK(
        x.is_contiguous() &&
        q_out.is_contiguous() && k_out.is_contiguous() && v_out.is_contiguous() &&
        wq.is_contiguous() && wk.is_contiguous() && wv.is_contiguous()
    );

    //qblock sizes
    TORCH_CHECK(
        (q_qblock_size == 256 || q_qblock_size == 32) && 
        (k_qblock_size == 256 || k_qblock_size == 32) && 
        (v_qblock_size == 256 || v_qblock_size == 32)
    );

    TORCH_INTERNAL_ASSERT(
        (x.device().type() == at::DeviceType::CUDA) &&
        (q_out.device().type() == at::DeviceType::CUDA) &&
        (k_out.device().type() == at::DeviceType::CUDA) && 
        (v_out.device().type() == at::DeviceType::CUDA) &&
        (wq.device().type() == at::DeviceType::CUDA) &&
        (wk.device().type() == at::DeviceType::CUDA) &&
        (wv.device().type() == at::DeviceType::CUDA)
    );

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    
    half* q_out_ptr = reinterpret_cast<half*>(q_out.data_ptr<at::Half>());
    half* k_out_ptr = reinterpret_cast<half*>(k_out.data_ptr<at::Half>());
    half* v_out_ptr = reinterpret_cast<half*>(v_out.data_ptr<at::Half>());

    const uint8_t* wq_ptr = wq.data_ptr<uint8_t>();
    const uint8_t* wk_ptr = wk.data_ptr<uint8_t>();
    const uint8_t* wv_ptr = wv.data_ptr<uint8_t>();

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t N_Q = wq.size(0);
    int64_t N_KV = wk.size(0);
    int64_t N = std::max(N_Q, N_KV);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    constexpr int TILE_SIZE = 64;
    constexpr int TILE_K = 256;
    
    dim3 block(512); // TODO: tune this or adjust as needed
    dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    qkv_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K><<<grid, block, 0, stream>>>(
        q_qtype_int, k_qtype_int, v_qtype_int,
        q_qblock_size, k_qblock_size, v_qblock_size, 
        q_qtype_size, k_qtype_size, v_qtype_size,
        M, K, N_Q, N_KV,
        x,
        q_out, k_out, v_out,
        wq, wk, wv
    );
}

void flash_attn_cuda() {
    TORCH_CHECK(false, "flash_attn not implemented");
}

// little difference to matmul_cuda_impl, except computing softmax with tile at end (online version)
template <int TILE_M, int TILE_N, int TILE_K, int BLOCK_SIZE>
__global__ void moe_scoring_cuda_impl(
    int qtype_int,
    int qblock_size,
    int qtype_size,
    int64_t M, int64_t K, int64_t N,
    float* __restrict__ tiles_d,
    float* __restrict__ tiles_m,
    int* __restrict__ row_visit_cnt,
    const half* __restrict__ x,
    half* __restrict__ out,
    const uint8_t* __restrict__ w
) {

    constexpr int TILE_SIZE = TILE_M;
    constexpr int THR_PER_ROW = BLOCK_SIZE / TILE_M;
    constexpr int NUM_SUBTILES = (TILE_M/16)*(TILE_N/16); // wmma operates on 16x16 subtiles
    constexpr int num_warps = BLOCK_SIZE / 32;
    constexpr int tiles_per_warp = NUM_SUBTILES / num_warps;

    int warp_id = threadIdx.x / 32;
    int row_in_tile = threadIdx.x / THR_PER_ROW;

    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ half w_shared[TILE_K][TILE_N];

    compute_tile<TILE_M, TILE_N, TILE_K, true>(
        qtype_int, qblock_size, qtype_size,
        K, N,
        warp_id, tiles_per_warp,
        x_shared, w_shared,
        x, out, w
    );

    // TODO: update(?) online softmax using computed tile here
    int cnt = compute_dm_tile<TILE_SIZE, THR_PER_ROW>(
        tiles_d, tiles_m, row_visit_cnt,
        out,
        M,
    );

    __shared__ float rows_d[TILE_M];
    __shared__ float rows_m[TILE_M];

    extern __shared__ float reduce_scratch[]; 
    
    // last tile to finish does:
    // does tree reduction to get one d and m per row
    // applies softmax over TILE_M x N subsection using d and m

    if (cnt == (gridDim.y-1)) {
        reduce_dm_tiles<TILE_M>(
            tiles_d, tiles_m,
            reduce_scratch,
            rows_d, rows_m
        );

        apply_dm<TILE_M>(
            out,
            rows_d, rows_m,
            N
        );
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
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);
    TORCH_CHECK(qtype == GGMLQuantizationType::F16);

    TORCH_CHECK(w.size(1) == x.size(-1));
    TORCH_CHECK(w.size(0) == out.size(-1));
    TORCH_CHECK(x.numel() / x.size(-1) == out.numel() / out.size(-1));

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kByte);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    const uint8_t* w_ptr = w.data_ptr<uint8_t>();

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t N = w.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int TILE_SIZE=64;
    constexpr int TILE_K=256;
    constexpr int BLOCK_SIZE=512;

    dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    dim3 block(BLOCK_SIZE); // TODO: tune this or adjust as needed

    if (!moe_scratch.init) { // lazy init of moe_scratch
        size_t scratch_size = grid.x*grid.y*TILE_SIZE; // TILE_SIZE entries per block (due to each row)
        cudaMalloc(&moe_scratch.tiles_d, scratch_size*sizeof(float));
        cudaMalloc(&moe_scratch.tiles_m, scratch_size*sizeof(float));
        cudaMalloc(&moe_scratch.row_visit_cnt, M*sizeof(int));
        moe_scratch.init = true;
    }

    cudaMemset(moe_scratch.row_visit_cnt, 0, M*sizeof(int));

    moe_scoring_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K, BLOCK_SIZE><<<grid, block, 2*grid.y*sizeof(float), stream>>>(
        qtype_int, qblock_size, qtype_size,
        M, K, N,
        moe_scratch.tiles_d, moe_scratch.tiles_m, moe_scratch.row_visit_cnt, 
        x_ptr, out_ptr, w_ptr
    );
}

template <int TILE_M, int TILE_N, int TILE_K, int BLOCK_SIZE>
__global__ void ffn_cuda_impl(
    int up_qtype_int, int gate_qtype_int, int down_qtype_int,
    int up_qblock_size, int gate_qblock_size, int down_qblock_size,
    int up_qtype_size, int gate_qtype_size, int down_qtype_size,
    int64_t M, int64_t K_GU, int64_t N_GU, int64_t K_DOWN, int64_t N_DOWN,
    const half* __restrict__ in,
    half* __restrict__ out,
    half* __restrict__ hb,
    half* __restrict__ hb2,
    const uint8_t* __restrict__ ws_up,
    const uint8_t* __restrict__ ws_gate,
    const uint8_t* __restrict__ ws_down
) {

    constexpr int NUM_SUBTILES = (TILE_M/16)*(TILE_N/16);
    constexpr int num_warps = BLOCK_SIZE / 32;
    constexpr int tiles_per_warp = NUM_SUBTILES / num_warps;
    
    int warp_id = threadIdx.x / 32;
    
    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ half w_shared[TILE_K][TILE_N];
    
    // gate proj stored in hb
    bool gate_is_fp16 = static_cast<GGMLQuantizationType>(gate_qtype_int) == GGMLQuantizationType::F16;
    if (gate_is_fp16) {
        compute_tile<TILE_M, TILE_N, TILE_K, true>(
            gate_qtype_int, gate_qblock_size, gate_qtype_size,
            K_GU, N_GU, warp_id, tiles_per_warp,
            x_shared, w_shared, in, hb, ws_gate
        );
    } else {
        compute_tile<TILE_M, TILE_N, TILE_K, false>(
            gate_qtype_int, gate_qblock_size, gate_qtype_size,
            K_GU, N_GU, warp_id, tiles_per_warp,
            x_shared, w_shared, in, hb, ws_gate
        );
    }
    __syncthreads();
    
    // apply swish to hb (swish(x) = x*sigmoid(beta*x), beta taken to be 1 here)
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=BLOCK_SIZE) {
        float val = __half2float(hb[blockIdx.x*TILE_M*N_GU + blockIdx.y*TILE_N + idx]);
        val = val / (1.0f+expf(-val));
        hb[blockIdx.x*TILE_M*N_GU + blockIdx.y*TILE_N + idx] = __float2half(val);
    }
    __syncthreads();
    
    // up proj stored in hb2
    bool up_is_fp16 = static_cast<GGMLQuantizationType>(up_qtype_int) == GGMLQuantizationType::F16;
    if (up_is_fp16) {
        compute_tile<TILE_M, TILE_N, TILE_K, true>(
            up_qtype_int, up_qblock_size, up_qtype_size,
            K_GU, N_GU, warp_id, tiles_per_warp,
            x_shared, w_shared, in, hb2, ws_up
        );
    } else {
        compute_tile<TILE_M, TILE_N, TILE_K, false>(
            up_qtype_int, up_qblock_size, up_qtype_size,
            K_GU, N_GU, warp_id, tiles_per_warp,
            x_shared, w_shared, in, hb2, ws_up
        );
    }
    __syncthreads();
    
    // element-wise mul of hb and hb2, stored in hb2
    for (int idx=threadIdx.x; idx<TILE_M*TILE_N; idx+=BLOCK_SIZE) {
        int tile_idx = blockIdx.x*TILE_M*N_GU + blockIdx.y*TILE_N + idx;
        float a = __half2float(hb[tile_idx]);
        float b = __half2float(hb2[tile_idx]);
        hb2[tile_idx] = __float2half(a*b);
    }
    __syncthreads();
    
    bool down_is_fp16 = static_cast<GGMLQuantizationType>(down_qtype_int) == GGMLQuantizationType::F16;
    if (down_is_fp16) {
        compute_tile<TILE_M, TILE_N, TILE_K, true>(
            down_qtype_int, down_qblock_size, down_qtype_size,
            K_DOWN, N_DOWN, warp_id, tiles_per_warp,
            x_shared, w_shared, hb2, out, ws_down
        );
    } else {
        compute_tile<TILE_M, TILE_N, TILE_K, false>(
            down_qtype_int, down_qblock_size, down_qtype_size,
            K_DOWN, N_DOWN, warp_id, tiles_per_warp,
            x_shared, w_shared, hb2, out, ws_down
        );
    }

}

// elementwise multiplication of up proj tile with swiglu (tile of swish(gate(x))), then apply downproj
void ffn_cuda(
    int64_t up_qtype_int, int64_t gate_qtype_int, int64_t down_qtype_int,
    int64_t up_qblock_size, int64_t gate_qblock_size, int64_t down_qblock_size,
    int64_t up_qtype_size, int64_t gate_qtype_size, int64_t down_qtype_size,
    const at::Tensor& in, // [B,L,hidden_dim]
    at::Tensor& out, // [B,L,hidden_dim]
    at::Tensor& hb, // [B,L,n_local_exps,mlp_dim]
    at::Tensor& hb2, // [B,L,n_local_exps,mlp_dim]
    const at::Tensor& ws_up, // [n_local_exps, mlp_dim, hidden_dim]
    const at::Tensor& ws_gate, // // [n_local_exps, mlp_dim, hidden_dim]
    const at::Tensor& ws_down // [n_local_exps, hidden_dim, mlp_dim]
) {
    //validation 

    //dims
    TORCH_CHECK((in.dim() == 3) && (out.dim() == 3));
    TORCH_CHECK((hb.dim() == 4) && (hb2.dim() == 4));
    TORCH_CHECK((ws_up.dim() == 3) && (ws_gate.dim() == 3) && (ws_down.dim() == 3));

    //checks between dimensions
    TORCH_CHECK(in.sizes().equals(out.sizes()));
    TORCH_CHECK(hb.sizes().equals(hb2.sizes()));
    TORCH_CHECK(
        (ws_up.numel() / ws_up.size(1) == ws_down.numel() / ws_down.size(-1)) &&
        (ws_gate.numel() / ws_gate.size(1) == ws_down.numel() / ws_down.size(-1))
    );

    //dtypes
    TORCH_CHECK(
        (in.dtype() == at::kHalf) && (out.dtype() == at::kHalf) && 
        (hb.dtype() == at::kHalf) && (hb2.dtype() == at::kHalf) &&
        (ws_up.dtype() == at::kByte) && (ws_gate.dtype() == at::kByte) && (ws_down.dtype() == at::kByte)
    );

    TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(hb.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(hb2.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws_up.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws_gate.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(ws_down.device().type() == at::DeviceType::CUDA);

    const half* in_ptr = reinterpret_cast<const half*>(in.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    half* hb_ptr = reinterpret_cast<half*>(hb.data_ptr<at::Half>());
    half* hb2_ptr = reinterpret_cast<half*>(hb2.data_ptr<at::Half>());

    const uint8_t* ws_up_ptr = ws_up.data_ptr<uint8_t>();
    const uint8_t* ws_gate_ptr = ws_gate.data_ptr<uint8_t>();
    const uint8_t* ws_down_ptr = ws_down.data_ptr<uint8_t>();

    int64_t M = in.numel() / in.size(-1);
    int64_t N_GU = hb.size(2) * hb.size(3);
    int64_t N_DOWN = in.size(-1);
    int64_t N = std::max(N_GU, N_DOWN);

    int64_t K_GU = in.size(-1);
    int64_t K_DOWN = hb.size(2) * hb.size(3);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    constexpr int TILE_SIZE=64;
    constexpr int TILE_K=256;
    constexpr int BLOCK_SIZE=512;
    
    dim3 block(512); // TODO: tune this or adjust as needed
    dim3 grid((M+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    ffn_cuda_impl<TILE_SIZE, TILE_SIZE, TILE_K, BLOCK_SIZE><<<grid, block, 0, stream>>>(
        up_qtype_int, gate_qtype_int, down_qtype_int,
        up_qblock_size, gate_qblock_size, down_qblock_size,
        up_qtype_size, gate_qtype_size,down_qtype_size,
        M, K_GU, N_GU, K_DOWN, N_DOWN,
        in_ptr, out_ptr, hb_ptr, hb2_ptr,
        ws_up_ptr, ws_gate_ptr, ws_down_ptr
    );
}

// TODO: add this back in once finished
// TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
//     m.impl("rmsnorm", &rmsnorm_cuda);
//     m.impl("il_rope", &il_rope_cuda);
//     m.impl("neox_rope", &neox_rope_cuda);
//     m.impl("matmul", &matmul_cuda);
//     m.impl("embed", &embed_cuda);
//     m.impl("qkv", &qkv_cuda);
//     m.impl("flash_attn", &flash_attn_cuda);
//     m.impl("moe_scoring", &moe_scoring_cuda);
//     m.impl("ffn", &ffn_cuda);
// }

// for computing online softmax in moe scoring
static struct MoeScoring {
    float* tiles_d;
    float* tiles_m;
    int* row_visit_cnt;
    bool init;
    
    MoeScoring() { // these are lazily init in moe_scoring_cuda
        tiles_d = nullptr;
        tiles_m = nullptr;
        row_visit_cnt = nullptr;
        init = false;
    }
    ~MoeScoring() {
        if (tiles_d) cudaFree(tiles_d);
        if (tiles_m) cudaFree(tiles_m);
        if (row_visit_cnt) cudaFree(row_visit_cnt);
    }
} moe_scratch;

}