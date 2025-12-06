#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cstdint>

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>

#include <cassert>

#include "quants_impl.cuh"
#include "impl_common.hpp"

using namespace nvcuda;

// TODO: complete me!
namespace minfer {

// helpers for warp-level reductions
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

// helper for dispatch in matmul_cuda_impl
template <typename T>
static __device__ void dequant_block(
    int qtype_int,
    const uint8_t* __restrict__ w,
    T* __restrict__ y,
    int64_t stride,
    int tid
) {
    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);

    switch (qtype) {
        case GGMLQuantizationType::Q4_0: dequant_block_q4_0<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q4_1: dequant_block_q4_1<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q5_0: dequant_block_q5_0<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q5_1: dequant_block_q5_1<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q8_0: dequant_block_q8_0<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::MXFP4: dequant_block_mxfp4<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q2_K: dequant_block_q2_K<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q3_K: dequant_block_q3_K<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q4_K: dequant_block_q4_K<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q5_K: dequant_block_q5_K<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q6_K: dequant_block_q6_K<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::TQ1_0: dequant_block_tq1_0<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::TQ2_0: dequant_block_tq2_0<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ2_XXS: dequant_block_iq2_xxs<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ2_XS: dequant_block_iq2_xs<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ2_S: dequant_block_iq2_s<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ3_XXS: dequant_block_iq3_xxs<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ3_S: dequant_block_iq3_s<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ1_S: dequant_block_iq1_s<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ1_M: dequant_block_iq1_m<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ4_NL: dequant_block_iq4_nl<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::IQ4_XS: dequant_block_iq4_xs<T>(w, y, stride, tid); break;
        case GGMLQuantizationType::Q8_K: dequant_block_q8_K<T>(w, y, stride, tid); break;
        default: assert(false && "Unsupported dtype"); // this gets compiled out in non-debug builds...
    }
}

__device__ float blockreduce_sum(float* vs, float v, int tid) {
    v = warp_reduce_sum(v);
    if (tid%32 == 0) vs[tid/32] = v;
    __syncthreads();
    v = vs[tid%32];
    return warp_reduce_sum(v);
}

__device__ float blockreduce_max(float* vs, float v, int tid) {
    v = warp_reduce_max(v);
    if (tid%32 == 0) vs[tid/32] = v;
    __syncthreads();
    v = vs[tid%32];
    return warp_reduce_max(v);
}

__global__ void rmsnorm_cuda_impl(
    const half* __restrict__ in,
    half* __restrict__ out,
    const half* __restrict__ w,
    int dim,
    float eps
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

void rmsnorm_cuda(const at::Tensor& in, at::Tensor& out, const at::Tensor& w, double eps) {
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
    rmsnorm_cuda_impl<<<n_blocks, 1024, 0, stream>>>(in_ptr, out_ptr, w_ptr, dim, eps);
}

__global__ void il_rope_cuda_impl (
    half* __restrict__ x,
    int n_heads,
    int rotary_dim,
    int head_dim,
    int64_t start_pos,
    float freq_base
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
    at::Tensor& x, // [B,L,n_heads,head_dim]
    int64_t rotary_dim,
    int64_t start_pos,
    double freq_base
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
    il_rope_cuda_impl<<<grid, 32, 0, stream>>>(x_ptr, n_heads, rotary_dim, head_dim, start_pos, freq_base);
}

__global__ void neox_rope_cuda_impl(
    half* __restrict__ x,
    int n_heads,
    int rotary_dim,
    int head_dim,
    int64_t start_pos,
    float freq_base
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
    at::Tensor& x, // [B,L,n_heads,head_dim]
    int64_t rotary_dim,
    int64_t start_pos,
    double freq_base
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
    neox_rope_cuda_impl<<<grid, 32, 0, stream>>>(x_ptr, n_heads, rotary_dim, head_dim, start_pos, freq_base);
}

template <int TILE_SIZE, int QBLOCK_SIZE>
__global__ void matmul_cuda_impl(
    int qtype_int,
    const half* __restrict__ x,
    half* __restrict__ out,
    const uint8_t* __restrict__ w,
    int64_t M, int64_t K, int64_t N
) {
    constexpr int TILE_M = TILE_SIZE;
    constexpr int TILE_N = TILE_SIZE;
    constexpr int TILE_K = QBLOCK_SIZE; // we have to dequant this many elems at a time
    constexpr int NUM_SUBTILES = (TILE_M/16)*(TILE_N/16); // wmma operates on 16x16 subtiles
    constexpr int SUBTILE_DIM = TILE_M/16;

    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    int tiles_per_warp = NUM_SUBTILES / num_warps;

    __shared__ half x_shared[TILE_M][TILE_K];
    __shared__ half w_shared[TILE_K][TILE_N];

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_frag;

    // split work over the subtiles (arranged in a grid of TILE_SIZE/16xTILE_SIZE/16), each of size 16x16.
    for (int i=0; i<tiles_per_warp; ++i) {
        int tile_id = warp_id * tiles_per_warp + i;
        int tile_m = (tile_id/SUBTILE_DIM)*16;
        int tile_n = (tile_id%SUBTILE_DIM)*16;
        
        // K-dimension processed in TILE_K chunks
        for (int64_t k1=0; k1<K; k1+=TILE_K) {
            // coop koad x tile to shared mem
            for (int idx = threadIdx.x; idx < TILE_M*TILE_K; idx += blockDim.x) {
                int m = idx / TILE_K;
                int k = idx % TILE_K;
                x_shared[m][k] = x[(blockIdx.x * TILE_M + m) * K + (k1 + k)];
            }

            // coop load and dequantize w tile to shared mem
            for (int n=0; n<TILE_N; ++n) {
                const uint8_t* q_block = &w[(blockIdx.y*TILE_N+n)*(K/QBLOCK_SIZE)+(k1/QBLOCK_SIZE)];

                dequant_block<half>( // this kernel is half-specific
                    qtype_int,
                    q_block,
                    &w_shared[0][n],
                    TILE_N,
                    threadIdx.x
                );
            }

            __syncthreads();

            // process TILE_K with WMMA in chunks of 16
            for (int k2=0; k2<TILE_K; k2+=16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, &x_shared[tile_m][k2], TILE_K);
                wmma::load_matrix_sync(b_frag, &w_shared[k2][tile_n], TILE_N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            __syncthreads();
        }

        wmma::store_matrix_sync(
            &out[(blockIdx.x*TILE_M+tile_m)*N+(blockIdx.y*TILE_N+tile_n)],
            acc_frag,
            N,
            wmma::mem_row_major
        );
        wmma::fill_fragment(acc_frag, __float2half(0.0f));
    }
}

void matmul_cuda(
    int64_t qtype_int,
    const at::Tensor& x,
    at::Tensor& out,
    const at::Tensor& w,
    int64_t dim_in,
    int64_t dim_out,
    int64_t qblock_size
) {
    TORCH_CHECK(x.size(-1) == dim_in);
    TORCH_CHECK(out.size(-1) == dim_out);
    TORCH_CHECK(w.size(0) == dim_out && w.size(1) == dim_in);

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kByte);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(qblock_size == 256 || qblock_size == 32);

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
    dim3 block(256); // TODO: tune this or adjust as needed

    if (qblock_size == 256) {
        dim3 grid((M+32-1)/32, (N+32-1)/32);
        matmul_cuda_impl<32,256><<<grid, block, 0, stream>>>(qtype_int, x_ptr, out_ptr, w_ptr, M, K, N);
    } else{
        dim3 grid((M+256-1)/256, (N+256-1)/256);
        matmul_cuda_impl<256,32><<<grid, block, 0, stream>>>(qtype_int, x_ptr, out_ptr, w_ptr, M, K, N);
    }
}

template <typename T>
__global__ void embed_cuda_impl(
    int qtype_int,
    const int64_t* __restrict__ token_ids,
    const uint8_t* __restrict__ w,
    T* __restrict__ x,
    int64_t qblock_size,
    int64_t qtype_size,
    int b, // bytes per row
    int k // dequant elems per row
) {
    int B = gridDim.x;
    int L = gridDim.y;

    int iB = blockIdx.x;
    int iL = blockIdx.y;
    int block_in_row = blockIdx.z;
    int token_id = token_ids[iB*L+iL];

    const uint8_t* w_block = w + token_id*b + block_in_row*qtype_size;
    T* x_block = x + iB*L*k + iL*k + block_in_row*qblock_size;

    dequant_block(qtype_int, w_block, x_block, 1, threadIdx.x);
}

void embed_cuda(
    int64_t qtype_int,
    const at::Tensor& token_ids, // [B,L]
    const at::Tensor& w, // [vocab_size,bytes_per_row]
    at::Tensor& x, // [B,L,D]
    int64_t qblock_size,
    int64_t qtype_size
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

    const int64_t * token_ids_ptr = token_ids.data_ptr<int64_t>();
    const uint8_t * w_ptr = w.data_ptr<uint8_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int n_qblocks_per_row = x.size(-1) / qblock_size;
    dim3 grid(x.size(0), x.size(1), n_qblocks_per_row);

    switch (x.scalar_type()) {
        case at::kFloat: {
            float* x_ptr = x.data_ptr<float>();
            embed_cuda_impl<float><<<grid, qblock_size, 0, stream>>>(qtype_int, token_ids_ptr, w_ptr, x_ptr, qblock_size, qtype_size, w.size(-1), x.size(-1));
            break;
        }

        case at::kHalf: {
            half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());
            embed_cuda_impl<half><<<grid, qblock_size, 0, stream>>>(qtype_int, token_ids_ptr, w_ptr, x_ptr, qblock_size, qtype_size, w.size(-1), x.size(-1));
            break;
        }

        default: TORCH_CHECK(false && "Unsupported activation (x) dtype"); break;
    }
}

void qkv_cuda() {
    TORCH_CHECK(false, "qkv not implemented");
}

void flash_attn_cuda() {
    TORCH_CHECK(false, "flash_attn not implemented");
}

void moe_scoring_cuda() {
    TORCH_CHECK(false, "moe_scoring not implemented");
}

void ffn_cuda() {
    TORCH_CHECK(false, "ffn not implemented");
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

}