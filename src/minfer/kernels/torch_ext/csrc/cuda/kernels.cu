#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// TODO: complete me!
namespace minfer {

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

void rmsnorm_cuda(const at::Tensor& in, at::Tensor& out, const at::Tensor& w, float eps) {
    // checks
    TORCH_CHECK(in.shape() == out.shape());
    TORCH_CHECK(in.dims() == 3 || in.dims() == 4);
    TORCH_CHECK(in.size(-1) == w.size(0) || in.size(-1) % w.size(0) == 0); // per-head or per-entire vector
    
    TORCH_CHECK(in.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);
    
    TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    
    TORCH_CHECK(in.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    const half* in_ptr = in.data_ptr<at::Half>();
    half* out_ptr = out.data_ptr<at::Half>();
    const half* w_ptr = w.data_ptr<at::Half>();

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
    int start_pos,
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
    int rotary_dim,
    int head_dim,
    int start_pos,
    float freq_base
) {
    TORCH_CHECK(rotary_dim <= head_dim);
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(x.is_contiguous());

    half* x_ptr = x.data_ptr<at::Half>();

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
    int start_pos,
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
    int rotary_dim,
    int head_dim,
    int start_pos,
    float freq_base
) {
    TORCH_CHECK(rotary_dim <= head_dim);
    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4);
    TORCH_CHECK(x.is_contiguous());

    half* x_ptr = x.data_ptr<at::Half>();

    int B = x.size(0);
    int L = x.size(1);
    int n_heads = x.size(2);
    int head_dim = x.size(3);

    dim3 grid(B, L, n_heads*(rotary_dim/2+31)/32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    neox_rope_cuda_impl<<<grid, 32, 0, stream>>>(x_ptr, n_heads, rotary_dim, head_dim, start_pos, freq_base);
}

void matmul_cuda() {
    TORCH_CHECK(false, "matmul not implemented");
}

void embed_cuda() {
    TORCH_CHECK(false, "embed not implemented");
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