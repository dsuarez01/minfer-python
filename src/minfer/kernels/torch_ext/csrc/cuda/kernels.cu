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

    const half* head_in = in + blockIdx.x * dim;
    half* head_out = out + blockIdx.x * dim;
    
    __shared__ float shared_sum[32];
    
    // one pass for the squared sum (then parallel reduction over block)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i<dim; i += blockDim.x) {
        float val = __half2float(head_in[i]);
        sum_sq += val*val;
    }

    sum_sq = blockreduce_sum(shared_sum, sum_sq, threadIdx.x);
    float sc = rsqrt(sum_sq / float(dim) + eps);
    
    // one pass to apply weight and scale
    for (int i = threadIdx.x; i<dim; i += blockDim.x) {
        float in_f = __half2float(head_in[i]);
        float w_f = __half2float(w[i]);
        head_out[i] = __float2half(in_f*w_f*sc);
    }
}

void rmsnorm_cuda(const at::Tensor& in, at::Tensor& out, const at::Tensor& w, int dim, float eps) {
    // checks
    TORCH_CHECK(in.size(0) == out.size(0));
    TORCH_CHECK(in.size(0) == w.size(0) || in.size(0) % w.size(0) == 0)
    TORCH_CHECK(in.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);
    TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    
    TORCH_CHECK(in.is_contiguous());
    TORCH_CHECK(out.is_contiguous());

    const half* in_ptr = in.data_ptr<at::Half>();
    half* out_ptr = out.data_ptr<at::Half>();
    const half* w_ptr = out.data_ptr<at::Half>();

    const int n_heads = in.size(0) / w.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rmsnorm_cuda_impl<<<n_heads, 1024, 0, stream>>>(in_ptr, out_ptr, w_ptr, dim, eps);
}

void il_rope_cuda() {
    TORCH_CHECK(false, "il_rope not implemented");
}

void neox_rope_cuda() {
    TORCH_CHECK(false, "neox_rope not implemented");
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