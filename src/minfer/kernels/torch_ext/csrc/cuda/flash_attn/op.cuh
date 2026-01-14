#pragma once

#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

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

    // constexpr int WARPS_PER_BLOCK = 4; // TODO: tune this or adjust as needed
    // constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
    // constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
    // constexpr int HEAD_DIM = 128;
    // dim3 grid(B, n_heads, (L+ROWS_M-1)/ROWS_M);
    // dim3 block(BLOCK_SIZE);

    // flash_attn_cuda_impl<WARPS_PER_BLOCK, HEAD_DIM><<<grid, block, 0, stream>>>(
    //     strides,
    //     L, n_kv_heads,
    //     q_ptr, k_ptr, v_ptr,
    //     mask_ptr,
    //     out_ptr
    // );
}