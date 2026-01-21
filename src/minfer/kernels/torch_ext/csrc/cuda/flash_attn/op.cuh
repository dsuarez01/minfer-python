#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

namespace minfer {

using namespace impl;

void flash_attn_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const torch::stable::Tensor& q, // [B, n_heads, L, HEAD_DIM]
    const torch::stable::Tensor& k, // [B, n_kv_heads, L, HEAD_DIM]
    const torch::stable::Tensor& v,  // [B, n_kv_heads, L, HEAD_DIM]
    torch::stable::Tensor& mask, // [B, 1, L, L]
    torch::stable::Tensor& out     // [B, n_heads, L, HEAD_DIM]
) {
    // validation
    STD_TORCH_CHECK(
        (q.device().type() == torch::headeronly::DeviceType::CUDA) &&
        (k.device().type() == torch::headeronly::DeviceType::CUDA) && 
        (v.device().type() == torch::headeronly::DeviceType::CUDA) &&
        (mask.device().type() == torch::headeronly::DeviceType::CUDA) &&
        (out.device().type() == torch::headeronly::DeviceType::CUDA)
    );

    // contiguity in the last dimension
    // except mask, is fully contiguous
    STD_TORCH_CHECK(
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
    STD_TORCH_CHECK(
        (q.scalar_type() == torch::headeronly::ScalarType::Half) &&
        (k.scalar_type() == torch::headeronly::ScalarType::Half) &&
        (v.scalar_type() == torch::headeronly::ScalarType::Half) &&
        (out.scalar_type() == torch::headeronly::ScalarType::Half) && 
        (mask.scalar_type() == torch::headeronly::ScalarType::Bool)
    );

    // dims
    STD_TORCH_CHECK(
        (q.dim() == 4) &&
        (k.dim() == 4) &&
        (v.dim() == 4) &&
        (out.dim() == 4) &&
        (mask.dim() == 4)
    );

    // checks btwn different dims
    STD_TORCH_CHECK(q.sizes().equals(out.sizes()));
    STD_TORCH_CHECK(k.sizes().equals(v.sizes()));
    STD_TORCH_CHECK(
        (q.size(0) == k.size(0)) && (q.size(0) == mask.size(0)) &&
        (q.size(2) == k.size(2)) && (q.size(2) == mask.size(2)) && (q.size(2) == mask.size(3)) &&
        (q.size(3) == k.size(3))
    );

    QKVStrides strides = {
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2)
    };

    const half* q_ptr = reinterpret_cast<const half*>(q.const_data_ptr<half_t>());
    const half* k_ptr = reinterpret_cast<const half*>(k.const_data_ptr<half_t>());
    const half* v_ptr = reinterpret_cast<const half*>(v.const_data_ptr<half_t>());
    half* out_ptr = reinterpret_cast<half*>(out.mutable_data_ptr<half_t>());
    const uint8_t* mask_ptr = mask.const_data_ptr<uint8_t>();

    int64_t B = q.size(0);
    int64_t n_heads = q.size(1);
    int64_t n_kv_heads = k.size(1);
    int64_t head_dim = q.size(-1);
    STD_TORCH_CHECK(head_dim == 128);
    int64_t L = q.size(2);

    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
    );
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

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
    
    // STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}