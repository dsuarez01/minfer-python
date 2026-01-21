#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

namespace minfer {

    using namespace impl;

    void qkv_cuda(
        int64_t q_qtype_int, int64_t k_qtype_int, int64_t v_qtype_int,
        int64_t q_qblock_size, int64_t k_qblock_size, int64_t v_qblock_size,
        int64_t q_qtype_size, int64_t k_qtype_size, int64_t v_qtype_size,
        const torch::stable::Tensor& x, // [B, L, hidden_dim]
        const torch::stable::Tensor& wq, // [hidden_dim, q_dim (possibly in bytes)]
        const torch::stable::Tensor& wk, // [hidden_dim, kv_dim (possibly in bytes)]
        const torch::stable::Tensor& wv, // [hidden_dim, kv_dim (possibly in bytes)]
        torch::stable::Tensor& q_out, // [B, L, q_dim]
        torch::stable::Tensor& k_out, // [B, L, kv_dim]
        torch::stable::Tensor& v_out // [B, L, kv_dim]
    ) {
        // // validation
        // TORCH_CHECK(is_valid_qtype(q_qtype_int) && is_valid_qtype(k_qtype_int) && is_valid_qtype(v_qtype_int));
        // auto q_qtype = static_cast<GGMLQuantizationType>(q_qtype_int);
        // auto k_qtype = static_cast<GGMLQuantizationType>(k_qtype_int);
        // auto v_qtype = static_cast<GGMLQuantizationType>(v_qtype_int);

        // TORCH_INTERNAL_ASSERT(
        //     (x.device().type() == at::DeviceType::CUDA) &&
        //     (q_out.device().type() == at::DeviceType::CUDA) &&
        //     (k_out.device().type() == at::DeviceType::CUDA) && 
        //     (v_out.device().type() == at::DeviceType::CUDA) &&
        //     (wq.device().type() == at::DeviceType::CUDA) &&
        //     (wk.device().type() == at::DeviceType::CUDA) &&
        //     (wv.device().type() == at::DeviceType::CUDA)
        // );

        // TORCH_CHECK(
        //     x.is_contiguous() &&
        //     q_out.is_contiguous() && k_out.is_contiguous() && v_out.is_contiguous() &&
        //     wq.is_contiguous() && wk.is_contiguous() && wv.is_contiguous()
        // );

        // //dtypes
        // TORCH_CHECK(
        //     (x.scalar_dtype() == at::kHalf) && 
        //     (q_out.scalar_type() == at::kHalf) && (k_out.scalar_type() == at::kHalf) && (v_out.scalar_type() == at::kHalf) &&
        //     (wq.scalar_type() == at::kByte || wq.scalar_type() == at::kHalf) && 
        //     (wk.scalar_type() == at::kByte || wk.scalar_type() == at::kHalf) && 
        //     (wv.scalar_type() == at::kByte || wv.scalar_type() == at::kHalf)
        // );

        // // dim
        // TORCH_CHECK(
        //     (x.dim() == 3) && 
        //     (q_out.dim() == 3) && (k_out.dim() == 3) && (v_out.dim() == 3) &&
        //     (wq.dim() == 2) && (wk.dim() == 2) && (wv.dim() == 2)
        // );

        // //size
        // TORCH_CHECK(
        //     (wq.size(0) == q_out.size(-1)) &&
        //     (wk.size(0) == k_out.size(-1)) &&
        //     (wv.size(0) == v_out.size(-1)) &&
        //     (wk.size(-1) == wv.size(-1))
        // );

        // TORCH_CHECK(
        //     (x.numel() / x.size(-1) == q_out.numel() / q_out.size(-1)) &&
        //     (x.numel() / x.size(-1) == k_out.numel() / k_out.size(-1)) &&
        //     (x.numel() / x.size(-1) == v_out.numel() / v_out.size(-1)) &&
        //     (k_out.size(-1) == v_out.size(-1))
        // );

        // const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
        
        // half* q_out_ptr = reinterpret_cast<half*>(q_out.data_ptr<at::Half>());
        // half* k_out_ptr = reinterpret_cast<half*>(k_out.data_ptr<at::Half>());
        // half* v_out_ptr = reinterpret_cast<half*>(v_out.data_ptr<at::Half>());

        // int64_t M = x.numel() / x.size(-1);
        // int64_t K = x.size(-1);
        // int64_t N_Q = wq.size(0);
        // int64_t N_KV = wk.size(0);
        // int64_t N = std::max(N_Q, N_KV);
        
        // cudaDeviceProp deviceProp;
        // AT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, x.device().index()));

        // if (deviceProp.major < 7) {
        //     TORCH_CHECK(false, "SM 7.0 or higher required to use tensor cores");
        // }

        // // assume: either all of QKV are quantized (not F16) or dequantized (all of QKV are FP16)
        // if (
        //     q_qtype == GGMLQuantizationType::F16 
        //     && k_qtype == GGMLQuantizationType::F16 
        //     && v_qtype == GGMLQuantizationType::F16
        // ) {

        //     TORCH_CHECK(
        //         (wq.size(1) == x.size(-1)) &&
        //         (wk.size(1) == x.size(-1)) &&
        //         (wv.size(1) == x.size(-1))
        //     );

        //     const half* wq_ptr = reinterpret_cast<const half*>(wq.data_ptr<at::Half>());
        //     const half* wk_ptr = reinterpret_cast<const half*>(wk.data_ptr<at::Half>());
        //     const half* wv_ptr = reinterpret_cast<const half*>(wv.data_ptr<at::Half>());

        //     launch_qkv_f16_kernel<Config_QKV_F16_T>(
        //         deviceProp,
        //         M, K, N_Q, N_KV,
        //         x_ptr, wq_ptr, wk_ptr, wv_ptr,
        //         q_out_ptr, k_out_ptr, v_out_ptr
        //     );
        // } else {

        //     TORCH_CHECK(
        //         (wq.size(1) == (x.size(-1) / q_qblock_size) * q_qtype_size) &&
        //         (wk.size(1) == (x.size(-1) / k_qblock_size) * k_qtype_size) &&
        //         (wv.size(1) == (x.size(-1) / v_qblock_size) * v_qtype_size)
        //     );

        //     const uint8_t* wq_ptr = wq.data_ptr<uint8_t>();
        //     const uint8_t* wk_ptr = wk.data_ptr<uint8_t>();
        //     const uint8_t* wv_ptr = wv.data_ptr<uint8_t>();

        //     launch_qkv_quant_kernel<Config_QKV_Quant_T>(
        //         deviceProp,
        //         q_qtype_int, k_qtype_int, v_qtype_int,
        //         q_qblock_size, k_qblock_size, v_qblock_size,
        //         q_qtype_size, k_qtype_size, v_qtype_size,
        //         M, K, N_Q, N_KV,
        //         x_ptr, wq_ptr, wk_ptr, wv_ptr,
        //         q_out_ptr, k_out_ptr, v_out_ptr
        //     );

        // }
        // STD_CUDA_KERNEL_LAUNCH_CHECK();
    }
}