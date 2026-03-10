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
#include "kernels/kernel.cuh"

namespace minfer {

void moe_scoring_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const torch::stable::Tensor& x, // [B,L,hidden_dim]
    const torch::stable::Tensor& w, // [n_experts, hidden_dim]
    torch::stable::Tensor& act_exps, // [B,L,n_act_exps]
    torch::stable::Tensor& act_exps_scores, // [B,L,n_act_exps]
    torch::stable::Tensor& scores // [B,L,n_experts]
) {
    STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(scores.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(w.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(act_exps.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(act_exps_scores.device().type() == torch::headeronly::DeviceType::CUDA);

    STD_TORCH_CHECK(x.is_contiguous());
    STD_TORCH_CHECK(scores.is_contiguous());
    STD_TORCH_CHECK(w.is_contiguous());
    STD_TORCH_CHECK(act_exps.is_contiguous());
    STD_TORCH_CHECK(act_exps_scores.is_contiguous());

    STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(scores.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(w.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(act_exps.scalar_type() == torch::headeronly::ScalarType::Byte);
    STD_TORCH_CHECK(act_exps_scores.scalar_type() == torch::headeronly::ScalarType::Half);

    const half* x_ptr = reinterpret_cast<const half*>(x.const_data_ptr<half_t>());
    const half* w_ptr = reinterpret_cast<const half*>(w.const_data_ptr<half_t>());
    uint8_t* act_exps_ptr = act_exps.mutable_data_ptr<uint8_t>();
    half* act_exps_scores_ptr = reinterpret_cast<half*>(act_exps_scores.mutable_data_ptr<half_t>());
    half* scores_ptr = reinterpret_cast<half*>(scores.mutable_data_ptr<half_t>());

    size_t M = x.numel() / x.size(-1);
    size_t K = x.size(-1);
    int n_exps = scores.size(-1);
    int top_k = act_exps.size(-1);

    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
    );
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // TO-DO: implement
}

}