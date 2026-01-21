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

namespace {

    using namespace minfer::impl;

    template <int TILE_M, int N_EXPS>
    inline void dispatch_top_k(
        int64_t top_k, int64_t M, int64_t K,
        const half* x_ptr, const half* w_ptr,
        uint8_t* act_exps_ptr, half* act_exps_scores_ptr, half* scores_ptr,
        cudaStream_t stream
    ) {
        // dim3 grid((M+TILE_M-1)/TILE_M);
        // dim3 block(TILE_M*N_EXPS);
        
        // switch (top_k) {
        //     case 1: moe_scoring_cuda_impl<TILE_M, N_EXPS, 1><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        //     case 2: moe_scoring_cuda_impl<TILE_M, N_EXPS, 2><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        //     case 4: moe_scoring_cuda_impl<TILE_M, N_EXPS, 4><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        //     case 8: moe_scoring_cuda_impl<TILE_M, N_EXPS, 8><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        //     default: TORCH_CHECK(false, "unsupported top_k: ", top_k);
        // }
    }

}

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

        const half* x_ptr = reinterpret_cast<const half*>(x.const_data_ptr<torch::headeronly::ScalarType::Half>());
        const half* w_ptr = reinterpret_cast<const half*>(w.const_data_ptr<torch::headeronly::ScalarType::Half>());
        uint8_t* act_exps_ptr = act_exps.mutable_data_ptr<uint8_t>();
        half* act_exps_scores_ptr = reinterpret_cast<half*>(act_exps_scores.mutable_data_ptr<torch::headeronly::ScalarType::Half>());
        half* scores_ptr = reinterpret_cast<half*>(scores.mutable_data_ptr<torch::headeronly::ScalarType::Half>());

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

        switch (n_exps) {
            case 8:   dispatch_top_k<128, 8>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
            case 16:  dispatch_top_k<64, 16>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
            case 32:  dispatch_top_k<32, 32>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
            case 64:  dispatch_top_k<16, 64>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
            case 128: dispatch_top_k<8, 128>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
            case 256: dispatch_top_k<4, 256>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
            default: TORCH_CHECK(false, "unsupported n_exps: ", n_exps);
        }

        STD_CUDA_KERNEL_LAUNCH_CHECK();
    }
}