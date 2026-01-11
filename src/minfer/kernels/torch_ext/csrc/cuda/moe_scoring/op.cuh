#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

template <int TILE_M, int N_EXPS>
inline void dispatch_top_k(
    int64_t top_k, int64_t M, int64_t K,
    const half* x_ptr, const half* w_ptr,
    uint8_t* act_exps_ptr, half* act_exps_scores_ptr, half* scores_ptr,
    cudaStream_t stream
) {
    dim3 grid((M+TILE_M-1)/TILE_M);
    dim3 block(TILE_M*N_EXPS);
    
    switch (top_k) {
        case 1: moe_scoring_cuda_impl<TILE_M, N_EXPS, 1><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        case 2: moe_scoring_cuda_impl<TILE_M, N_EXPS, 2><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        case 4: moe_scoring_cuda_impl<TILE_M, N_EXPS, 4><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        case 8: moe_scoring_cuda_impl<TILE_M, N_EXPS, 8><<<grid, block, 0, stream>>>(M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr); break;
        default: TORCH_CHECK(false, "unsupported top_k: ", top_k);
    }
}

// NOTE: ffn_gate_inp seems to be kept in half precision.
// need to handle in compute_tiles and elsewhere properly
void moe_scoring_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x, // [B,L,hidden_dim]
    const at::Tensor& w, // [n_experts, hidden_dim]
    at::Tensor& act_exps, // [B,L,n_act_exps]
    at::Tensor& act_exps_scores, // [B,L,n_act_exps]
    at::Tensor& scores // [B,L,n_experts]
) {
    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(scores.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(act_exps.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(act_exps_scores.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(scores.is_contiguous());
    TORCH_CHECK(w.is_contiguous());
    TORCH_CHECK(act_exps.is_contiguous());
    TORCH_CHECK(act_exps_scores.is_contiguous());

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(scores.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kHalf);
    TORCH_CHECK(act_exps.dtype() == at::kByte);
    TORCH_CHECK(act_exps_scores.dtype() == at::kHalf);

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
    uint8_t* act_exps_ptr = act_exps.data_ptr<uint8_t>();
    half* act_exps_scores_ptr = reinterpret_cast<half*>(act_exps_scores.data_ptr<at::Half>());
    half* scores_ptr = reinterpret_cast<half*>(scores.data_ptr<at::Half>());

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    int64_t n_exps = scores.size(-1);
    int64_t top_k = act_exps.size(-1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (n_exps) {
        case 8:   dispatch_top_k<128, 8>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 16:  dispatch_top_k<64, 16>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 32:  dispatch_top_k<32, 32>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 64:  dispatch_top_k<16, 64>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 128: dispatch_top_k<8, 128>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        case 256: dispatch_top_k<4, 256>(top_k, M, K, x_ptr, w_ptr, act_exps_ptr, act_exps_scores_ptr, scores_ptr, stream); break;
        default: TORCH_CHECK(false, "unsupported n_exps: ", n_exps);
    }
}