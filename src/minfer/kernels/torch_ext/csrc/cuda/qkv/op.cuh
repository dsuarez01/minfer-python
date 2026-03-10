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
    // TO-DO: implement
}

}