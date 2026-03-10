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

// elemwise multiplication of up proj tile with swiglu (tile of swish(gate(x))), then apply downproj
void ffn_cuda(
    int64_t up_qtype_int, int64_t gate_qtype_int, int64_t down_qtype_int,
    int64_t up_qblock_size, int64_t gate_qblock_size, int64_t down_qblock_size,
    int64_t up_qtype_size, int64_t gate_qtype_size, int64_t down_qtype_size,
    const torch::stable::Tensor& in, // [B,L,hidden_dim]
    const torch::stable::Tensor& ws_up, // [n_local_exps, mlp_dim, hidden_dim]
    const torch::stable::Tensor& ws_gate, // // [n_local_exps, mlp_dim, hidden_dim]
    const torch::stable::Tensor& ws_down, // [n_local_exps, hidden_dim, mlp_dim]
    torch::stable::Tensor& hb, // [n_local_exps,B,L,mlp_dim]
    torch::stable::Tensor& hb2, // [n_local_exps,B,L,mlp_dim]
    torch::stable::Tensor& out // [n_local_exps,B,L,hidden_dim]
) {
    // TODO: implement
}

}