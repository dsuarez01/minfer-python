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

void embed_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const torch::stable::Tensor& token_ids, // [B,L]
    const torch::stable::Tensor& w, // [vocab_size, hidden_dim (possibly bytes)]
    torch::stable::Tensor& x // [B,L,D]
) {
    STD_TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(token_ids.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(w.device().type() == torch::headeronly::DeviceType::CUDA);

    STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(token_ids.scalar_type() == torch::headeronly::ScalarType::Long);
    STD_TORCH_CHECK(
        w.scalar_type() == torch::headeronly::ScalarType::Byte || 
        w.scalar_type() == torch::headeronly::ScalarType::Half
    );

    STD_TORCH_CHECK(x.dim() == 3);
    STD_TORCH_CHECK(token_ids.dim() == 2);
    STD_TORCH_CHECK(w.dim() == 2);

    STD_TORCH_CHECK(x.is_contiguous());
    STD_TORCH_CHECK(token_ids.is_contiguous());
    STD_TORCH_CHECK(w.is_contiguous());

    STD_TORCH_CHECK(x.size(0) == token_ids.size(0) && x.size(1) == token_ids.size(1));

    const int64_t* token_ids_ptr = token_ids.const_data_ptr<int64_t>();
    half* x_ptr = reinterpret_cast<half*>(x.mutable_data_ptr<half_t>());

    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
    );
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    size_t B = x.size(0);
    size_t L = x.size(1);
    size_t hidden_dim = x.size(2);

    if (qtype == GGMLQuantizationType::F16) {
        STD_TORCH_CHECK(w.size(-1) == x.size(-1));
        const half* w_ptr = reinterpret_cast<const half*>(w.const_data_ptr<half_t>());
        dim3 grid(
            static_cast<unsigned int>(B*L)
        );
        embed_f16_cuda_impl<<<grid, 1024u, 0, stream>>>(
            L, hidden_dim, token_ids_ptr, w_ptr, x_ptr
        );
    } else {
        STD_TORCH_CHECK(is_dequant_qtype(qtype_int));
        STD_TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size));
        const uint8_t* w_ptr = w.const_data_ptr<uint8_t>();
        unsigned int n_qblocks_per_row = hidden_dim / qblock_size;
        size_t b = w.size(-1); // bytes per row
        size_t k = hidden_dim; // dequant elems per row
        dim3 grid(
            static_cast<unsigned int>(B*L),
            n_qblocks_per_row
        );
        constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(half);
        unsigned int thrs_per_block = qblock_size/ELEMS_PER_THR;
        embed_quant_cuda_impl<<<grid, thrs_per_block, 0, stream>>>(
            qtype_int, qblock_size, static_cast<size_t>(qtype_size), L, b, k, token_ids_ptr, w_ptr, x_ptr
        );
    }

    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}