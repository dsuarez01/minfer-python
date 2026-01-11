#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

void embed_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& token_ids, // [B,L]
    const at::Tensor& w, // [vocab_size, hidden_dim (possibly bytes)]
    at::Tensor& x // [B,L,D]
) {
    TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(token_ids.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(token_ids.dtype() == at::kLong);
    TORCH_CHECK(w.dtype() == at::kByte || w.dtype() == at::kHalf);

    TORCH_CHECK(x.dim() == 3);
    TORCH_CHECK(token_ids.dim() == 2);
    TORCH_CHECK(w.dim() == 2);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(token_ids.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(x.size(0) == token_ids.size(0) && x.size(1) == token_ids.size(1));

    const int64_t* token_ids_ptr = token_ids.data_ptr<int64_t>();
    half* x_ptr = reinterpret_cast<half*>(x.data_ptr<at::Half>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t B = x.size(0);
    size_t L = x.size(1);
    size_t hidden_dim = x.size(2);

    if (qtype == GGMLQuantizationType::F16) {
        TORCH_CHECK(w.size(-1) == x.size(-1));
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());
        dim3 grid(
            static_cast<unsigned int>(B*L)
        );
        embed_f16_cuda_impl<<<grid, 1024u, 0, stream>>>(
            L, hidden_dim, token_ids_ptr, w_ptr, x_ptr
        );
    } else {
        TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size));
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
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
            qtype_int, qblock_size, qtype_size, L, b, k, token_ids_ptr, w_ptr, x_ptr
        );
    }

    AT_CUDA_CHECK(cudaGetLastError());
}