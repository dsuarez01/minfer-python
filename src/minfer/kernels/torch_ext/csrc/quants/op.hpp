#pragma once

#include <torch/csrc/stable/tensor_struct.h>

namespace minfer {
    void dequant_cpu(
        int64_t qtype_int,
        int64_t qblock_size,
        int64_t qtype_size,
        const torch::stable::Tensor& x,
        torch::stable::Tensor& y
    );

    void quant_cpu(
        int64_t qtype_int,
        int64_t qblock_size,
        int64_t qtype_size,
        const torch::stable::Tensor& x,
        torch::stable::Tensor& y
    );
}