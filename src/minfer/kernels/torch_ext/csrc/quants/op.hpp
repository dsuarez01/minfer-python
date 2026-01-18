#pragma once

#include <torch/types.h>

void dequant_cpu(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x,
    at::Tensor& y
);

void quant_cpu(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x,
    at::Tensor& y
);