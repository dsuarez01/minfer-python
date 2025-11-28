#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>
// #include <ATen/cuda/CUDAContext.h>

#include "quants_impl.cuh"


namespace minfer {

// TODO: complete me!
void dequant_row_cuda(
    int qtype_int, 
    const at::Tensor& x, 
    at::Tensor& y, 
    int64_t b, 
    int64_t k
) {
    // TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
    TORCH_CHECK(x.size(0) == y.size(0), "x and y must have the same number of rows");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous")
    TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
    TORCH_CHECK(y.dtype() == at::kByte, "y must be uint8 (byte)");

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CUDA);
    
    TORCH_CHECK(false, "_dequant_row not implemented");
}

TORCH_LIBRARY_IMPL(minfer, CUDA, m) {
    m.impl("dequant_row", &dequant_row_cuda);
}

}