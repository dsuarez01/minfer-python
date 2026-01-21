#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <omp.h>

#include "common/types.hpp"
#include "impl.hpp"

namespace {
    using namespace minfer::impl;

    struct Initializer {

        Initializer() {
            // NOTE: 2-3 min to init
            iq2xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ2_XXS));
            iq2xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ2_XS));
            iq2xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ2_S));
            iq2xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ1_S));
            iq2xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ1_M));
            iq3xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ3_XXS));
            iq3xs_init_impl(static_cast<int>(GGMLQuantizationType::IQ3_S));
        }
        ~Initializer() {
            iq2xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ2_XXS));
            iq2xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ2_XS));
            iq2xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ1_S));
            iq2xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ1_M));
            iq2xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ2_S));
            iq3xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ3_XXS));
            iq3xs_free_impl(static_cast<int>(GGMLQuantizationType::IQ3_S));
        }

    } init;

    template <typename T>
    void dequant_row_cpu(
        GGMLQuantizationType qtype,
        const uint8_t* __restrict__ xr,
        T* __restrict__ y,
        int64_t k
    ) {
        switch (qtype) {
            case GGMLQuantizationType::Q4_0: dequant_row_q4_0<T>(xr, y, k); break;
            case GGMLQuantizationType::Q4_1: dequant_row_q4_1<T>(xr, y, k); break;
            case GGMLQuantizationType::Q5_0: dequant_row_q5_0<T>(xr, y, k); break;
            case GGMLQuantizationType::Q5_1: dequant_row_q5_1<T>(xr, y, k); break;
            case GGMLQuantizationType::Q8_0: dequant_row_q8_0<T>(xr, y, k); break;
            case GGMLQuantizationType::MXFP4: dequant_row_mxfp4<T>(xr, y, k); break;
            case GGMLQuantizationType::Q2_K: dequant_row_q2_K<T>(xr, y, k); break;
            case GGMLQuantizationType::Q3_K: dequant_row_q3_K<T>(xr, y, k); break;
            case GGMLQuantizationType::Q4_K: dequant_row_q4_K<T>(xr, y, k); break;
            case GGMLQuantizationType::Q5_K: dequant_row_q5_K<T>(xr, y, k); break;
            case GGMLQuantizationType::Q6_K: dequant_row_q6_K<T>(xr, y, k); break;
            case GGMLQuantizationType::TQ1_0: dequant_row_tq1_0<T>(xr, y, k); break;
            case GGMLQuantizationType::TQ2_0: dequant_row_tq2_0<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ2_XXS: dequant_row_iq2_xxs<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ2_XS: dequant_row_iq2_xs<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ2_S: dequant_row_iq2_s<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ3_XXS: dequant_row_iq3_xxs<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ3_S: dequant_row_iq3_s<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ1_S: dequant_row_iq1_s<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ1_M: dequant_row_iq1_m<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ4_NL: dequant_row_iq4_nl<T>(xr, y, k); break;
            case GGMLQuantizationType::IQ4_XS: dequant_row_iq4_xs<T>(xr, y, k); break;
            case GGMLQuantizationType::Q8_K: dequant_row_q8_K<T>(xr, y, k); break;
        }
    }

    void quant_row_cpu(
        GGMLQuantizationType qtype,
        const float* __restrict__ x,
        uint8_t* __restrict__ yr,
        int64_t n
    ) {
        switch (qtype) {
            case GGMLQuantizationType::Q4_0: quant_row_q4_0(x, yr, n); break;
            case GGMLQuantizationType::Q4_1: quant_row_q4_1(x, yr, n); break;
            case GGMLQuantizationType::Q5_0: quant_row_q5_0(x, yr, n); break;
            case GGMLQuantizationType::Q5_1: quant_row_q5_1(x, yr, n); break;
            case GGMLQuantizationType::Q8_0: quant_row_q8_0(x, yr, n); break;
            case GGMLQuantizationType::MXFP4: quant_row_mxfp4(x, yr, n); break;
            case GGMLQuantizationType::Q2_K: quant_row_q2_K(x, yr, n); break;
            case GGMLQuantizationType::Q3_K: quant_row_q3_K(x, yr, n); break;
            case GGMLQuantizationType::Q4_K: quant_row_q4_K(x, yr, n); break;
            case GGMLQuantizationType::Q5_K: quant_row_q5_K(x, yr, n); break;
            case GGMLQuantizationType::Q6_K: quant_row_q6_K(x, yr, n); break;
            case GGMLQuantizationType::TQ1_0: quant_row_tq1_0(x, yr, n); break;
            case GGMLQuantizationType::TQ2_0: quant_row_tq2_0(x, yr, n); break;
            case GGMLQuantizationType::IQ2_XXS: quant_row_iq2_xxs(x, yr, n); break;
            case GGMLQuantizationType::IQ2_XS: quant_row_iq2_xs(x, yr, n); break;
            case GGMLQuantizationType::IQ2_S: quant_row_iq2_s(x, yr, n); break;
            case GGMLQuantizationType::IQ3_XXS: quant_row_iq3_xxs(x, yr, n); break;
            case GGMLQuantizationType::IQ3_S: quant_row_iq3_s(x, yr, n); break;
            case GGMLQuantizationType::IQ1_S: quant_row_iq1_s(x, yr, n); break;
            case GGMLQuantizationType::IQ1_M: quant_row_iq1_m(x, yr, n); break;
            case GGMLQuantizationType::IQ4_NL: quant_row_iq4_nl(x, yr, n); break;
            case GGMLQuantizationType::IQ4_XS: quant_row_iq4_xs(x, yr, n); break;
            case GGMLQuantizationType::Q8_K: quant_row_q8_K(x, yr, n); break;
        }
    }
}

namespace minfer {
    
    // NOTE: only call on 2D tensors
    void dequant_cpu(
        int64_t qtype_int,
        int64_t qblock_size, // num deq elems in block
        int64_t qtype_size, // byte size of block
        const torch::stable::Tensor& x,
        torch::stable::Tensor& y
    ) {

        STD_TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
        STD_TORCH_CHECK(is_dequant_qtype(qtype_int));
        STD_TORCH_CHECK(x.dim() == 2 && y.dim() == 2);
        STD_TORCH_CHECK(x.size(0) == y.size(0), "x and y must have the same number of rows");
        STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Byte, "x must be uint8 (byte)");
        STD_TORCH_CHECK(y.scalar_type() == torch::headeronly::ScalarType::Float || y.scalar_type() == torch::headeronly::ScalarType::Half, "y must be float32 or float16");
        STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        STD_TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

        STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CPU);
        STD_TORCH_CHECK(y.device().type() == torch::headeronly::DeviceType::CPU);

        GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);
        const uint8_t* __restrict__ x_ptr = x.const_data_ptr<uint8_t>();
        int n_rows = x.size(0);
        int b = x.size(-1);
        int k = y.size(-1);

        switch (y.scalar_type()) {
            case torch::headeronly::ScalarType::Float: {
                auto* __restrict__ y_ptr = y.mutable_data_ptr<float>();
                #pragma omp parallel for
                for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
                    dequant_row_cpu<float>(qtype, x_ptr+row_idx*b, y_ptr+row_idx*k, k);
                }
                break;
            }
            case torch::headeronly::ScalarType::Half: {
                auto* __restrict__ y_ptr = y.mutable_data_ptr<half_t>();
                #pragma omp parallel for
                for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
                    dequant_row_cpu<half_t>(qtype, x_ptr+row_idx*b, y_ptr+row_idx*k, k);
                }
                break;
            }
            default: STD_TORCH_CHECK(false, "Expected y scalar dtype to be float32 or float16, got ", y.scalar_type()); break;
        }
    }

    // NOTE: only call on 2D tensors for right now, 
    // no need for larger since this is just to test the impl
    void quant_cpu(
        int64_t qtype_int,
        int64_t qblock_size, // num deq elems in block
        int64_t qtype_size, // byte size of block
        const torch::stable::Tensor& x,
        torch::stable::Tensor& y
    ) {

        STD_TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
        STD_TORCH_CHECK(x.dim() == 2 && y.dim() == 2);
        STD_TORCH_CHECK(x.size(0) == y.size(0), "x and y must have same number of rows");
        STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Float, "x must be float32");
        STD_TORCH_CHECK(y.scalar_type() == torch::headeronly::ScalarType::Byte, "y must be uint8 (byte)");
        STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
        STD_TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

        STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CPU);
        STD_TORCH_CHECK(y.device().type() == torch::headeronly::DeviceType::CPU);

        GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);
        uint8_t* __restrict__ y_ptr = y.mutable_data_ptr<uint8_t>();
        int n_rows = x.size(0);
        int64_t n = x.size(-1);
        int64_t b = y.size(-1);

        switch (x.scalar_type()) {
            case torch::headeronly::ScalarType::Float: {
                const auto* __restrict__ x_ptr = x.const_data_ptr<float>();
                #pragma omp parallel for
                for (int row_idx = 0; row_idx < n_rows; ++row_idx) {
                    quant_row_cpu(qtype, x_ptr+row_idx*n, y_ptr+row_idx*b, n);
                }
                break;
            }
            default: STD_TORCH_CHECK(false, "Expected x scalar dtype to be float32, got ", x.scalar_type()); break;
        }
    }
}