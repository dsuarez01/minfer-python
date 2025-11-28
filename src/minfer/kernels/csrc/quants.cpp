#include <Python.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include "quants_impl.hpp"

extern "C" {
    PyObject* PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

namespace minfer {

template <typename T>
void dequant_row_(
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
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

void dequant_row(
    int qtype_int,
    const at::Tensor& x,
    at::Tensor& y,
    int64_t b,
    int64_t k
) {
    TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
    TORCH_CHECK(x.size(0) == y.size(0), "x and y must have the same number of rows");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CPU);

    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);
    const uint8_t* __restrict__ x_ptr = x.data_ptr<uint8_t>();
    int64_t n_rows = x.size(0);

    switch (y.scalar_type()) {
        case at::kFloat: {
            float* __restrict__ y_ptr = y.data_ptr<float>();
            for (int64_t row_idx = 0; row_idx < n_rows; ++row_idx) {
                dequant_row_<float>(qtype, x_ptr+row_idx*b, y_ptr+row_idx*k, k);
            }
            break;
        }
        case at::kHalf: {
            half_t* __restrict__ y_ptr = y.data_ptr<half_t>();
            for (int64_t row_idx = 0; row_idx < n_rows; ++row_idx) {
                dequant_row_<half_t>(qtype, x_ptr+row_idx*b, y_ptr+row_idx*k, k);
            }
            break;
        }
        default: TORCH_CHECK(false, "Expected y scalar dtype to be float32 or float16, got ", y.scalar_type()); break;
    }
}

void quant_row_(
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
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

void quant_row(
    int qtype_int,
    const at::Tensor& x,
    at::Tensor& y,
    int64_t b,
    int64_t n
) {
    TORCH_CHECK(is_valid_qtype(qtype_int), "Invalid qtype: ", qtype_int);
    TORCH_CHECK(x.size(0) == y.size(0), "x and y must have same number of rows");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(y.device().type() == at::DeviceType::CPU);

    GGMLQuantizationType qtype = static_cast<GGMLQuantizationType>(qtype_int);
    uint8_t* __restrict__ y_ptr = y.data_ptr<uint8_t>();
    int64_t n_rows = x.size(0);

    switch (x.scalar_type()) {
        case at::kFloat: {
            const float* __restrict__ x_ptr = x.data_ptr<float>();
            for (int64_t row_idx = 0; row_idx < n_rows; ++row_idx) {
                quant_row_(qtype, x_ptr+row_idx*n, y_ptr+row_idx*b, n);
            }
            break;
        }
        default: TORCH_CHECK(false, "Expected x scalar dtype to be float32, got ", x.scalar_type()); break;
    }
}

TORCH_LIBRARY(minfer, m) {
    m.def("dequant_row(int qtype, Tensor x, Tensor(a!) Tensor y, int b, int k) -> ()");
    m.def("quant_row(int qtype, Tensor x, Tensor(a!) Tensor y, int b, int n) -> ()");
}

TORCH_LIBRARY_IMPL(minfer, CPU, m) {
    m.impl("dequant_row", &dequant_row);
    m.impl("quant_row", &quant_row);
}

static struct Initializer {
    Initializer() {
        // NOTE: 2-3 min to initialize upon first import
        iq2xs_init_impl(GGMLQuantizationType::IQ2_XXS);
        iq2xs_init_impl(GGMLQuantizationType::IQ2_XS);
        iq2xs_init_impl(GGMLQuantizationType::IQ2_S);
        iq2xs_init_impl(GGMLQuantizationType::IQ1_S);
        iq2xs_init_impl(GGMLQuantizationType::IQ1_M);
        iq3xs_init_impl(GGMLQuantizationType::IQ3_XXS); 
        iq3xs_init_impl(GGMLQuantizationType::IQ3_S);
    }
    ~Initializer() {
        iq2xs_free_impl(GGMLQuantizationType::IQ2_XXS);
        iq2xs_free_impl(GGMLQuantizationType::IQ2_XS);
        iq2xs_free_impl(GGMLQuantizationType::IQ1_S);
        iq2xs_free_impl(GGMLQuantizationType::IQ1_M);
        iq2xs_free_impl(GGMLQuantizationType::IQ2_S);
        iq3xs_free_impl(GGMLQuantizationType::IQ3_XXS);
        iq3xs_free_impl(GGMLQuantizationType::IQ3_S);
    }

} init;

}