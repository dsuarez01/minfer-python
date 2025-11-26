#include <torch/extension.h>
#include "quants_impl.hpp"

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
    GGMLQuantizationType qtype,
    torch::Tensor x,
    torch::Tensor y,
    int64_t row_idx,
    int64_t b,
    int64_t k
) {

    const uint8_t* __restrict__ x_ptr = x.data_ptr<uint8_t>();

    switch (y.scalar_type()) {
        case torch::kFloat32: {
            float* __restrict__ y_ptr = y.data_ptr<float>();
            dequant_row_<float>(qtype, x_ptr+row_idx*b, y_ptr+row_idx*k, k);
            break;
        }
        case torch::kFloat16: {
            half_t* __restrict__ y_ptr = y.data_ptr<half_t>();
            dequant_row_<half_t>(qtype, x_ptr+row_idx*b, y_ptr+row_idx*k, k);
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
    GGMLQuantizationType qtype,
    torch::Tensor x,
    torch::Tensor y,
    int64_t row_idx,
    int64_t b,
    int64_t n
) {

    uint8_t* __restrict__ y_ptr = y.data_ptr<uint8_t>();

    switch (x.scalar_type()) {
        case torch::kFloat32: {
            const float* __restrict__ x_ptr = x.data_ptr<float>();
            quant_row_(qtype, x_ptr+row_idx*n, y_ptr+row_idx*b, n);
            break;
        }
        default: TORCH_CHECK(false, "Expected x scalar dtype to be float32, got ", x.scalar_type()); break;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_row", &dequant_row);
    m.def("quant_row", &quant_row);
    
    // for clean-up once python exits
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
        iq2xs_free_impl(GGMLQuantizationType::IQ2_XXS);
        iq2xs_free_impl(GGMLQuantizationType::IQ2_XS);
        iq2xs_free_impl(GGMLQuantizationType::IQ1_S);
        iq2xs_free_impl(GGMLQuantizationType::IQ1_M);
        iq2xs_free_impl(GGMLQuantizationType::IQ2_S);
        iq3xs_free_impl(GGMLQuantizationType::IQ3_XXS);
        iq3xs_free_impl(GGMLQuantizationType::IQ3_S);
    }));
}