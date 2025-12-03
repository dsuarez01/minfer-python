#pragma once

#include <cstdint>

// move these into the global namespace out of convenience
using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

// this assumes impl_common.hpp is in the translation unit from the source file
enum class GGMLQuantizationType : int; 

inline bool is_valid_qtype(int qtype_int) {
    switch (qtype_int) {
        case 0: case 1: case 2: case 3: 
        case 6: case 7: case 8: case 10:
        case 11: case 12: case 13: case 14:
        case 15: case 16: case 17: case 18:
        case 19: case 20: case 21: case 22:
        case 23: case 24: case 25: case 26:
        case 27: case 28: case 29: case 30:
        case 34: case 35: case 39:
            return true;
        default:
            return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// (De)quant Impls.
////////////////////////////////////////////////////////////////////////////////

template <typename T> void dequant_row_q4_0(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q4_1(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q5_0(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q5_1(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q8_0(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_mxfp4(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q2_K(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q3_K(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q4_K(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q5_K(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q6_K(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_tq1_0(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_tq2_0(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq2_xxs(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq2_xs(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq2_s(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq3_xxs(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq3_s(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq1_s(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq1_m(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq4_nl(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_iq4_xs(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);
template <typename T> void dequant_row_q8_K(const uint8_t* __restrict__ xr, T* __restrict__ y, int64_t k);

void quant_row_q4_0(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q4_1(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q5_0(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q5_1(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q8_0(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_mxfp4(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q2_K(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q3_K(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q4_K(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q5_K(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_q6_K(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_tq1_0(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_tq2_0(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);
void quant_row_iq2_xxs(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq2_xs(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq2_s(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq3_xxs(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq3_s(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq1_s(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq1_m(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq4_nl(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_iq4_xs(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n, const float* quant_weights = nullptr);
void quant_row_q8_K(const float* __restrict__ x, uint8_t* __restrict__ yr, int64_t n);

// fcns for init. and clean-up (called in quants.cpp)
void iq2xs_init_impl(GGMLQuantizationType type);
void iq3xs_init_impl(GGMLQuantizationType type);
void iq2xs_free_impl(GGMLQuantizationType type);
void iq3xs_free_impl(GGMLQuantizationType type);