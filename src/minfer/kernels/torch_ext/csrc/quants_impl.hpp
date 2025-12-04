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
void iq2xs_init_impl(int qtype_int);
void iq3xs_init_impl(int qtype_int);
void iq2xs_free_impl(int qtype_int);
void iq3xs_free_impl(int qtype_int);