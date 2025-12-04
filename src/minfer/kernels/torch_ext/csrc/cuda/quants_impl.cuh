#pragma once

#include <cuda_runtime.h>

#include <cstdint>
using std::uint8_t;

template <typename T> __device__ void dequant_block_q4_0(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q4_1(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q5_0(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q5_1(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q8_0(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_mxfp4(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q2_K(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q3_K(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q4_K(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q5_K(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q6_K(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_tq1_0(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_tq2_0(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq2_xxs(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq2_xs(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq2_s(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq3_xxs(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq3_s(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq1_s(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq1_m(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq4_nl(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_iq4_xs(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);
template <typename T> __device__ void dequant_block_q8_K(const uint8_t* __restrict__ w, T* __restrict__ y, int64_t stride, int tid, int n_threads);