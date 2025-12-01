#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

#include "quants_impl.cuh"

// dequant helpers

__device__ inline float E8M0_TO_FP32_HALF(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    if (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    else {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    return *reintepret_cast<float*>(&bits); // reinterpret cast here should be fine
}

__device__ inline void get_scale_min_k4(int j, const uint8_t * __restrict__ q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template <typename T>
__device__ void dequant_block_q4_0(
    const uint8_t* w,
    T* __restrict__ y,
    int stride,
    int tid,
    int n_threads
) {
    const block_q4_0 * __restrict__ block = reinterpret_cast<const block_q4_0 *>(w);
    constexpr int qk = QK4_0;
    const float d = (float) block->d;

    for (int j = tid; j < qk/2; j += n_threads) {
        const int x0 = (block->qs[j] & 0x0F) - 8;
        const int x1 = (block->qs[j] >> 4) - 8;
        
        y[j*stride] = (T)(x0 * d);
        y[(j+qk/2)*stride] = (T)(x1 * d);
    }
}

template <typename T>
__device__ void dequant_block_q4_1(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y,
    int stride,
    int tid,
    int n_threads
) {
    const block_q4_1 * __restrict__ block = reinterpret_cast<const block_q4_1 *>(w);
    
    constexpr int qk = QK4_1;

    const float d = (float)block->d;
    const float m = (float)block->m;

    for (int j = tid; j < qk/2; j += n_threads) {
        const int x0 = (block->qs[j] & 0x0F);
        const int x1 = (block->qs[j] >>   4);

        y[j*stride] = (T)(x0*d + m);
        y[(j+qk/2)*stride] = (T)(x1*d + m);
    }
}

template <typename T>
__device__ void dequant_block_q5_0(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_q5_0 * __restrict__ block = reinterpret_cast<const block_q5_0 *>(w);
    
    constexpr int qk = QK5_0;

    const float d = (float)block->d;

    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh)); // TODO: fix all of these (reinterpret_cast might not work)

    for (int j = tid; j < qk/2; j += n_threads) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

        const int32_t x0 = ((block->qs[j] & 0x0F) | xh_0) - 16;
        const int32_t x1 = ((block->qs[j] >>   4) | xh_1) - 16;

        y[j*stride] = (T)(x0*d);
        y[(j+qk/2)*stride] = (T)(x1*d);
    }
}

template <typename T>
__device__ void dequant_block_q5_1(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    int stride,
    int tid,
    int n_threads
) {
    const block_q5_1 * __restrict__ block = reinterpret_cast<const block_q5_1 *>(w);
    
    constexpr int qk = QK5_1;

    const float d = (float)block->d;
    const float m = (float)block->m;

    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));

    for (int j = tid; j < qk/2; j += n_threads) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

        const int x0 = (block->qs[j] & 0x0F) | xh_0;
        const int x1 = (block->qs[j] >>   4) | xh_1;

        y[j*stride] = (T)(x0*d + m);
        y[(j+qk/2)*stride] = (T)(x1*d + m);
    }
}

template <typename T>
__device__ void dequant_block_q8_0(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    int stride,
    int tid,
    int n_threads
) {
    const block_q8_0 * __restrict__ block = reinterpret_cast<const block_q8_0 *>(w);
    
    constexpr int qk = QK8_0;

    const float d = (float)block->d;

    for (int j = tid; j < qk; j += n_threads) {
        y[j*stride] = (T)(block->qs[j]*d);
    }
}

template <typename T>
__device__ void dequant_block_mxfp4(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_mxfp4 * __restrict__ block = reinterpret_cast<const block_mxfp4 *>(w);
    
    constexpr int qk = QK_MXFP4;

    const float d = E8M0_TO_FP32_HALF(block->e);

    for (int j = tid; j < qk/2; j += stride) {
        const int8_t x0 = kvalues_mxfp4[block->qs[j] & 0x0F];
        const int8_t x1 = kvalues_mxfp4[block->qs[j] >>   4];

        y[j*stride] = (T)(x0*d);
        y[(j+qk/2)*stride] = (T)(x1*d);
    }
}