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

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

__device__ inline void get_scale_min_k4(int j, const uint8_t * __restrict__ q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// actual impls

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
    memcpy(&qh, block->qh, sizeof(qh));

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

template <typename T>
__device__ void dequant_block_q2_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_q2_K * __restrict__ block = reinterpret_cast<const block_q2_K *>(w);

    int idx = tid;
    if (idx >= QK_K) return;
    
    int n = idx/128;
    int rem = idx % 128;
    int j = rem / 32;
    int half = (rem / 16) % 2;
    int l = rem % 16;

    int shift = j*2;
    int is = n * 8 + j*2 + half;
    const uint8_t* q = block->qs + n * 32;

    uint8_t sc = block->scales[is];
    const float d = (float) block->d;
    const float min = (float) block->dmin;

    float dl = d * (sc & 0xF);
    float ml = min * (sc >> 4);
    int q_off = half*16 + l;
    y[idx * stride] = (T)(dl * ((int8_t)((q[q_off] >> shift) & 3)) - ml);
}

template <typename T>
__device__ void dequant_block_q3_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_q3_K * __restrict__ block = reinterpret_cast<const block_q3_K *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    // unpacking scales
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, block->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * scales = (const int8_t*)aux;
    
    const float d = (float)block->d;
    
    int n = idx/128;
    int rem = idx%128;
    int j = rem/32;
    int half = (rem/16) % 2;
    int l = rem%16;
    
    int shift = j*2;
    int is = n*8 + j*2 + half;
    const uint8_t* q = block->qs + n*32;
    const uint8_t* hm = block->hmask;
    uint8_t m = 1<<(j*2+half);
    
    float dl = d*(scales[is]-32);
    int q_off = half*16+l;
    int off =  4*((hm[n*32+q_off] & m) == 0);
    y[idx*stride] = (T)(dl * ((int8_t)((q[q_off] >> shift) & 3) - off));
}

template <typename T>
__device__ void dequant_block_q4_K(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    int stride,
    int tid,
    int n_threads
) {
    const block_q4_K * __restrict__ block = reinterpret_cast<const block_q4_K *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    const float min = (float)block->dmin;
    
    int j = idx/64;
    int rem = idx%64;
    int half = rem/32;
    int shift = half*4;
    int l = rem % 32;
    
    int is = j*2 + half;
    const uint8_t* q = block->qs + j*32;
    
    uint8_t sc, m;
    get_scale_min_k4(is, block->scales, &sc, &m);
    
    float dl = d*sc;
    float ml = min*m;
    
    y[idx*stride] = (T)(dl * ((q[l] >> shift) & 0xF) - ml);
}

template <typename T>
__device__ void dequant_block_q5_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_q5_K * __restrict__ block = reinterpret_cast<const block_q5_K *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    const float min = (float)block->dmin;
    
    int j = idx/64;
    int rem = idx%64;
    int half = rem/32;
    int shift = half*4;
    int l = rem%32;
    
    int is = j*2 + half;
    const uint8_t* ql = block->qs + j*32;
    const uint8_t* qh = block->qh;
    
    uint8_t sc, m;
    get_scale_min_k4(is, block->scales, &sc, &m);
    
    float dl = d*sc;
    float ml = min*m;
    
    uint8_t u_shift =  j*2+half;
    uint8_t u = 1 << u_shift;
    int high_bit = ((qh[j*32+l] & u) != 0) << 4;
    
    y[idx*stride] = (T)(dl * (((ql[l] >> shift) & 0xF) + high_bit) - ml);
}

template <typename T>
__device__ void dequant_block_q6_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_q6_K * __restrict__ block = reinterpret_cast<const block_q6_K *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int n = idx/128;
    int l = idx%128;
    
    int chunk = (l/32)%4;
    int pos = l%32;
    
    const uint8_t* ql = block->ql + n*64;
    const uint8_t* qh = block->qh + n*32;
    const int8_t* sc = block->scales + n*8;
    
    int is = pos/16;
    
    int ql_offset = (chunk & 1)*32;
    int ql_shift = (chunk >> 1)*4;
    int qh_shift = chunk*2;
    int sc_offset = chunk*2;
    
    int8_t q = (int8_t)(((ql[pos+ql_offset] >> ql_shift) & 0xF) | (((qh[pos] >> qh_shift) & 3) << 4)) - 32;
    
    y[idx * stride] = (T)(d * sc[is+sc_offset] * q);
}

template <typename T>
__device__ void dequant_block_tq1_0(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_tq1_0 * __restrict__ block = reinterpret_cast<const block_tq1_0 *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    
    constexpr int qs_total = sizeof(block->qs) * 5;
    
    uint8_t q_byte;
    int n;
    
    if (idx < qs_total) {
        n = idx%5;
        int byte_idx = idx/5;
        q_byte = block->qs[byte_idx];
    } else {
        int qh_idx = idx-qs_total;
        n = qh_idx/sizeof(block->qh);
        int byte_idx = qh_idx%sizeof(block->qh);
        q_byte = block->qh[byte_idx];
    }
    
    uint8_t q = q_byte*pow3[n];
    int16_t xi = ((uint16_t)q*3) >> 8;
    y[idx*stride] = (T)((float)(xi-1) * d);
}

template <typename T>
__device__ void dequant_block_tq2_0(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_tq2_0 * __restrict__ block = reinterpret_cast<const block_tq2_0 *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int l = idx%4;
    int base_idx = idx/4;
    int j = (base_idx/32)*32;
    int m = base_idx%32;
    
    int8_t q = (block->qs[j+m] >> (l*2)) & 3;
    y[idx*stride] = (T)((float)(q-1)*d);
}

template <typename T>
__device__ void dequant_block_iq2_xxs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq2_xxs * __restrict__ block = reinterpret_cast<const block_iq2_xxs *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int ib32 = idx/32;
    int pos = idx%32;
    
    uint32_t aux32[2];
    memcpy(aux32, block->qs + 4*ib32, 2*sizeof(uint32_t));
    const uint8_t * aux8 = (const uint8_t *)aux32;
    
    const float db = d*(0.5f + (aux32[1] >> 28))*0.25f;
    
    int l = pos/8;
    int j = pos%8;
    
    const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
    const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
    float sign = 1.f - 2.f * (float)((signs & kmask_iq2xs[j]) != 0);

    y[idx*stride] = (T)(db * grid[j] * sign);
}

template <typename T>
__device__ void dequant_block_iq2_xs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq2_xs * __restrict__ block = reinterpret_cast<const block_iq2_xs *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int ib32 = idx/32;
    int pos = idx%32;
    
    float db[2];
    db[0] = d * (0.5f + (block->scales[ib32] & 0xf)) * 0.25f;
    db[1] = d * (0.5f + (block->scales[ib32] >> 4)) * 0.25f;
    
    int l = pos/8;
    int j = pos%8;
    
    const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (block->qs[4*ib32 + l] & 511));
    const uint8_t signs = ksigns_iq2xs[block->qs[4*ib32 + l] >> 9];
    float sign = 1.f - 2.f * (float)((signs & kmask_iq2xs[j]) != 0);
    
    y[idx*stride] = (T)(db[l/2] * grid[j] * sign);
}

template <typename T>
__device__ void dequant_block_iq2_s(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq2_s * __restrict__ block = reinterpret_cast<const block_iq2_s *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int ib32 = idx/32;
    int pos = idx%32;
    
    float db[2];
    db[0] = d * (0.5f + (block->scales[ib32] & 0xf)) * 0.25f;
    db[1] = d * (0.5f + (block->scales[ib32] >> 4)) * 0.25f;
    
    const uint8_t * qs = block->qs + ib32*4;
    const uint8_t * qh = block->qh;
    const uint8_t * signs = block->qs + QK_K/8 + ib32*4;
    
    int l = pos/8;
    int j = pos%8;
    
    const float dl = db[l/2];
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
    float sign = 1.f - 2.f * (float)((signs & kmask_iq2xs[j]) != 0);

    y[idx*stride] = (T)(dl * grid[j] * sign);
}

template <typename T>
__device__ void dequant_block_iq3_xxs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq3_xxs * __restrict__ block = reinterpret_cast<const block_iq3_xxs *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    const uint8_t * qs = block->qs;
    const uint8_t * scales_and_signs = qs + QK_K/4;
    
    int ib32 = idx/32;
    int pos = idx%32;
    
    uint32_t aux32;
    memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
    const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
    
    int l = pos/8;
    int j = pos%8;
    
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
    const uint8_t * grid_qs = qs + ib32*8;
    
    int grid_idx = 2*l + (j >> 2);
    const uint8_t * grid = (const uint8_t *)(iq3xxs_grid + grid_qs[grid_idx]);
    int j_offset = j & 3;
    float sign = 1.f - 2.f * (float)((signs & kmask_iq2xs[j]) != 0);

    y[idx*stride] = (T)(db * grid[j_offset] * sign);
}

template <typename T>
__device__ void dequant_block_iq3_s(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq3_s * __restrict__ block = reinterpret_cast<const block_iq3_s *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int ib32 = idx/32;
    int pos = idx%32;
    
    const float db = d * (1 + 2*(block->scales[ib32/2] >> (4*(ib32%2))));
    
    const uint8_t * qs = block->qs + (ib32/2) * 16 + (ib32%2) * 8;
    const uint8_t * qh = block->qh + (ib32/2) * 2 + (ib32%2);
    const uint8_t * signs = block->signs + (ib32/2) * 8 + (ib32%2) * 4;
    
    int l = pos/8;
    int j = pos%8;

    int grid_idx = 2*l + (j >> 2);
    const uint8_t * grid = (const uint8_t *)(iq3s_grid + (qs[grid_idx] | ((qh[0] << (8 - 2*l - (j >> 2))) & 256)));
    int j_offset = j & 3;
    float sign = 1.f - 2.f * (float)((signs[l] & kmask_iq2xs[j]) != 0);

    y[idx*stride] = (T)(db * grid[j_offset] * sign);
}

template <typename T>
__device__ void dequant_block_iq1_s(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq1_s * __restrict__ block = reinterpret_cast<const block_iq1_s *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int ib = idx/32;
    int pos = idx%32;
    
    const uint16_t * qh = block->qh;
    const uint8_t * qs = block->qs + ib * 4;
    
    const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
    const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
    
    int l = pos/8;
    int j = pos%8;
    
    const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
    
    y[idx*stride] = (T)(dl * (grid[j] + delta));
}

template <typename T>
__device__ void dequant_block_iq1_m(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq1_m * __restrict__ block = reinterpret_cast<const block_iq1_m *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const uint16_t * sc = (const uint16_t *)block->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = (float)scale.f16;
    
    int ib = idx/32;
    int pos = idx%32;
    
    const float dl = d * (2*((sc[ib/2] >> (6*(ib%2) + 3*(pos/16))) & 0x7) + 1);
    
    const uint8_t * qs = block->qs + ib*4;
    const uint8_t * qh = block->qh + ib*2;
    
    int half = pos >> 4;
    int l = (pos&15)/8;
    int j = pos%8;
    
    uint16_t idx_grid = qs[l + half*2] | ((qh[half] << 8) & 0x700);
    float delta = IQ1S_DELTA * (1.f - 2.f * (float)((qh[half] & 0x08) != 0));

    const int8_t * grid = (const int8_t *)(iq1s_grid + idx_grid);
    y[idx*stride] = (T)(dl * (grid[j] + delta));
}

template <typename T>
__device__ void dequant_block_iq4_nl(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq4_nl * __restrict__ block = reinterpret_cast<const block_iq4_nl *>(w);
    
    int idx = tid;
    if (idx >= QK4_NL) return;
    
    const float d = (float)block->d;
    const uint8_t * qs = block->qs;
    
    int half = idx/(QK4_NL/2);
    int j = idx%(QK4_NL/2);
    
    int shift = half*4;
    y[idx*stride] = (T)(d * kvalues_iq4_nl[(qs[j] >> shift) & 0xf]);
}

template <typename T>
__device__ void dequant_block_iq4_xs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_iq4_xs * __restrict__ block = reinterpret_cast<const block_iq4_xs *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    const float d = (float)block->d;
    
    int ib = idx/32;
    int pos = idx%32;
    
    const int ls = ((block->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((block->scales_h >> 2*ib) & 3) << 4);
    const float dl = d*(ls-32);
    
    const uint8_t * qs = block->qs + ib*16;
    int j = pos%16;
    int half = pos/16;
    int shift = half*4;
    
    y[idx*stride] = (T)(dl * kvalues_iq4_nl[(qs[j] >> shift) & 0xf]);
}

template <typename T>
__device__ void dequant_block_q8_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    int stride,
    int tid,
    int n_threads
) {
    const block_q8_K * __restrict__ block = reinterpret_cast<const block_q8_K *>(w);
    
    int idx = tid;
    if (idx >= QK_K) return;
    
    y[idx*stride] = (T)(block->d * block->qs[idx]);
}