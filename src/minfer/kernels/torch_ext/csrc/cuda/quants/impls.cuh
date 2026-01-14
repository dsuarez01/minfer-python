#pragma once

#include <cuda_fp16.h>

#include "common/types.hpp"
#include "helpers.cuh"

// actual impls

template <typename T>
__device__ __forceinline__ void dequant_block_q4_0(
    const uint8_t* w,
    T* __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK4_0) return;

    T buf[ELEMS_PER_THR];

    const block_q4_0 * __restrict__ block = reinterpret_cast<const block_q4_0 *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int x = ((block->qs[idx%(QK4_0/2)] >> 4*(idx/(QK4_0/2))) & 0x0F);
        buf[i] = convert_float<T>((x-8)*d);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q4_1(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK4_1) return;
    
    T buf[ELEMS_PER_THR];

    const block_q4_1 * __restrict__ block = reinterpret_cast<const block_q4_1 *>(w);
    float d = __half2float(block->d);
    float m = __half2float(block->m);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int x = ((block->qs[idx%(QK4_1/2)] >> 4*(idx/(QK4_1/2))) & 0x0F);
        buf[i] = convert_float<T>(x*d+m);  
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q5_0(
    const uint8_t * __restrict__ w,
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK5_0) return;

    T buf[ELEMS_PER_THR];
    
    const block_q5_0 * __restrict__ block = reinterpret_cast<const block_q5_0 *>(w);
    float d = __half2float(block->d);
    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        uint8_t xh = ((qh >> idx) << 4) & 0x10;
        int32_t x = ((block->qs[idx%(QK5_0/2)] >> 4*(idx/(QK5_0/2))) & 0x0F) | xh;   
        buf[i] = convert_float<T>((x-16)*d);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q5_1(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK5_1) return;

    T buf[ELEMS_PER_THR];

    const block_q5_1 * __restrict__ block = reinterpret_cast<const block_q5_1 *>(w);
    float d = __half2float(block->d);
    float m = __half2float(block->m);

    uint32_t qh;
    memcpy(&qh, block->qh, sizeof(qh));

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        uint8_t xh = ((qh >> idx) << 4) & 0x10;
        int x = ((block->qs[idx%(QK5_1/2)] >> 4*(idx/(QK5_1/2))) & 0x0F) | xh;
        buf[i] = convert_float<T>(x*d+m);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q8_0(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK8_0) return;

    T buf[ELEMS_PER_THR];

    const block_q8_0 * __restrict__ block = reinterpret_cast<const block_q8_0 *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        buf[i] = convert_float<T>(block->qs[idx]*d);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_mxfp4(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_MXFP4) return;

    T buf[ELEMS_PER_THR];

    const block_mxfp4 * __restrict__ block = reinterpret_cast<const block_mxfp4 *>(w);
    float d = E8M0_TO_FP32_HALF(block->e);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx=base_idx+i;
        int8_t x = kvalues_mxfp4[((block->qs[idx%(QK_MXFP4/2)] >> 4*(idx/(QK_MXFP4/2))) & 0x0F)];
        buf[i] = convert_float<T>(x*d);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q2_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_q2_K * __restrict__ block = reinterpret_cast<const block_q2_K *>(w);
    float d = __half2float(block->d);
    float min = __half2float(block->dmin);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int n = idx/(32*4);
        int j = (idx/32)%4;
        int l = idx%32;

        const uint8_t * q = block->qs + n*32;

        int is = n*8 + j*2 + l/16;
        uint8_t sc = block->scales[is];

        float dl = d*(sc&0xF);
        float ml = min*(sc>>4);

        int8_t x = (q[l] >> 2*j) & 3;
        buf[i] = convert_float<T>(dl*x-ml);
    }
    
    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q3_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_q3_K * __restrict__ block = reinterpret_cast<const block_q3_K *>(w);
    // unpacking scales
    uint32_t kmask1 = 0x03030303;
    uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, block->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t * scales = (const int8_t *)aux;
    
    float d_all = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int n = idx/(32*4);
        int j = (idx/32)%4;
        int l = idx%32;
        
        int is = n*8+j*2+l/16;

        const uint8_t * q = block->qs + n*32;
        const uint8_t * hm = block->hmask;

        uint8_t m = 1<<(4*n+j);
        
        float dl = d_all*(scales[is]-32);

        int hbit_off =  4*((hm[l] & m) == 0);
        int8_t x = (q[l] >> 2*j) & 3;
        buf[i] = convert_float<T>(dl*(x-hbit_off));
    }
    
    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q4_K(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_q4_K * __restrict__ block = reinterpret_cast<const block_q4_K *>(w);
    float d = __half2float(block->d);
    float min = __half2float(block->dmin);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int j = idx/64;
        int l = idx%64;

        const uint8_t * q = block->qs + j*32;
        
        int is = j*2+l/32;

        uint8_t sc, m;
        get_scale_min_k4(is, block->scales, &sc, &m);
        
        float dl = d*sc;
        float ml = min*m;
        buf[i] = convert_float<T>(dl*((q[l%32] >> 4*(l/32))&0xF)-ml);
    }
    
    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q5_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];
    
    const block_q5_K * __restrict__ block = reinterpret_cast<const block_q5_K *>(w);
    float d = __half2float(block->d);
    float min = __half2float(block->dmin);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int j = idx/64;
        int l = idx%64;
        
        const uint8_t * ql = block->qs + j*32;
        const uint8_t * qh = block->qh;

        int is = j*2+l/32;
        
        uint8_t sc, m;
        
        get_scale_min_k4(is, block->scales, &sc, &m);
        float dl = d*sc; float ml = min*m;

        uint8_t u = (1+l/32)<<(j*2);
        int hbit_off = ((qh[l%32] & u) != 0) << 4;
        buf[i] = convert_float<T>(dl * (((ql[l%32]>>4*(l/32)) & 0xF) + hbit_off) - ml);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q6_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_q6_K * __restrict__ block = reinterpret_cast<const block_q6_K *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        int n = idx/(4*32);
        int j = (idx/32)%4; // middle loop
        int l = idx%32;

        const uint8_t * ql = block->ql + n*64;
        const uint8_t * qh = block->qh + n*32;
        const int8_t * sc = block->scales + n*8;
        
        int is = l/16;
        int8_t q = (int8_t)(((ql[l+32*(j%2)] >> 4*(j/2)) & 0xF) | (((qh[l] >> (j*2)) & 3) << 4)) - 32;
        buf[i] = convert_float<T>(d * sc[is+j*2] * q);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_tq1_0(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_tq1_0 * __restrict__ block = reinterpret_cast<const block_tq1_0 *>(w);
    
    constexpr int qs_elem_total = sizeof(block->qs) * 5; // evals to 240
    constexpr int qs_elem_cutoff = (sizeof(block->qs)-sizeof(block->qs)%32) * 5; // evals to 160
    constexpr int qh_size = sizeof(block->qh); // evals to 4
    
    float d = __half2float(block->d);
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    uint8_t q_byte;
    int n;

    // TO-DO: potential issue with warp divergence, profile to see how slow?
    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        if (idx < qs_elem_cutoff) { // elems 0 thru 159
            n = idx/32;
            int j = 0;
            int m = idx%32;
            q_byte = block->qs[j+m];
        } else if (idx < qs_elem_total) { // elems 160 thru 239
            int offset = idx-qs_elem_cutoff;
            n = offset/16;
            int j = 32;
            int m = offset%16;
            q_byte = block->qs[j+m];
        } else { // elems 240 thru 255
            int offset = idx-qs_elem_total;
            n = offset/qh_size;
            int j = offset%qh_size;
            q_byte = block->qh[j];
        }
        
        uint8_t q = q_byte*pow3[n];
        int16_t xi = ((uint16_t)q*3) >> 8;

        buf[i] = convert_float<T>((float)(xi-1) * d);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_tq2_0(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_tq2_0 * __restrict__ block = reinterpret_cast<const block_tq2_0 *>(w);
    float d = __half2float(block->d);


    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int j = (idx/(32*4))*32;
        int l = (idx/32)%4;
        int m = idx%32;
        
        int8_t q = (block->qs[j+m] >> (l*2)) & 3;

        buf[i] = convert_float<T>((float)(q-1)*d);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq2_xxs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq2_xxs * __restrict__ block = reinterpret_cast<const block_iq2_xxs *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib32 = idx/(8*4);
        int l = (idx/8)%4;
        int j = idx%8;
        
        uint32_t aux32[2];
        memcpy(aux32, block->qs + 4*ib32, 2*sizeof(uint32_t));
        const uint8_t * aux8 = (const uint8_t *)aux32;
        
        float db = d*(0.5f + (aux32[1] >> 28))*0.25f;
        
        const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
        uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
        float sign = signs & kmask_iq2xs[j] ? -1.f : 1.f;

        buf[i] = convert_float<T>(db * grid[j] * sign);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq2_xs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq2_xs * __restrict__ block = reinterpret_cast<const block_iq2_xs *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib32 = idx/(8*4);
        int l = (idx/8)%4;
        int j = idx%8;
        
        float db[2];
        db[0] = d * (0.5f + (block->scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (block->scales[ib32] >> 4)) * 0.25f;
        
        const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (block->qs[4*ib32+l] & 511));
        uint8_t signs = ksigns_iq2xs[block->qs[4*ib32+l] >> 9];
        float sign = signs & kmask_iq2xs[j] ? -1.f : 1.f;

        buf[i] = convert_float<T>(db[l/2] * grid[j] * sign);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq2_s(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq2_s * __restrict__ block = reinterpret_cast<const block_iq2_s *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib32 = idx/32;
        int l = (idx/8)%4;
        int j = idx%8;
        
        float db[2];
        db[0] = d * (0.5f + (block->scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (block->scales[ib32] >> 4)) * 0.25f;
        
        const uint8_t * qs = block->qs + ib32*4;
        const uint8_t * qh = block->qh;
        const uint8_t * signs = block->qs + QK_K/8 + ib32*4;
        
        const float dl = db[l/2];
        const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
        float sign = signs[l] & kmask_iq2xs[j] ? -1.f : 1.f;

        buf[i] = convert_float<T>(dl * grid[j] * sign);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq3_xxs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq3_xxs * __restrict__ block = reinterpret_cast<const block_iq3_xxs *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib32 = idx/(8*4);
        int l = (idx/8)%4;
        int j = idx%8;
        
        const uint8_t * qs = block->qs + ib32*8;
        const uint8_t * scales_and_signs = block->qs + QK_K/4;

        uint32_t aux32;
        memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
        float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        
        uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
        const uint8_t * grid = (const uint8_t *)(iq3xxs_grid + qs[2*l+j/4]);

        float sign = signs & kmask_iq2xs[j] ? -1.f : 1.f;

        buf[i] = convert_float<T>(db * grid[j%4] * sign);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq3_s(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq3_s * __restrict__ block = reinterpret_cast<const block_iq3_s *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib32 = (idx/(8*8));
        int l = (idx/8)%8;
        int j = idx%8;
        
        const uint8_t * qs = block->qs + ib32*16 + (l/4)*8;
        const uint8_t * qh = block->qh + ib32*2;
        const uint8_t * signs = block->signs + ib32*8 + (l/4)*4;

        const uint8_t * grid = (const uint8_t *)(iq3s_grid + (qs[2*(l%4)+(j/4)] | ((qh[l/4] << (8-(j/4)-2*(l%4))) & 256)));
        
        float db = d * (1 + 2*((block->scales[ib32] >> 4*(l/4)) & 0xf));
        float sign = (signs[l%4] & kmask_iq2xs[j]) ? -1.f : 1.f;

        buf[i] = convert_float<T>(db * grid[j%4] * sign);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq1_s(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq1_s * __restrict__ block = reinterpret_cast<const block_iq1_s *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib = idx/(8*4);
        int l = (idx/8)%4;
        int j = idx%8;
        
        const uint16_t * qh = block->qh;
        const uint8_t * qs = block->qs + ib*4;
        
        float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
        float delta = (qh[ib] & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        
        const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> (3*l)) & 7) << 8)));

        buf[i] = convert_float<T>(dl * (grid[j] + delta));
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq1_m(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq1_m * __restrict__ block = reinterpret_cast<const block_iq1_m *>(w);
    const uint16_t * sc = (const uint16_t *)block->scales;
    // unpacking scale
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    float d = __half2float(scale.f16);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        
        int ib = idx/(8*4);
        int l = (idx/8)%4;
        int j = idx%8;

        const uint8_t * qs = block->qs + ib*4;
        const uint8_t * qh = block->qh + ib*2;

        float delta[4];
        uint16_t idx_grid[4];

        delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;

        idx_grid[0] = qs[0] | ((qh[0] << 8) & 0x700);
        idx_grid[1] = qs[1] | ((qh[0] << 4) & 0x700);
        idx_grid[2] = qs[2] | ((qh[1] << 8) & 0x700);
        idx_grid[3] = qs[3] | ((qh[1] << 4) & 0x700);
        
        float dl = d * (2*((sc[ib/2] >> (6*(ib%2)+3*(l/2))) & 0x7) + 1);

        const int8_t * grid = (const int8_t *)(iq1s_grid + idx_grid[l]);

        buf[i] = convert_float<T>(dl * (grid[j] + delta[l]));
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq4_nl(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK4_NL) return;

    T buf[ELEMS_PER_THR];

    const block_iq4_nl * __restrict__ block = reinterpret_cast<const block_iq4_nl *>(w);
    float d = __half2float(block->d);
    const uint8_t * qs = block->qs;

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        buf[i] = convert_float<T>(d * kvalues_iq4_nl[(qs[idx%(QK4_NL/2)] >> 4*(idx/(QK4_NL/2))) & 0xF]);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_iq4_xs(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_iq4_xs * __restrict__ block = reinterpret_cast<const block_iq4_xs *>(w);
    float d = __half2float(block->d);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;

        int ib = idx/32;
        int j = idx%32;
        
        int ls = ((block->scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((block->scales_h >> 2*ib) & 3) << 4);
        float dl = d*(ls-32);
        
        const uint8_t * qs = block->qs + ib*16;
        
        int8_t x = kvalues_iq4_nl[(qs[j%16] >> 4*(j/16)) & 0xF];

        buf[i] = convert_float<T>(dl*x);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_q8_K(
    const uint8_t * __restrict__ w, 
    T * __restrict__ y, 
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const block_q8_K * __restrict__ block = reinterpret_cast<const block_q8_K *>(w);
    
    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        buf[i] = convert_float<T>(block->d * block->qs[idx]);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}

template <typename T>
__device__ __forceinline__ void dequant_block_bf16(
    const uint8_t * __restrict__ w,
    T * __restrict__ y,
    unsigned int tid
) {
    constexpr int ELEMS_PER_THR = sizeof(int4)/sizeof(T);
    int base_idx = tid*ELEMS_PER_THR;
    if (base_idx >= QK_K) return;

    T buf[ELEMS_PER_THR];

    const uint16_t* block = reinterpret_cast<const uint16_t*>(w);

    #pragma unroll
    for (int i=0; i<ELEMS_PER_THR; ++i) {
        int idx = base_idx+i;
        buf[i] = convert_bf16<T>(block[idx]);
    }

    *reinterpret_cast<int4*>(&y[base_idx]) = *reinterpret_cast<int4*>(buf);
}