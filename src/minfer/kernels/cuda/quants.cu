// TODO: complete me!

// Adapted from: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
#include <torch/extension.h>
#include "quants.h"

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

// exposed for testing in Python host-side code
template <typename T>
__global__ void _dequant_row(
    GGMLQuantizationType qtype,
    const uint8_t* __restrict__ x,
    T* __restrict__ y,
    int64_t b,
    int64_t k
) {
    int row_idx = blockIdx.x;
    const uint8_t* x_row = x + row_idx * b;
    T* y_row = y + row_idx * k;
    __dequant_row<T>(qtype, x_row, y_row, k);
}

// for device-side calls
template <typename T>
__device__ void __dequant_row(
    GGMLQuantizationType qtype,
    const uint8_t* __restrict__ x,
    T* __restrict__ y,
    int64_t k
) {
    switch (qtype) {
        case GGMLQuantizationType::Q4_0: __dequant_row_q4_0<T>((const block_q4_0*)x, y, k); break;
        case GGMLQuantizationType::Q4_1: __dequant_row_q4_1<T>((const block_q4_1*)x, y, k); break;
        case GGMLQuantizationType::Q5_0: __dequant_row_q5_0<T>((const block_q5_0*)x, y, k); break;
        case GGMLQuantizationType::Q5_1: __dequant_row_q5_1<T>((const block_q5_1*)x, y, k); break;
        case GGMLQuantizationType::Q8_0: __dequant_row_q8_0<T>((const block_q8_0*)x, y, k); break;
        case GGMLQuantizationType::MXFP4: __dequant_row_mxfp4<T>((const block_mxfp4*)x, y, k); break;
        case GGMLQuantizationType::Q2_K: __dequant_row_q2_K<T>((const block_q2_K*)x, y, k); break;
        case GGMLQuantizationType::Q3_K: __dequant_row_q3_K<T>((const block_q3_K*)x, y, k); break;
        case GGMLQuantizationType::Q4_K: __dequant_row_q4_K<T>((const block_q4_K*)x, y, k); break;
        case GGMLQuantizationType::Q5_K: __dequant_row_q5_K<T>((const block_q5_K*)x, y, k); break;
        case GGMLQuantizationType::Q6_K: __dequant_row_q6_K<T>((const block_q6_K*)x, y, k); break;
        case GGMLQuantizationType::TQ1_0: __dequant_row_tq1_0<T>((const block_tq1_0*)x, y, k); break;
        case GGMLQuantizationType::TQ2_0: __dequant_row_tq2_0<T>((const block_tq2_0*)x, y, k); break;
        case GGMLQuantizationType::IQ2_XXS: __dequant_row_iq2_xxs<T>((const block_iq2_xxs*)x, y, k); break;
        case GGMLQuantizationType::IQ2_XS: __dequant_row_iq2_xs<T>((const block_iq2_xs*)x, y, k); break;
        case GGMLQuantizationType::IQ2_S: __dequant_row_iq2_s<T>((const block_iq2_s*)x, y, k); break;
        case GGMLQuantizationType::IQ3_XXS: __dequant_row_iq3_xxs<T>((const block_iq3_xxs*)x, y, k); break;
        case GGMLQuantizationType::IQ3_S: __dequant_row_iq3_s<T>((const block_iq3_s*)x, y, k); break;
        case GGMLQuantizationType::IQ1_S: __dequant_row_iq1_s<T>((const block_iq1_s*)x, y, k); break;
        case GGMLQuantizationType::IQ1_M: __dequant_row_iq1_m<T>((const block_iq1_m*)x, y, k); break;
        case GGMLQuantizationType::IQ4_NL: __dequant_row_iq4_nl<T>((const block_iq4_nl*)x, y, k); break;
        case GGMLQuantizationType::IQ4_XS: __dequant_row_iq4_xs<T>((const block_iq4_xs*)x, y, k); break;
        case GGMLQuantizationType::Q8_K: __dequant_row_q8_K<T>((const block_q8_K*)x, y, k); break;
        default: TORCH_CHECK(false, "Unsupported dtype");
    }
}

template __device__ void __dequant_row<float>(GGMLQuantizationType, const uint8_t*, float*, int64_t);
template __device__ void __dequant_row<half>(GGMLQuantizationType, const uint8_t*, half*, int64_t);

// legacy / simple quants

template <typename T>
__device__ void __dequant_row_q4_0(const block_q4_0 * __restrict__ x, T * __restrict__ y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = (T)(x0*d);
            y[i*qk + j + qk/2] = (T)(x1*d);
        }
    }
}

template <typename T>
void __dequant_row_q4_1(const block_q4_1 * __restrict__ x, T * __restrict__ y, int64_t k) {
    static const int qk = QK4_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;
        const float m = (float)x[i].m;

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);

            y[i*qk + j + 0   ] = (T)(x0*d + m);
            y[i*qk + j + qk/2] = (T)(x1*d + m);
        }
    }
}

template <typename T>
void __dequant_row_q5_0(const block_q5_0 * __restrict__ x, T * __restrict__ y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = (T)(x0*d);
            y[i*qk + j + qk/2] = (T)(x1*d);
        }
    }
}

template <typename T>
void __dequant_row_q5_1(const block_q5_1 * __restrict__ x, T * __restrict__ y, int64_t k) {
    static const int qk = QK5_1;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;
        const float m = (float)x[i].m;

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;

            y[i*qk + j + 0   ] = (T)(x0*d + m);
            y[i*qk + j + qk/2] = (T)(x1*d + m);
        }
    }
}

template <typename T>
void __dequant_row_q8_0(const block_q8_0 * __restrict__ x, T * __restrict__ y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = (T)(x[i].qs[j]*d);
        }
    }
}

// "microscaling" quant

template <typename T>
void __dequant_row_mxfp4(const block_mxfp4 * __restrict__ x, T * __restrict__ y, int64_t k) {
    static const int qk = QK_MXFP4;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = E8M0_TO_FP32_HALF(x[i].e);

        for (int j = 0; j < qk/2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >>   4];

            y[i*qk + j + 0   ] = (T)(x0*d);
            y[i*qk + j + qk/2] = (T)(x1*d);
        }
    }
}

//
// 2-6 bit quantization in super-blocks
//

template <typename T>
void __dequant_row_q2_K(const block_q2_K * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;
        const float min = (float)x[i].dmin;

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = (T)(dl * ((int8_t)((q[l] >> shift) & 3)) - ml);

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = (T)(dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml);

                shift += 2;
            }
            q += 32;
        }
    }
}

template <typename T>
void __dequant_row_q3_K(const block_q3_K * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = (float)x[i].d;

        const uint8_t * __restrict__ q = x[i].qs;
        const uint8_t * __restrict__ hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = (T)(dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4)));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = (T)(dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4)));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}

template <typename T>
void __dequant_row_q4_K(const block_q4_K * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = (float)x[i].d;
        const float min = (float)x[i].dmin;

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = (T)(d1 * (q[l] & 0xF) - m1);
            for (int l = 0; l < 32; ++l) *y++ = (T)(d2 * (q[l]  >> 4) - m2);
            q += 32; is += 2;
        }
    }
}

template <typename T>
void __dequant_row_q5_K(const block_q5_K * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = (float)x[i].d;
        const float min = (float)x[i].dmin;

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = (T)(d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1);
            for (int l = 0; l < 32; ++l) *y++ = (T)(d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2);
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

template <typename T>
void __dequant_row_q6_K(const block_q6_K * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;

        const uint8_t * __restrict__ ql = x[i].ql;
        const uint8_t * __restrict__ qh = x[i].qh;
        const int8_t  * __restrict__ sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = (T)(d * sc[is + 0] * q1);
                y[l + 32] = (T)(d * sc[is + 2] * q2);
                y[l + 64] = (T)(d * sc[is + 4] * q3);
                y[l + 96] = (T)(d * sc[is + 6] * q4);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

template <typename T>
void __dequant_row_tq1_0(const block_tq1_0 * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    for (int64_t i = 0; i < nb; ++i) {

        const float d = (float)x[i].d;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (T)((float) (xi - 1) * d);
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t n = 0; n < 5; ++n) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[n];
                    int16_t xi = ((uint16_t) q * 3) >> 8;
                    *y++ = (T)((float) (xi - 1) * d);
                }
            }
        }

        for (size_t n = 0; n < 4; ++n) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[n];
                int16_t xi = ((uint16_t) q * 3) >> 8;
                *y++ = (T)((float) (xi - 1) * d);
            }
        }
    }
}

template <typename T>
void __dequant_row_tq2_0(const block_tq2_0 * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {

        const float d = (float)x[i].d;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
                    *y++ = (T)((float) (q - 1) * d);
                }
            }
        }
    }
}

// ====================== "True" 2-bit (de)-quantization

template <typename T>
void __dequant_row_iq2_xxs(const block_iq2_xxs * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = (T)(db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.3125 bpw (de)-quantization

template <typename T>
void __dequant_row_iq2_xs(const block_iq2_xs * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = (T)(db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
                }
                y += 8;
            }
        }
    }
}

// ====================== 2.5625 bpw (de)-quantization

template <typename T>
void __dequant_row_iq2_s(const block_iq2_s * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = (T)(dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f));
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

// ====================== 3.0625 bpw (de)-quantization

template <typename T>
void __dequant_row_iq3_xxs(const block_iq3_xxs * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32;

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * scales_and_signs = qs + QK_K/4;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = (T)(db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
                    y[j+4] = (T)(db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

// ====================== 3.3125 bpw (de)-quantization

template <typename T>
void __dequant_row_iq3_s(const block_iq3_s * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = (T)(db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f));
                    y[j+4] = (T)(db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f));
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = (T)(db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f));
                    y[j+4] = (T)(db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f));
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

// ====================== 1.5625 bpw (de)-quantization

template <typename T>
void __dequant_row_iq1_s(const block_iq1_s * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = (float)x[i].d;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
            const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = (T)(dl * (grid[j] + delta));
                }
                y += 8;
            }
            qs += 4;
        }
    }
}

template <typename T>
void __dequant_row_iq1_m(const block_iq1_m * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float delta[4];
    uint16_t idx[4];

    iq1m_scale_t scale;

    for (int i = 0; i < nb; i++) {

        const uint16_t * sc = (const uint16_t *)x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const float d = (float)scale.f16;

        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
            const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);

            idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
            idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
            idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
            idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
            delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
            delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 2; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = (T)(dl1 * (grid[j] + delta[l]));
                }
                y += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + idx[l]);
                for (int j = 0; j < 8; ++j) {
                    y[j] = (T)(dl2 * (grid[j] + delta[l]));
                }
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

template <typename T>
void __dequant_row_iq4_nl(const block_iq4_nl * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = (float)x[i].d;
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = (T)(d * kvalues_iq4nl[qs[j] & 0xf]);
            y[j+QK4_NL/2] = (T)(d * kvalues_iq4nl[qs[j] >>  4]);
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

template <typename T>
void __dequant_row_iq4_xs(const block_iq4_xs * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = (float)x[i].d;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = (T)(dl * kvalues_iq4nl[qs[j] & 0xf]);
                y[j+16] = (T)(dl * kvalues_iq4nl[qs[j] >>  4]);
            }
            y  += 32;
            qs += 16;
        }
    }
}

//===================================== Q8_K ==============================================

template <typename T>
void __dequant_row_q8_K(const block_q8_K * __restrict__ x, T * __restrict__ y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = (T)(x[i].d * x[i].qs[j]);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("_dequant_row", &_dequant_row<float>);
    m.def("_dequant_row", &_dequant_row<half>);
}