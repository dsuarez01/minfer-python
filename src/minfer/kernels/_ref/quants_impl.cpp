// TODO: complete me!

// Adapted from: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c (MIT License)

#include "quants_impl.hpp"

#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <cassert>
#include <cfloat>
#include <cstdlib> // for qsort
// #include <mutex> // for impl inits
#include <algorithm> // for min and max

constexpr float GROUP_MAX_EPS = 1e-15f;
constexpr float GROUP_MAX_EPS_IQ3_XXS = 1e-8f;
constexpr float GROUP_MAX_EPS_IQ2_S = 1e-8f;
constexpr float GROUP_MAX_EPS_IQ1_M = 1e-7f;
constexpr float GROUP_MAX_EPS_IQ1_S = 1e-12f;

// dequant and quant helpers

// dequant
static inline float E8M0_TO_FP32_HALF(uint8_t x) {
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

static inline void get_scale_min_k4(int j, const uint8_t * __restrict__ q, uint8_t * __restrict__ d, uint8_t * __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// quant

static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static inline int best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

float make_qkx2_quants(int n, int nmax, const float * __restrict__ x, const float * __restrict__ weights,
        uint8_t * __restrict__ L, float * __restrict__ the_min, uint8_t * __restrict__ Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

float make_q3_quants(int n, int nmax, const float * __restrict__ x, int8_t * __restrict__ L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = std::max(-nmax, std::min(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

float make_qx_quants(int n, int nmax, const float * __restrict__ x, int8_t * __restrict__ L, int rmse_type,
        const float * __restrict__ qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

float make_qp_quants(int n, int nmax, const float * __restrict__ x, uint8_t * __restrict__ L, const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) {
        max = std::max(max, x[i]);
    }
    if (max < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int(iscale * x[i]);
    }
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = std::min(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::min(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = std::min(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

// ====== for IQ1 / IQ2 Quantization ======

// static std::once_flag iq2_xxs_init_flag;
// static std::once_flag iq2_xs_init_flag;
// static std::once_flag iq1_s_init_flag;
// static std::once_flag iq1_m_init_flag;
// static std::once_flag iq2_s_init_flag;

typedef struct {
    uint64_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq2_entry_t;

static iq2_entry_t iq2_data[4] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int iq2_data_index(GGMLQuantizationType type) {
    assert(
        type == GGMLQuantizationType::IQ2_XXS || 
        type == GGMLQuantizationType::IQ2_XS || 
        type == GGMLQuantizationType::IQ1_S || 
        type == GGMLQuantizationType::IQ1_M || 
        type == GGMLQuantizationType::IQ2_S
    );
    return type == GGMLQuantizationType::IQ2_XXS ? 0 :
           type == GGMLQuantizationType::IQ2_XS  ? 1 :
           type == GGMLQuantizationType::IQ1_S || type == GGMLQuantizationType::IQ1_M ? 2 : 3;
}

static inline int iq2_grid_size(GGMLQuantizationType type) {
    assert(
        type == GGMLQuantizationType::IQ2_XXS || 
        type == GGMLQuantizationType::IQ2_XS || 
        type == GGMLQuantizationType::IQ1_S || 
        type == GGMLQuantizationType::IQ1_M || 
        type == GGMLQuantizationType::IQ2_S
    );
    return type == GGMLQuantizationType::IQ2_XXS ? 256 :
           type == GGMLQuantizationType::IQ2_XS  ? 512 :
           type == GGMLQuantizationType::IQ1_S || type == GGMLQuantizationType::IQ1_M ? NGRID_IQ1S : 1024;
}

static int iq2_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq2xs_init_impl(GGMLQuantizationType type) {
    const int gindex = iq2_data_index(type);
    const int grid_size = iq2_grid_size(type);
    // if (iq2_data[gindex].grid) { // not necessary
    //     return;
    // }

    // NOTE: moved initialization of grid arrays to header

    const int kmap_size = 43692;
    //const int nwant = type == GGMLQuantizationType::IQ1_S ? 3 : 2;
    const int nwant = type == GGMLQuantizationType::IQ1_S || type == GGMLQuantizationType::IQ1_M ? 3 : type == GGMLQuantizationType::IQ2_S ? 1 : 2;
    const uint16_t * kgrid = type == GGMLQuantizationType::IQ2_XXS ? kgrid_2bit_256 :
                             type == GGMLQuantizationType::IQ2_XS  ? kgrid_2bit_512 :
                             type == GGMLQuantizationType::IQ1_S || type == GGMLQuantizationType::IQ1_M ? kgrid_1bit_2048 : kgrid_2bit_1024;
    uint64_t * kgrid_q2xs;
    int      * kmap_q2xs;
    uint16_t * kneighbors_q2xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    uint64_t * the_grid = (uint64_t *)malloc(grid_size*sizeof(uint64_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 8; ++i) {
            int l = (kgrid[k] >> 2*i) & 0x3;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q2xs = the_grid;
    iq2_data[gindex].grid = the_grid;
    kmap_q2xs = (int *)malloc(kmap_size*sizeof(int));
    iq2_data[gindex].map = kmap_q2xs;
    for (int i = 0; i < kmap_size; ++i) kmap_q2xs[i] = -1;
    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    for (int i = 0; i < grid_size; ++i) {
        aux64 = kgrid_q2xs[i];
        uint16_t index = 0;
        for (int k=0; k<8; ++k) {
            uint16_t q = (aux8[k] - 1)/2;
            index |= (q << 2*k);
        }
        kmap_q2xs[index] = i;
    }
    int8_t pos[8];
    int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
    int num_neighbors = 0, num_not_in_map = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) continue;
        ++num_not_in_map;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2*k) & 0x3;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        int n = 0; int d2 = dist2[0];
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    //printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbors_q2xs = (uint16_t *)malloc((num_neighbors + num_not_in_map)*sizeof(uint16_t));
    iq2_data[gindex].neighbours = kneighbors_q2xs;
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q2xs[i] >= 0) continue;
        for (int k = 0; k < 8; ++k) {
            int l = (i >> 2*k) & 0x3;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq2_compare_func);
        kmap_q2xs[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbors_q2xs[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2*j];
                ++nhave;
            }
            kneighbors_q2xs[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;
    }
    free(dist2);
}

void iq2xs_free_impl(GGMLQuantizationType type) {
    assert(
        type == GGMLQuantizationType::IQ2_XXS || 
        type == GGMLQuantizationType::IQ2_XS || 
        type == GGMLQuantizationType::IQ1_S || 
        type == GGMLQuantizationType::IQ1_M || 
        type == GGMLQuantizationType::IQ2_S
    );
    const int gindex = iq2_data_index(type);
    if (iq2_data[gindex].grid) {
        free(iq2_data[gindex].grid);       iq2_data[gindex].grid = NULL;
        free(iq2_data[gindex].map);        iq2_data[gindex].map  = NULL;
        free(iq2_data[gindex].neighbours); iq2_data[gindex].neighbours = NULL;
    }
}

static int iq2_find_best_neighbour(const uint16_t * __restrict__ neighbours, const uint64_t * __restrict__ grid,
        const float * __restrict__ xval, const float * __restrict__ weight, float scale, int8_t * __restrict__ L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

// ====== for IQ3 Quantization ======

// static std::once_flag iq3_xxs_init_flag;
// static std::once_flag iq3_s_init_flag;

typedef struct {
    uint32_t * grid;
    int      * map;
    uint16_t * neighbours;
} iq3_entry_t;

static iq3_entry_t iq3_data[2] = {
    {NULL, NULL, NULL},
    {NULL, NULL, NULL},
};

static inline int iq3_data_index(GGMLQuantizationType type) {
    assert(type == GGMLQuantizationType::IQ3_XXS || type == GGMLQuantizationType::IQ3_S);
    return type == GGMLQuantizationType::IQ3_XXS ? 0 : 1;
}

static int iq3_compare_func(const void * left, const void * right) {
    const int * l = (const int *)left;
    const int * r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

void iq3xs_init_impl(GGMLQuantizationType type) {
    const int gindex = iq3_data_index(type); // this asserts down to 2 types
    const int grid_size = type == GGMLQuantizationType::IQ3_XXS ? 256 : 512;
    // if (iq3_data[gindex].grid) {
    //     return;
    // }

    // NOTE: moved initialization of grid vals to header

    const int kmap_size = 4096;
    const int nwant = grid_size == 256 ? 2 : 3;
    const uint16_t * kgrid = grid_size == 256 ? kgrid_256 : kgrid_512;
    uint32_t * kgrid_q3xs;
    int      * kmap_q3xs;
    uint16_t * kneighbors_q3xs;

    //printf("================================================================= %s(grid_size = %d)\n", __func__, grid_size);
    uint32_t * the_grid = (uint32_t *)malloc(grid_size*sizeof(uint32_t));
    for (int k = 0; k < grid_size; ++k) {
        int8_t * pos = (int8_t *)(the_grid + k);
        for (int i = 0; i < 4; ++i) {
            int l = (kgrid[k] >> 3*i) & 0x7;
            pos[i] = 2*l + 1;
        }
    }
    kgrid_q3xs = the_grid;
    iq3_data[gindex].grid = the_grid;
    kmap_q3xs = (int *)malloc(kmap_size*sizeof(int));
    iq3_data[gindex].map = kmap_q3xs;
    for (int i = 0; i < kmap_size; ++i) kmap_q3xs[i] = -1;
    uint32_t aux32;
    uint8_t * aux8 = (uint8_t *)&aux32;
    for (int i = 0; i < grid_size; ++i) {
        aux32 = kgrid_q3xs[i];
        uint16_t index = 0;
        for (int k=0; k<4; ++k) {
            uint16_t q = (aux8[k] - 1)/2;
            index |= (q << 3*k);
        }
        kmap_q3xs[index] = i;
    }
    int8_t pos[4];
    int * dist2 = (int *)malloc(2*grid_size*sizeof(int));
    int num_neighbors = 0, num_not_in_map = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q3xs[i] >= 0) continue;
        ++num_not_in_map;
        for (int k = 0; k < 4; ++k) {
            int l = (i >> 3*k) & 0x7;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q3xs + j);
            int d2 = 0;
            for (int k = 0; k < 4; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq3_compare_func);
        int n = 0; int d2 = dist2[0];
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2*j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    //printf("%s: %d neighbours in total\n", __func__, num_neighbors);
    kneighbors_q3xs = (uint16_t *)malloc((num_neighbors + num_not_in_map)*sizeof(uint16_t));
    iq3_data[gindex].neighbours = kneighbors_q3xs;
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (kmap_q3xs[i] >= 0) continue;
        for (int k = 0; k < 4; ++k) {
            int l = (i >> 3*k) & 0x7;
            pos[k] = 2*l + 1;
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t * pg = (const int8_t *)(kgrid_q3xs + j);
            int d2 = 0;
            for (int k = 0; k < 4; ++k) d2 += (pg[k] - pos[k])*(pg[k] - pos[k]);
            dist2[2*j+0] = d2;
            dist2[2*j+1] = j;
        }
        qsort(dist2, grid_size, 2*sizeof(int), iq3_compare_func);
        kmap_q3xs[i] = -(counter + 1);
        int d2 = dist2[0];
        uint16_t * start = &kneighbors_q3xs[counter++];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2*j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2*j];
                ++nhave;
            }
            kneighbors_q3xs[counter++] = dist2[2*j+1];
            ++n;
        }
        *start = n;
    }
    free(dist2);
}

void iq3xs_free_impl(GGMLQuantizationType type) {
    const int gindex = iq3_data_index(type);
    if (iq3_data[gindex].grid) {
        free(iq3_data[gindex].grid);       iq3_data[gindex].grid = NULL;
        free(iq3_data[gindex].map);        iq3_data[gindex].map  = NULL;
        free(iq3_data[gindex].neighbours); iq3_data[gindex].neighbours = NULL;
    }
}

static int iq3_find_best_neighbour(const uint16_t * __restrict__ neighbours, const uint32_t * __restrict__ grid,
    const float * __restrict__ xval, const float * __restrict__ weight, float scale, int8_t * __restrict__ L) {
    
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX; // NOTE: check this
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

// ====== IQ1 helpers ======

// NOTE: currently unused
static int iq1_find_best_neighbour(const uint16_t * __restrict__ neighbours, const uint64_t * __restrict__ grid,
        const float * __restrict__ xval, const float * __restrict__ weight, float * scale, int8_t * __restrict__ L, int ngrid) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_score = -FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = (pg[i] - 3)/2;
            float w = weight[i];
            sumqx += w*q*xval[i];
            sumq2 += w*q*q;
        }
        if (sumqx > 0 && sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
            *scale = sumqx/sumq2; best_score = *scale * sumqx;
            grid_index = neighbours[j];
        }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t * grid_i = (const int8_t *)(grid + i);
            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = weight[j];
                float q = (grid_i[j] - 3)/2;
                sumqx += w*q*xval[j];
                sumq2 += w*q*q;
            }
            if (sumqx > 0 && sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                *scale = sumqx/sumq2; best_score = *scale*sumqx;
                grid_index = i;
            }
        }
    }
    if (grid_index < 0) {
        printf("Oops, did not find grid point\n");
        printf("Have %d neighbours\n", num_neighbors);
        for (int j = 1; j <= num_neighbors; ++j) {
            const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 8; ++i) {
                float q = (pg[i] - 3)/2;
                float w = weight[i];
                sumqx += w*q*xval[i];
                sumq2 += w*q*q;
            }
            printf("    neighbour %d: sumqx = %g sumq2 = %g\n", j, (double)sumqx, (double)sumq2);
        }
    }
    assert(grid_index >= 0);
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    *scale *= 1.05f;  // This is a fudge factor. Don't ask me why it improves the result.
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static int iq1_find_best_neighbour2(const uint16_t * __restrict__ neighbours, const uint64_t * __restrict__ grid,
        const float * __restrict__ xval, const float * __restrict__ weight, float scale, const float * __restrict__ xg, int8_t * __restrict__ L, int ngrid) {
    
    
    int num_neighbors = neighbours[0];

    assert(num_neighbors > 0);
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1)/2];
            float w = weight[i];
            float diff = scale*q - xval[i];
            d2 += w*diff*diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = neighbours[j];
        }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t * grid_i = (const int8_t *)(grid + i);
            float d2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = weight[j];
                float q = xg[(grid_i[j] - 1)/2];
                float diff = scale*q - xval[i];
                d2 += w*diff*diff;
            }
            if (d2 < best_score) {
                best_score = d2;
                grid_index = i;
            }
        }
    }
    if (grid_index < 0) {
        printf("Oops, did not find grid point\n");
        printf("Have %d neighbours\n", num_neighbors);
        for (int j = 1; j <= num_neighbors; ++j) {
            const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 8; ++i) {
                float q = xg[(pg[i] - 1)/2];
                float w = weight[i];
                sumqx += w*q*xval[i];
                sumq2 += w*q*q;
            }
            printf("    neighbour %d: sumqx = %g sumq2 = %g\n", j, (double)sumqx, (double)sumq2);
        }
    }
    assert(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

static int iq1_sort_helper(const void * left, const void * right) {
    const float * l = (const float *) left;
    const float * r = (const float *) right;
    return *l < *r ? -1 : *l > *r ? 1 : 0;
}

// actual dequant and quant fcns (I use the reference fcns for quant since its performance is unimportant in our case):
// legacy / simple quants

template <typename T>
void dequant_row_q4_0(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q4_0 * __restrict__ x = reinterpret_cast<const block_q4_0 *>(xr);
    
    constexpr int qk = QK4_0;
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

void quant_row_q4_0(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q4_0 * __restrict__ y = reinterpret_cast<block_q4_0 *>(yr);
    
    constexpr int qk = QK4_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = std::min((int8_t) 15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = std::min((int8_t) 15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

template <typename T>
void dequant_row_q4_1(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q4_1 * __restrict__ x = reinterpret_cast<const block_q4_1 *>(xr);
    
    constexpr int qk = QK4_1;
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

void quant_row_q4_1(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q4_1 * __restrict__ y = reinterpret_cast<block_q4_1 *>(yr);
    
    const int qk = QK4_1;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;
        y[i].m = (half_t) min;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = std::min((int8_t) 15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = std::min((int8_t) 15, (int8_t)(x1 + 0.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

template <typename T>
void dequant_row_q5_0(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q5_0 * __restrict__ x = reinterpret_cast<const block_q5_0 *>(xr);
    
    constexpr int qk = QK5_0;
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

void quant_row_q5_0(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q5_0 * __restrict__ y = reinterpret_cast<block_q5_0 *>(yr);
    
    constexpr int qk = QK5_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = std::min((int8_t) 31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = std::min((int8_t) 31, (int8_t)(x1 + 16.5f));

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(qh));
    }
}

template <typename T>
void dequant_row_q5_1(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q5_1 * __restrict__ x = reinterpret_cast<const block_q5_1 *>(xr);
    
    constexpr int qk = QK5_1;
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

void quant_row_q5_1(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q5_1 * __restrict__ y = reinterpret_cast<block_q5_1 *>(yr);
    
    const int qk = QK5_1;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (v < min) min = v;
            if (v > max) max = v;
        }

        const float d  = (max - min) / ((1 << 5) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;
        y[i].m = (half_t) min;

        uint32_t qh = 0;

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = (x[i*qk + 0    + j] - min)*id;
            const float x1 = (x[i*qk + qk/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

            // get the 5-th bit and store it in qh at the right position
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
        }

        memcpy(&y[i].qh, &qh, sizeof(y[i].qh));
    }
}

template <typename T>
void dequant_row_q8_0(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q8_0 * __restrict__ x = reinterpret_cast<const block_q8_0 *>(xr);
    
    constexpr int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = (float)x[i].d;

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = (T)(x[i].qs[j]*d);
        }
    }
}

void quant_row_q8_0(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q8_0 * __restrict__ y = reinterpret_cast<block_q8_0 *>(yr);
    
    constexpr int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;

        for (int j = 0; j < qk; ++j) {
            const float x0 = x[i*qk + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

// "microscaling" quant

template <typename T>
void dequant_row_mxfp4(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_mxfp4 * __restrict__ x = reinterpret_cast<const block_mxfp4 *>(xr);
    
    constexpr int qk = QK_MXFP4;
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

void quant_row_mxfp4(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_mxfp4 * __restrict__ y = reinterpret_cast<block_mxfp4 *>(yr);
    
    constexpr int qk = QK_MXFP4;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];

            if (amax < fabsf(v)) {
                amax = fabsf(v);
            }
        }

        const uint8_t e = amax > 0.0f ? (uint8_t) (floorf(log2f(amax)) - 2 + 127) : 0;

        const float d = E8M0_TO_FP32_HALF(e);

        y[i].e = e;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t x0 = best_index_mxfp4(x[i*qk + 0    + j], d);
            const uint8_t x1 = best_index_mxfp4(x[i*qk + qk/2 + j], d);

            y[i].qs[j]  = x0;
            y[i].qs[j] |= x1 << 4;
        }
    }
}

//
// 2-6 bit quantization in super-blocks
//

template <typename T>
void dequant_row_q2_K(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q2_K * __restrict__ x = reinterpret_cast<const block_q2_K *>(xr);
    
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

void quant_row_q2_K(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q2_K * __restrict__ y = reinterpret_cast<block_q2_K *>(yr);

    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = (half_t) (max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = (half_t) 0.f;
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = (half_t) (max_min/q4scale);
        } else {
            y[i].dmin = (half_t) 0.f;
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = (float) (y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = (float) (y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = std::max(0, std::min(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

template <typename T>
void dequant_row_q3_K(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q3_K * __restrict__ x = reinterpret_cast<const block_q3_K *>(xr);

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

void quant_row_q3_K(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q3_K * __restrict__ y = reinterpret_cast<block_q3_K *>(yr);

    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = std::max((int8_t)-32, std::min((int8_t)31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = (half_t) (1/iscale);
        } else {
            y[i].d = (half_t) 0.f;
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = (float) (y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = std::max(-4, std::min(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

template <typename T>
void dequant_row_q4_K(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q4_K * __restrict__ x = reinterpret_cast<const block_q4_K *>(xr);

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

void quant_row_q4_K(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q4_K * __restrict__ y = reinterpret_cast<block_q4_K *>(yr);

    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = std::min((uint8_t)63, ls);
            lm = std::min((uint8_t)63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = (half_t) (max_scale/63.f);
        y[i].dmin = (half_t) (max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = (float) (y[i].d) * sc;
            if (!d) continue;
            const float dm = (float) (y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = std::max(0, std::min(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

template <typename T>
void dequant_row_q5_K(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q5_K * __restrict__ x = reinterpret_cast<const block_q5_K *>(xr);
    
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

void quant_row_q5_K(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q5_K * __restrict__ y = reinterpret_cast<block_q5_K *>(yr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint8_t L[QK_K];
    float mins[QK_K/32];
    float scales[QK_K/32];
    float weights[32];
    uint8_t Laux[32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 31, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 31, x + 32*j, weights, L + 32*j, &mins[j], Laux, -0.5f, 0.1f, 15, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = std::min((uint8_t) 63, ls);
            lm = std::min((uint8_t) 63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = (half_t) (max_scale/63.f);
        y[i].dmin = (half_t) (max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = (float) (y[i].d) * sc;
            if (!d) continue;
            const float dm = (float) (y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = std::max(0, std::min(31, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * __restrict__ qh = y[i].qh;
        uint8_t * __restrict__ ql = y[i].qs;
        memset(qh, 0, QK_K/8);

        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < QK_K; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16; qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16; qh[j] |= m2;
                }
                ql[j] = l1 | (l2 << 4);
            }
            m1 <<= 2; m2 <<= 2;
            ql += 32;
        }

        x += QK_K;
    }
}

template <typename T>
void dequant_row_q6_K(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q6_K * __restrict__ x = reinterpret_cast<const block_q6_K *>(xr);

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

void quant_row_q6_K(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q6_K * __restrict__ y = reinterpret_cast<block_q6_K *>(yr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = (half_t) 0.f;
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = (half_t) (1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = std::min((int8_t) 127, (int8_t) nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = (float) (y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = std::max(-32, std::min(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * __restrict__ ql = y[i].ql;
        uint8_t * __restrict__ qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

template <typename T>
void dequant_row_tq1_0(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_tq1_0 * __restrict__ x = reinterpret_cast<const block_tq1_0 *>(xr);

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

void quant_row_tq1_0(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_tq1_0 * __restrict__ y = reinterpret_cast<block_tq1_0 *>(yr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;

        // 5 elements per byte, along 32 bytes
        for (size_t j = 0; j < sizeof(y->qs) - sizeof(y->qs) % 32; j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*32] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*32;
        }
        // along 16 bytes
        for (size_t j = sizeof(y->qs) - sizeof(y->qs) % 32; j < sizeof(y->qs); j += 16) {
            for (size_t m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 5; ++n) {
                    int xi = lroundf(x[m + n*16] * id) + 1; // -1, 0, 1 -> 0, 1, 2
                    q *= 3;
                    q += xi;
                }
                // ceiling division (243 == pow(3, 5))
                q = ((uint16_t)q * 256 + (243 - 1)) / 243;
                y[i].qs[j + m] = q;
            }
            x += 5*16;
        }
        // 4 elements per byte
        for (size_t j = 0; j < sizeof(y->qh); ++j) {
            uint8_t q = 0;
            for (size_t m = 0; m < 4; ++m) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[j + m*sizeof(y->qh)] * id) + 1;
                q *= 3;
                q += xi;
            }
            // shift the first value to the most significant trit
            q *= 3;
            // ceiling division (243 == pow(3, 5))
            q = ((uint16_t)q * 256 + (243 - 1)) / 243;
            y[i].qh[j] = q;
        }
        x += 4*sizeof(y->qh);
    }
}

template <typename T>
void dequant_row_tq2_0(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_tq2_0 * __restrict__ x = reinterpret_cast<const block_tq2_0 *>(xr);

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

void quant_row_tq2_0(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_tq2_0 * __restrict__ y = reinterpret_cast<block_tq2_0 *>(yr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK_K; j++) {
            const float v = x[j];
            amax = std::max(amax, fabsf(v));
        }

        const float d = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = (half_t) d;

        for (size_t j = 0; j < sizeof(y->qs); j += 32) {
            for (size_t m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (size_t n = 0; n < 4; ++n) {
                    // -1, 0, 1 -> 0, 1, 2
                    int xi = lroundf(x[m + n*32] * id) + 1;
                    q += (xi & 3) << (2*n);
                }
                y[i].qs[j + m] = q;
            }
            x += 4*32;
        }
    }
}

// ====================== "True" 2-bit (de)-quantization

template <typename T>
void dequant_row_iq2_xxs(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq2_xxs * __restrict__ x = reinterpret_cast<const block_iq2_xxs *>(xr);

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

void quant_row_iq2_xxs(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq2_xxs_init_flag, []() {
    //     iq2xs_init_impl(GGMLQuantizationType::IQ2_XXS);
    // });

    block_iq2_xxs * __restrict__ y = reinterpret_cast<block_iq2_xxs *>(yr);
    const int gindex = iq2_data_index(GGMLQuantizationType::IQ2_XXS);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    // assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q2xs      && "iq2/3xs_init_impl failed");
    assert(kmap_q2xs       && "iq2/3xs_init_impl failed");
    assert(kneighbors_q2xs && "iq2/3xs_init_impl failed");
    assert(n%QK_K == 0);

    const int kMaxQ = 3;
    const int64_t nbl = n/QK_K;

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = half_t(0.f);
        memset(q2, 0, QK_K/4);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 32; ++i) weight[i] = xb[i] * xb[i]; // NOTE: added this in
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
            float eff_max = scale*kMaxQ;
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/eff_max;
                float this_scale = 1/id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = std::max(0, std::min(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = std::max(0, std::min(kMaxQ-1, l));
                        u |= (l << 2*i);
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
                    for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    assert(false && "fatal error");
                }
                q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
                q2[2*ib+1] |= (block_signs[k] << 7*k);
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = half_t(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = std::max(0, std::min(15, l));
            q2[2*ib+1] |= ((uint32_t)l << 28);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);
    }
}

// ====================== 2.3125 bpw (de)-quantization

template <typename T>
void dequant_row_iq2_xs(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq2_xs * __restrict__ x = reinterpret_cast<const block_iq2_xs *>(xr);

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

void quant_row_iq2_xs(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq2_xs_init_flag, []() {
    //     iq2xs_init_impl(GGMLQuantizationType::IQ2_XS);
    // });

    block_iq2_xs * __restrict__ y = reinterpret_cast<block_iq2_xs *>(yr);
    const int gindex = iq2_data_index(GGMLQuantizationType::IQ2_XS);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    // assert(quant_weights   && "missing quantization weights");
    assert(kmap_q2xs       && "iq2/3xs_init_impl failed");
    assert(kgrid_q2xs      && "iq2/3xs_init_impl failed");
    assert(kneighbors_q2xs && "iq2/3xs_init_impl failed");
    assert(n%QK_K == 0);

    const int kMaxQ = 3;
    const int64_t nbl = n/QK_K;

    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = half_t(0.f);
        memset(q2, 0, QK_K/4);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 16*ib;
                for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 16; ++i) weight[i] = xb[i] * xb[i]; // NOTE: added this in
            }
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS) {
                scales[ib] = 0;
                memset(L, 0, 16);
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = std::max(0, std::min(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = std::max(0, std::min(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    assert(false && "fatal error");
                }
                q2[2*ib+k] = grid_index | (block_signs[k] << 9);
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            memset(y[ibl].qs, 0, QK_K/4);
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = half_t(d);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = std::max(0, std::min(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
        memcpy(y[ibl].qs, q2, QK_K/4);

    }
}

// ====================== 2.5625 bpw (de)-quantization

template <typename T>
void dequant_row_iq2_s(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq2_s * __restrict__ x = reinterpret_cast<const block_iq2_s *>(xr);
    
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

void quant_row_iq2_s(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq2_s_init_flag, []() {
    //     iq2xs_init_impl(GGMLQuantizationType::IQ2_S);
    // });

    block_iq2_s * __restrict__ y = reinterpret_cast<block_iq2_s *>(yr);
    const int gindex = iq2_data_index(GGMLQuantizationType::IQ2_S);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    // assert(quant_weights   && "missing quantization weights");
    assert(kmap_q2xs       && "iq2/3xs_init_impl failed");
    assert(kgrid_q2xs      && "iq2/3xs_init_impl failed");
    assert(kneighbors_q2xs && "iq2/3xs_init_impl failed");
    assert(n%QK_K == 0);

    const int kMaxQ = 3;
    const int64_t nbl = n/QK_K;

    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float  waux[16];
    bool   is_on_grid[2];
    bool   is_on_grid_aux[2];
    uint8_t block_signs[2];

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_s));
        y[ibl].d = half_t(0.f);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 16*ib;
                for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                // for (int i = 0; i < 16; ++i) weight[i] = 0.25f*sigma2 + xb[i]*xb[i];
                for (int i = 0; i < 16; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ2_S) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.1f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        Laux[8*k+i] = std::max(0, std::min(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                    int grid_index = kmap_q2xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                        l = std::max(0, std::min(kMaxQ-1, l));
                        u |= (l << 2*i);
                        L[8*k + i] = l;
                    }
                    int grid_index = kmap_q2xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                        grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 8; ++i) printf(" %d", L[8*k+i]);
                    printf("\n");
                    assert(false && "fatal error");
                }
                const int i8 = 2*ib + k;
                y[ibl].qs[i8] = grid_index & 255;
                y[ibl].qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
                y[ibl].qs[QK_K/8 + i8] = block_signs[k];
            }
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = half_t(d * 0.9875f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = std::max(0, std::min(15, l));
            if (ib%2 == 0) y[ibl].scales[ib/2] = l;
            else y[ibl].scales[ib/2] |= (l << 4);
        }
    }
}

// ====================== 3.0625 bpw (de)-quantization

template <typename T>
void dequant_row_iq3_xxs(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq3_xxs * __restrict__ x = reinterpret_cast<const block_iq3_xxs *>(xr);
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

void quant_row_iq3_xxs(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq3_xxs_init_flag, []() {
    //     iq3xs_init_impl(GGMLQuantizationType::IQ3_XXS);
    // });

    block_iq3_xxs * __restrict__ y = reinterpret_cast<block_iq3_xxs *>(yr);
    const int gindex = iq3_data_index(GGMLQuantizationType::IQ3_XXS);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q3xs      && "iq2/3xs_init_impl failed");
    assert(kmap_q3xs       && "iq2/3xs_init_impl failed");
    assert(kneighbors_q3xs && "iq2/3xs_init_impl failed");
    assert(n % QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;
    
    half_t * dh = &y->d;
    uint8_t* qs = y->qs;
    int block_size = sizeof(block_iq3_xxs);
    int quant_size = block_size - sizeof(half_t);
    // NOTE: why is this even in the original code?
    // if (grid_size == 256) {
    //     block_iq3_xxs * y = vy;
    //     dh = &y->d;
    //     qs = y->qs;
    //     block_size = sizeof(block_iq3_xxs);
    // } else {
    //     block_iq3_s * y = vy;
    //     dh = &y->d;
    //     qs = y->qs;
    //     block_size = sizeof(block_iq3_s);
    // }

    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float  waux[32];
    bool   is_on_grid[8];
    bool   is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint8_t q3[3*(QK_K/8)+QK_K/32];

    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4);
    uint8_t  * qh = q3 + 3*(QK_K/8); // NOTE: unused
    

    for (int ibl = 0; ibl < nbl; ++ibl) {

        dh[0] = half_t(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const float * xb = xbl + 32*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i);
                    }
                }
                if (nflip%2) {
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin];
                    s ^= (1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = std::max(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true;
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = std::max(0, std::min(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = std::max(0, std::min(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    assert(false && "fatal error");
                }
                q3[8*ib+k] = grid_index;

            }
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/sizeof(half_t);
            qs += block_size;
            continue;
        }

        float d = max_scale/31;
        dh[0] = half_t(d * 1.0125f);  // small improvement via this fudge factor
        float id = 1/d;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = std::max(0, std::min(15, l));
            scales_and_signs[ib] |= ((uint32_t)l << 28);
        }
        memcpy(qs, q3, quant_size);

        dh += block_size/sizeof(half_t);
        qs += block_size;

    }
}


// ====================== 3.3125 bpw (de)-quantization

template <typename T>
void dequant_row_iq3_s(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq3_s * __restrict__ x = reinterpret_cast<const block_iq3_s *>(xr);

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

constexpr int IQ3S_BLOCK_SIZE = 32;
void quant_row_iq3_s(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq3_s_init_flag, []() {
    //     iq3xs_init_impl(GGMLQuantizationType::IQ3_S);
    // });

    block_iq3_s * __restrict__ y = reinterpret_cast<block_iq3_s *>(yr);
    const int gindex = iq3_data_index(GGMLQuantizationType::IQ3_S);

    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    //assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q3xs      && "iq2/3xs_init_impl failed");
    assert(kmap_q3xs       && "iq2/3xs_init_impl failed");
    assert(kneighbors_q3xs && "iq2/3xs_init_impl failed");
    assert(n%QK_K == 0);

    const int kMaxQ = 8;
    const int64_t nbl = n/QK_K;
    const int block_size = IQ3S_BLOCK_SIZE;

    float scales[QK_K/IQ3S_BLOCK_SIZE];
    float weight[IQ3S_BLOCK_SIZE];
    float xval[IQ3S_BLOCK_SIZE];
    int8_t L[IQ3S_BLOCK_SIZE];
    int8_t Laux[IQ3S_BLOCK_SIZE];
    float waux[IQ3S_BLOCK_SIZE];
    bool is_on_grid[IQ3S_BLOCK_SIZE/4];
    bool is_on_grid_aux[IQ3S_BLOCK_SIZE/4];
    uint8_t block_signs[IQ3S_BLOCK_SIZE/8];

    const int bs4 = block_size/4;
    const int bs8 = block_size/8;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = half_t(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = std::max(max, xval[i]);
            if (!max) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = std::max(0, std::min(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = std::max(0, std::min(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    assert(false && "fatal error");
                }
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            assert(scale >= 0);
            scales[ib] = scale;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = half_t(d * 1.033f);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = std::max(0, std::min(15, l1));
            int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = std::max(0, std::min(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}

// ====================== 1.5625 bpw (de)-quantization

template <typename T>
void dequant_row_iq1_s(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq1_s * __restrict__ x = reinterpret_cast<const block_iq1_s *>(xr);
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

constexpr int IQ1S_BLOCK_SIZE = 32;
void quant_row_iq1_s(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq1_s_init_flag, []() {
    //     iq2xs_init_impl(GGMLQuantizationType::IQ1_S);
    // });

    block_iq1_s * __restrict__ y = reinterpret_cast<block_iq1_s *>(yr);
    const int gindex = iq2_data_index(GGMLQuantizationType::IQ1_S);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    // assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q2xs      && "iq2/3xs_init_impl failed");
    assert(kmap_q2xs       && "iq2/3xs_init_impl failed");
    assert(kneighbors_q2xs && "iq2/3xs_init_impl failed");
    assert(n%QK_K == 0);

    const int64_t nbl = n/QK_K;
    const int block_size = IQ1S_BLOCK_SIZE;

    float scales[QK_K/IQ1S_BLOCK_SIZE];
    float weight[IQ1S_BLOCK_SIZE];
    int8_t L[IQ1S_BLOCK_SIZE];
    float sumx[IQ1S_BLOCK_SIZE+1];
    float sumw[IQ1S_BLOCK_SIZE+1];
    float pairs[2*IQ1S_BLOCK_SIZE];
    uint16_t index[IQ1S_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1S_BLOCK_SIZE];

    const float x_p[3] = {-1 + IQ1S_DELTA,  IQ1S_DELTA, 1 + IQ1S_DELTA};
    const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};


    int * idx = (int *)(pairs + 1);

    for (int ibl = 0; ibl < nbl; ++ibl) {

        y[ibl].d = half_t(0.f);
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;

            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i] * xb[i];
            }
            
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = std::max(max, fabsf(xb[i]));
            if (max < GROUP_MAX_EPS_IQ1_S) {
                scales[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            qsort(pairs, block_size, 2*sizeof(float), iq1_sort_helper);
            {
                sumx[0] = sumw[0] = 0;
                for (int j = 0; j < block_size; ++j) {
                    int i = idx[2*j];
                    sumx[j+1] = sumx[j] + weight[i]*xb[i];
                    sumw[j+1] = sumw[j] + weight[i];
                }
            }
            float best_score = -FLT_MAX, scale = max;
            int besti1 = -1, besti2 = -1, best_shift = 0;
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    float sumqx = (sumx[i1] - sumx[0])*x_p[0] + (sumx[i2] - sumx[i1])*x_p[1] + (sumx[block_size] - sumx[i2])*x_p[2];
                    float sumq2 = (sumw[i1] - sumw[0])*x_p[0]*x_p[0] + (sumw[i2] - sumw[i1])*x_p[1]*x_p[1] + (sumw[block_size] - sumw[i2])*x_p[2]*x_p[2];
                    if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                        scale = sumqx/sumq2; best_score = scale*sumqx;
                        besti1 = i1; besti2 = i2; best_shift = 1;
                    }
                    sumqx = (sumx[i1] - sumx[0])*x_m[0] + (sumx[i2] - sumx[i1])*x_m[1] + (sumx[block_size] - sumx[i2])*x_m[2];
                    sumq2 = (sumw[i1] - sumw[0])*x_m[0]*x_m[0] + (sumw[i2] - sumw[i1])*x_m[1]*x_m[1] + (sumw[block_size] - sumw[i2])*x_m[2]*x_m[2];
                    if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                        scale = sumqx/sumq2; best_score = scale*sumqx;
                        besti1 = i1; besti2 = i2; best_shift = -1;
                    }
                }
            }
            assert(besti1 >= 0 && besti2 >= 0 && best_shift != 0);
            for (int j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale; best_shift = -best_shift;
            }
            bool all_on_grid = true;
            const float * xx = best_shift == 1 ? x_p : x_m;
            for (int k = 0; k < block_size/8; ++k) {
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    assert(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx = 0, sumq2 = 0;
                for (int k = 0; k < block_size/8; ++k) {
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8*k + j];
                        float q = xx[(pg[j] - 1)/2];
                        sumqx += w*q*xb[8*k+j];
                        sumq2 += w*q*q;
                    }
                }
                if (sumqx > 0 && sumq2 > 0) scale = sumqx/sumq2;
            }
            uint16_t h = 0;
            for (int k = 0; k < block_size/8; ++k) {
                y[ibl].qs[(block_size/8)*ib + k] = index[k] & 255;
                h |= (index[k] >> 8) << 3*k;
            }
            y[ibl].qh[ib] = h;
            assert(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_shift;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/15;
        y[ibl].d = half_t(d*1.125f); // 1.125f is another fudge factor. Don't ask me why it is needed.
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = std::max(0, std::min(7, l));
            if (shifts[ib] == -1) l |= 8;
            y[ibl].qh[ib] |= (l << 12);
        }
    }
}

template <typename T>
void dequant_row_iq1_m(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq1_m * __restrict__ x = reinterpret_cast<const block_iq1_m *>(xr);

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

constexpr int IQ1M_BLOCK_SIZE = 16;
void quant_row_iq1_m(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {

    // std::call_once(iq1_m_init_flag, []() {
    //     iq2xs_init_impl(GGMLQuantizationType::IQ1_M);
    // });

    block_iq1_m * __restrict__ y = reinterpret_cast<block_iq1_m *>(yr);
    const int gindex = iq2_data_index(GGMLQuantizationType::IQ1_M);

    const uint64_t * kgrid_q2xs      = iq2_data[gindex].grid;
    const int      * kmap_q2xs       = iq2_data[gindex].map;
    const uint16_t * kneighbors_q2xs = iq2_data[gindex].neighbours;

    //assert(quant_weights   && "missing quantization weights");
    assert(kgrid_q2xs      && "iq2/3xs_init_impl failed");
    assert(kmap_q2xs       && "iq2/3xs_init_impl failed");
    assert(kneighbors_q2xs && "iq2/3xs_init_impl failed");
    assert(n%QK_K == 0);

    const int64_t nbl = n/QK_K;
    const int block_size = IQ1M_BLOCK_SIZE;

    float  scales[QK_K/IQ1M_BLOCK_SIZE];
    float  weight[IQ1M_BLOCK_SIZE];
    int8_t L[IQ1M_BLOCK_SIZE];
    float  pairs[2*IQ1M_BLOCK_SIZE];
    uint16_t index[IQ1M_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1M_BLOCK_SIZE];

    const float x_p[3] = {-1 + IQ1M_DELTA,  IQ1M_DELTA, 1 + IQ1M_DELTA};
    const float x_m[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};

    int * idx = (int *)(pairs + 1);

    float sumqx[4], sumq2[4];

    iq1m_scale_t s;
    const float * xx;

    for (int ibl = 0; ibl < nbl; ++ibl) {
        memset(y[ibl].qs, 0, QK_K/8);
        memset(y[ibl].qh, 0, QK_K/16);
        memset(y[ibl].scales, 0, QK_K/32);

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i] * xb[i];
            }
            float max = fabsf(xb[0]);
            for (int i = 1; i < block_size; ++i) max = std::max(max, fabsf(xb[i]));
            if (max < GROUP_MAX_EPS_IQ1_M) {
                scales[ib] = 0;
                memset(L, 1, block_size);
                continue;
            }
            // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
            // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
            // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
            // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
            // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
            // for each possible and score for each split.
            for (int j = 0; j < block_size; ++j) {
                pairs[2*j] = xb[j];
                idx[2*j] = j;
            }
            qsort(pairs, block_size, 2*sizeof(float), iq1_sort_helper);
            float best_score = -FLT_MAX, scale = max;
            int besti1 = -1, besti2 = -1, best_k = -1;
            // 0: +, +
            // 1: +, -
            // 2: -, +
            // 3: -, -
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    memset(sumqx, 0, 4*sizeof(float));
                    memset(sumq2, 0, 4*sizeof(float));
                    for (int j = 0; j < i1; ++j) {
                        int i = idx[2*j];
                        if (i < block_size/2) {
                            sumqx[0] += weight[i]*x_p[0]*xb[i];
                            sumqx[1] += weight[i]*x_p[0]*xb[i];
                            sumqx[2] += weight[i]*x_m[0]*xb[i];
                            sumqx[3] += weight[i]*x_m[0]*xb[i];
                            sumq2[0] += weight[i]*x_p[0]*x_p[0];
                            sumq2[1] += weight[i]*x_p[0]*x_p[0];
                            sumq2[2] += weight[i]*x_m[0]*x_m[0];
                            sumq2[3] += weight[i]*x_m[0]*x_m[0];
                        } else {
                            sumqx[0] += weight[i]*x_p[0]*xb[i];
                            sumqx[2] += weight[i]*x_p[0]*xb[i];
                            sumqx[1] += weight[i]*x_m[0]*xb[i];
                            sumqx[3] += weight[i]*x_m[0]*xb[i];
                            sumq2[0] += weight[i]*x_p[0]*x_p[0];
                            sumq2[2] += weight[i]*x_p[0]*x_p[0];
                            sumq2[1] += weight[i]*x_m[0]*x_m[0];
                            sumq2[3] += weight[i]*x_m[0]*x_m[0];
                        }
                    }
                    for (int j = i1; j < i2; ++j) {
                        int i = idx[2*j];
                        if (i < block_size/2) {
                            sumqx[0] += weight[i]*x_p[1]*xb[i];
                            sumqx[1] += weight[i]*x_p[1]*xb[i];
                            sumqx[2] += weight[i]*x_m[1]*xb[i];
                            sumqx[3] += weight[i]*x_m[1]*xb[i];
                            sumq2[0] += weight[i]*x_p[1]*x_p[1];
                            sumq2[1] += weight[i]*x_p[1]*x_p[1];
                            sumq2[2] += weight[i]*x_m[1]*x_m[1];
                            sumq2[3] += weight[i]*x_m[1]*x_m[1];
                        } else {
                            sumqx[0] += weight[i]*x_p[1]*xb[i];
                            sumqx[2] += weight[i]*x_p[1]*xb[i];
                            sumqx[1] += weight[i]*x_m[1]*xb[i];
                            sumqx[3] += weight[i]*x_m[1]*xb[i];
                            sumq2[0] += weight[i]*x_p[1]*x_p[1];
                            sumq2[2] += weight[i]*x_p[1]*x_p[1];
                            sumq2[1] += weight[i]*x_m[1]*x_m[1];
                            sumq2[3] += weight[i]*x_m[1]*x_m[1];
                        }
                    }
                    for (int j = i2; j < block_size; ++j) {
                        int i = idx[2*j];
                        if (i < block_size/2) {
                            sumqx[0] += weight[i]*x_p[2]*xb[i];
                            sumqx[1] += weight[i]*x_p[2]*xb[i];
                            sumqx[2] += weight[i]*x_m[2]*xb[i];
                            sumqx[3] += weight[i]*x_m[2]*xb[i];
                            sumq2[0] += weight[i]*x_p[2]*x_p[2];
                            sumq2[1] += weight[i]*x_p[2]*x_p[2];
                            sumq2[2] += weight[i]*x_m[2]*x_m[2];
                            sumq2[3] += weight[i]*x_m[2]*x_m[2];
                        } else {
                            sumqx[0] += weight[i]*x_p[2]*xb[i];
                            sumqx[2] += weight[i]*x_p[2]*xb[i];
                            sumqx[1] += weight[i]*x_m[2]*xb[i];
                            sumqx[3] += weight[i]*x_m[2]*xb[i];
                            sumq2[0] += weight[i]*x_p[2]*x_p[2];
                            sumq2[2] += weight[i]*x_p[2]*x_p[2];
                            sumq2[1] += weight[i]*x_m[2]*x_m[2];
                            sumq2[3] += weight[i]*x_m[2]*x_m[2];
                        }
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (sumq2[k] > 0 && sumqx[k]*sumqx[k] > best_score*sumq2[k]) {
                            scale = sumqx[k]/sumq2[k]; best_score = scale*sumqx[k];
                            besti1 = i1; besti2 = i2; best_k = k;
                        }
                    }
                }
            }
            assert(besti1 >= 0 && besti2 >= 0 && best_k >= 0);
            for (int j =      0; j < besti1; ++j) L[idx[2*j]] = 0;
            for (int j = besti1; j < besti2; ++j) L[idx[2*j]] = 1;
            for (int j = besti2; j < block_size; ++j) L[idx[2*j]] = 2;
            if (scale < 0) {
                for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
                scale = -scale;
                best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
            }
            bool all_on_grid = true;
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = best_k < 2 ? x_p : x_m;
                else xx = best_k%2 == 0 ? x_p : x_m;
                uint16_t u = 0;
                for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    all_on_grid = false;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
                    assert(grid_index >= 0);
                }
                index[k] = grid_index;
            }
            if (!all_on_grid) {
                float sumqx_f = 0, sumq2_f = 0;
                for (int k = 0; k < block_size/8; ++k) {
                    if (k == 0) xx = best_k < 2 ? x_p : x_m;
                    else xx = best_k%2 == 0 ? x_p : x_m;
                    const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                    for (int j = 0; j < 8; ++j) {
                        float w = weight[8*k + j];
                        float q = xx[(pg[j] - 1)/2];
                        sumqx_f += w*q*xb[8*k+j];
                        sumq2_f += w*q*q;
                    }
                }
                if (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f/sumq2_f;
            }
            y[ibl].qs[2*ib + 0] = index[0] & 255;
            y[ibl].qs[2*ib + 1] = index[1] & 255;
            y[ibl].qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
            assert(scale >= 0);
            scales[ib] = scale;
            shifts[ib] = best_k;
            max_scale = std::max(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        uint16_t * sc = (uint16_t *)y[ibl].scales;
        float d = max_scale/15;
        float id = 1/d;
        float sumqx_f = 0, sumq2_f = 0;
        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib+0]-1));
            l = std::max(0, std::min(7, l));
            sc[ib/4] |= (l << 3*(ib%4));
            y[ibl].qh[ib] |= masks[shifts[ib]];
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i] * xb[i];
            }
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = shifts[ib] < 2 ? x_p : x_m;
                else xx = shifts[ib]%2 == 0 ? x_p : x_m;
                const int8_t * pg = (const int8_t *)(kgrid_q2xs + y[ibl].qs[2*ib+k] + ((y[ibl].qh[ib] << (8 - 4*k)) & 0x700));
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8*k + j];
                    float q = xx[(pg[j] - 1)/2]*(2*l+1);
                    sumqx_f += w*q*xb[8*k+j];
                    sumq2_f += w*q*q;
                }
            }
        }
        if (sumq2_f > 0) d = sumqx_f/sumq2_f;
        s.f16 = half_t(d*1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
        sc[0] |= ((s.u16 & 0x000f) << 12);
        sc[1] |= ((s.u16 & 0x00f0) <<  8);
        sc[2] |= ((s.u16 & 0x0f00) <<  4);
        sc[3] |= ((s.u16 & 0xf000) <<  0);
    }
}

template <typename T>
void dequant_row_iq4_nl(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq4_nl * __restrict__ x = reinterpret_cast<const block_iq4_nl *>(xr);
    assert(k % QK4_NL == 0);
    const int64_t nb = k / QK4_NL;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = (float)x[i].d;
        for (int j = 0; j < QK4_NL/2; ++j) {
            y[j+       0] = (T)(d * kvalues_iq4_nl[qs[j] & 0xf]);
            y[j+QK4_NL/2] = (T)(d * kvalues_iq4_nl[qs[j] >>  4]);
        }
        y  += QK4_NL;
        qs += QK4_NL/2;
    }
}

void quant_row_iq4_nl(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {
    block_iq4_nl * __restrict__ y = reinterpret_cast<block_iq4_nl *>(yr);

    assert(n % QK4_NL == 0);
    int64_t nblock = n / QK4_NL;

    const int super_block_size = QK4_NL;
    const int block_size = 32;
    const int8_t* values = kvalues_iq4_nl;
    const int ntry = 7;
    
    float scales[super_block_size/block_size];
    float weight[super_block_size];
    uint8_t L[super_block_size];
    uint16_t scales_h_unused;
    uint16_t* scales_h = &scales_h_unused;
    uint8_t* scales_l;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        half_t* dh = &y[ibl].d;
        uint8_t* q4 = y[ibl].qs;
        
        const float* x_block = x + ibl * QK4_NL;
        const float* qw_block = quant_weights ? quant_weights + ibl * QK4_NL : nullptr;
        
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += x_block[j]*x_block[j];
        sigma2 *= 2.f/super_block_size;

        memset(q4, 0, super_block_size/2);
        dh[0] = half_t(0.f);

        float max_scale = 0, amax_scale = 0;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = x_block + ib*block_size;
            uint8_t * Lb = L + ib*block_size;
            if (qw_block) {
                const float * qw = qw_block + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                Lb[j] = l;
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            d = sumqx/sumq2;
            float best = d*sumqx;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx = sumq2 = 0;
                for (int j = 0; j < block_size; ++j) {
                    float al = id*xb[j];
                    int l = best_index_int8(16, values, al);
                    float q = values[l];
                    float w = weight[j];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    d = sumqx/sumq2; best = d * sumqx;
                }
            }
            scales[ib] = d;
            float abs_d = fabsf(d);
            if (abs_d > amax_scale) {
                amax_scale = abs_d; max_scale = d;
            }
        }

        if (super_block_size/block_size > 1) {
            int nb = super_block_size/block_size;
            memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
            float d = -max_scale/32;
            dh[0] = half_t(d);
            float id = d ? 1/d : 0.f;
            for (int ib = 0; ib < super_block_size/block_size; ++ib) {
                int l = nearest_int(id*scales[ib]);
                l = std::max(-32, std::min(31, l));
                float dl = d * l;
                float idl = dl ? 1/dl : 0.f;
                uint8_t * Lb = L + ib*block_size;
                const float * xb = x_block + ib*block_size;
                for (int j = 0; j < block_size; ++j) {
                    Lb[j] = best_index_int8(16, values, idl*xb[j]);
                }
                l += 32;
                uint8_t l_l = l & 0xf;
                uint8_t l_h = l >>  4;
                if (ib%2 == 0) scales_l[ib/2] = l_l;
                else scales_l[ib/2] |= (l_l << 4);
                scales_h[ib/8] |= (l_h << 2*(ib%8));
            }
        } else {
            dh[0] = half_t(scales[0]);
            if (ntry > 0) {
                float id = scales[0] ? 1/scales[0] : 0;
                for (int j = 0; j < super_block_size; ++j) {
                    L[j] = best_index_int8(16, values, id*x_block[j]);
                }
            }
        }

        for (int i = 0; i < super_block_size/32; ++i) {
            for (int j = 0; j < 16; ++j) {
                q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
            }
        }
    }
}

template <typename T>
void dequant_row_iq4_xs(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_iq4_xs * __restrict__ x = reinterpret_cast<const block_iq4_xs *>(xr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = (float)x[i].d;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = (T)(dl * kvalues_iq4_nl[qs[j] & 0xf]);
                y[j+16] = (T)(dl * kvalues_iq4_nl[qs[j] >>  4]);
            }
            y  += 32;
            qs += 16;
        }
    }
}

void quant_row_iq4_xs(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t n, const float * quant_weights) {
    block_iq4_xs * __restrict__ y = reinterpret_cast<block_iq4_xs *>(yr);

    assert(n % QK4_NL == 0);
    int64_t nblock = n / QK4_NL;

    const int super_block_size = QK4_NL;
    const int block_size = 32;
    const int8_t* values = kvalues_iq4_nl;
    const int ntry = 7;
    
    float scales[super_block_size/block_size];
    float weight[super_block_size];
    uint8_t L[super_block_size];

    for (int ibl = 0; ibl < nblock; ++ibl) {

        half_t* dh = &y[ibl].d;
        uint8_t* q4 = y[ibl].qs;
        uint16_t* scales_h = &y[ibl].scales_h;
        uint8_t* scales_l = y[ibl].scales_l;
        
        const float* x_block = x + ibl * QK4_NL;
        const float* qw_block = quant_weights ? quant_weights + ibl * QK4_NL : nullptr;
        
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += x_block[j]*x_block[j];
        sigma2 *= 2.f/super_block_size;

        memset(q4, 0, super_block_size/2);
        dh[0] = half_t(0.f);

        float max_scale = 0, amax_scale = 0;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = x_block + ib*block_size;
            uint8_t * Lb = L + ib*block_size;
            if (qw_block) {
                const float * qw = qw_block + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < GROUP_MAX_EPS) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                Lb[j] = l;
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            d = sumqx/sumq2;
            float best = d*sumqx;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx = sumq2 = 0;
                for (int j = 0; j < block_size; ++j) {
                    float al = id*xb[j];
                    int l = best_index_int8(16, values, al);
                    float q = values[l];
                    float w = weight[j];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    d = sumqx/sumq2; best = d * sumqx;
                }
            }
            scales[ib] = d;
            float abs_d = fabsf(d);
            if (abs_d > amax_scale) {
                amax_scale = abs_d; max_scale = d;
            }
        }

        if (super_block_size/block_size > 1) {
            int nb = super_block_size/block_size;
            memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
            float d = -max_scale/32;
            dh[0] = half_t(d);
            float id = d ? 1/d : 0.f;
            for (int ib = 0; ib < super_block_size/block_size; ++ib) {
                int l = nearest_int(id*scales[ib]);
                l = std::max(-32, std::min(31, l));
                float dl = d * l;
                float idl = dl ? 1/dl : 0.f;
                uint8_t * Lb = L + ib*block_size;
                const float * xb = x_block + ib*block_size;
                for (int j = 0; j < block_size; ++j) {
                    Lb[j] = best_index_int8(16, values, idl*xb[j]);
                }
                l += 32;
                uint8_t l_l = l & 0xf;
                uint8_t l_h = l >>  4;
                if (ib%2 == 0) scales_l[ib/2] = l_l;
                else scales_l[ib/2] |= (l_l << 4);
                scales_h[ib/8] |= (l_h << 2*(ib%8));
            }
        } else {
            dh[0] = half_t(scales[0]);
            if (ntry > 0) {
                float id = scales[0] ? 1/scales[0] : 0;
                for (int j = 0; j < super_block_size; ++j) {
                    L[j] = best_index_int8(16, values, id*x_block[j]);
                }
            }
        }

        for (int i = 0; i < super_block_size/32; ++i) {
            for (int j = 0; j < 16; ++j) {
                q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
            }
        }
    }
}

//===================================== Q8_K ==============================================

template <typename T>
void dequant_row_q8_K(const uint8_t * __restrict__ xr, T * __restrict__ y, int64_t k) {
    const block_q8_K * __restrict__ x = reinterpret_cast<const block_q8_K *>(xr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = (T)(x[i].d * x[i].qs[j]);
        }
    }
}

void quant_row_q8_K(const float * __restrict__ x, uint8_t * __restrict__ yr, int64_t k) {
    block_q8_K * __restrict__ y = reinterpret_cast<block_q8_K *>(yr);

    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = std::min((int8_t) 127, (int8_t) v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}

// explicit instantiations for dequant row template

// float
template void dequant_row_q4_0<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q4_1<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q5_0<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q5_1<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q8_0<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_mxfp4<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q2_K<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q3_K<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q4_K<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q5_K<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q6_K<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_tq1_0<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_tq2_0<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq2_xxs<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq2_xs<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq2_s<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq3_xxs<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq3_s<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq1_s<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq1_m<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq4_nl<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_iq4_xs<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);
template void dequant_row_q8_K<float>(const uint8_t* __restrict__ xr, float* __restrict__ y, int64_t k);

// half 
template void dequant_row_q4_0<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q4_1<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q5_0<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q5_1<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q8_0<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_mxfp4<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q2_K<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q3_K<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q4_K<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q5_K<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q6_K<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_tq1_0<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_tq2_0<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq2_xxs<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq2_xs<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq2_s<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq3_xxs<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq3_s<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq1_s<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq1_m<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq4_nl<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_iq4_xs<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);
template void dequant_row_q8_K<half_t>(const uint8_t* __restrict__ xr, half_t* __restrict__ y, int64_t k);