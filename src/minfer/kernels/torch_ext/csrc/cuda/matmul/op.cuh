#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#include "common/types.hpp"
#include "lookup.cuh"
#include "kernels/xw.cuh"

// basic compatibility checks here
// note that this is optimized for L40S

namespace minfer::impl {

inline void dispatch_f16_xwt(
    size_t M, size_t K, size_t N,
    const half* __restrict__ x_ptr,
    const half* __restrict__ w_ptr,
    half* __restrict__ out_ptr
) {

}

template <
    unsigned int DIM_BM,
    unsigned int DIM_BK,
    unsigned int DIM_BN,
    unsigned int DIM_WM,
    unsigned int DIM_WK,
    unsigned int DIM_WN,
    unsigned int DIM_MM,
    unsigned int DIM_MK,
    unsigned int DIM_MN,
    unsigned int K_PIPE_MAX,
    unsigned int USE_SYNC
>
inline void launch_xw_kernel(
    size_t M, size_t K, size_t N,
    const half* __restrict__ x_ptr,
    const half* __restrict__ w_ptr,
    half* __restrict__ out_ptr,
    const cudaDeviceProp& deviceProp,
    int device_index
) {
    constexpr unsigned int WARPS_M = (DIM_BM+DIM_WM-1) / DIM_WM;
    constexpr unsigned int TILES_K = (DIM_BK+DIM_WK-1) / DIM_WK;
    constexpr unsigned int WARPS_N = (DIM_BN+DIM_WN-1) / DIM_WN;

    const unsigned int blocks_m = (M+DIM_BM-1)/DIM_BM;
    const unsigned int blocks_n = (N+DIM_BN-1)/DIM_BN;

    constexpr unsigned int SIZE_WARP = 32;
    constexpr unsigned int THRS_N = SIZE_WARP * WARPS_N;
    constexpr unsigned int THRS_M = WARPS_M;
    constexpr unsigned int NUM_THRS = THRS_M * THRS_N;
    constexpr unsigned int SHMEM_SZ = K_PIPE_MAX*(DIM_BM*DIM_BK+DIM_BK*DIM_BN)*sizeof(half);

    // (toShmem for x and w)
    static_assert(DIM_BM*(DIM_BK/8) >= NUM_THRS);
    static_assert(DIM_BK*(DIM_BN/8) >= NUM_THRS);

    // (toGmem for output)
    static_assert(DIM_BM*(DIM_BN/8) >= NUM_THRS);
    
    // (reuse shmem for output)
    static_assert(DIM_BM*DIM_BN*sizeof(half) <= K_PIPE_MAX*(DIM_BM*DIM_BK+DIM_BK*DIM_BN)*sizeof(half));
    
    STD_TORCH_CHECK(NUM_THRS <= deviceProp.maxThreadsPerBlock, "Too many threads (per block) requested");
    STD_TORCH_CHECK(SHMEM_SZ <= deviceProp.sharedMemPerBlockOptin, "Too much shmem (per block) requested");

    dim3 grid(blocks_n, blocks_m);
    dim3 block(THRS_N, THRS_M);

    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
    );
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    static bool shmem_attr_set = false;

    // set attr (avoid triggering JIT recompilation)
    if (!shmem_attr_set) {
        if constexpr (USE_SYNC == 1u) {
            STD_CUDA_CHECK(
                cudaFuncSetAttribute(
                    xw_sync_impl<
                        DIM_BM,
                        DIM_BK,
                        DIM_BN,
                        DIM_WM,
                        DIM_WK,
                        DIM_WN,
                        DIM_MM,
                        DIM_MK,
                        DIM_MN,
                        TILES_K,
                        NUM_THRS
                    >,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SHMEM_SZ
                )
            );
        } else {
            STD_CUDA_CHECK(
                cudaFuncSetAttribute(
                    xw_async_impl<
                        DIM_BM,
                        DIM_BK,
                        DIM_BN,
                        DIM_WM,
                        DIM_WK,
                        DIM_WN,
                        DIM_MM,
                        DIM_MK,
                        DIM_MN,
                        TILES_K,
                        K_PIPE_MAX,
                        NUM_THRS
                    >,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SHMEM_SZ
                )
            );
        }
        
        shmem_attr_set = true;
    }

    // launch
    if constexpr (USE_SYNC == 1) {
        xw_sync_impl<
            DIM_BM, DIM_BK, DIM_BN, 
            DIM_WM, DIM_WK, DIM_WN, 
            DIM_MM, DIM_MK, DIM_MN,
            TILES_K, NUM_THRS
        ><<<grid, block, SHMEM_SZ, stream>>>(
            M, K, N, x_ptr, w_ptr, out_ptr
        );
    } else {
        xw_async_impl<
            DIM_BM, DIM_BK, DIM_BN, 
            DIM_WM, DIM_WK, DIM_WN, 
            DIM_MM, DIM_MK, DIM_MN,
            TILES_K, K_PIPE_MAX, NUM_THRS
        ><<<grid, block, SHMEM_SZ, stream>>>(
            M, K, N, x_ptr, w_ptr, out_ptr
        );
    }
}

inline void dispatch_f16_xw(
    size_t M, size_t K, size_t N,
    const half* __restrict__ x_ptr,
    const half* __restrict__ w_ptr,
    half* __restrict__ out_ptr
) {
    static cudaDeviceProp deviceProp;
    static bool is_device_init = false; // assumes all GPUs are the same
    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();

    if (!is_device_init) { // NOTE: isn't an issue unless you're using different GPU types on a single node
        STD_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device_index));
        STD_TORCH_CHECK(deviceProp.major >= 8, "SM 8.0 or higher required");
        is_device_init = true;
    }

    const size_t best_idx = find_nearest_config(M, K, N);

    switch(best_idx) {
#define X(IDX) case IDX: { \
    constexpr auto& entry = LOOKUP_TABLE[IDX]; \
    launch_xw_kernel<entry.BM, entry.BK, entry.BN, entry.WM, entry.WK, entry.WN, entry.MM, entry.MK, entry.MN, entry.K_PIPE_MAX, entry.USE_SYNC>( \
        M, K, N, x_ptr, w_ptr, out_ptr, deviceProp, device_index); \
    break; }
LOOKUP_INDEX_CASES
#undef X
    }
}

// TODO: incomplete, finish
inline void dispatch_quant_xwt(
    int qtype_int, int qblock_size, int qtype_size,
    size_t M, size_t K, size_t N,
    const half* __restrict__ x_ptr,
    const uint8_t* __restrict__ w_ptr,
    half* __restrict__ out_ptr
) {

}

}

namespace minfer {

using namespace impl;

void matmul_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const torch::stable::Tensor& x, // [B,L,in_dim]
    const torch::stable::Tensor& w, // [out_dim, in_dim (possibly bytes)]
    torch::stable::Tensor& out // [B,L,out_dim]
) {
    STD_TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    STD_TORCH_CHECK(x.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(out.device().type() == torch::headeronly::DeviceType::CUDA);
    STD_TORCH_CHECK(w.device().type() == torch::headeronly::DeviceType::CUDA);

    STD_TORCH_CHECK(x.device() == w.device() && x.device() == out.device());

    STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    STD_TORCH_CHECK(
        w.scalar_type() == torch::headeronly::ScalarType::Byte || 
        w.scalar_type() == torch::headeronly::ScalarType::Half
    );

    STD_TORCH_CHECK(x.dim() == 3);
    STD_TORCH_CHECK(out.dim() == 3);
    STD_TORCH_CHECK(w.dim() == 2);

    STD_TORCH_CHECK(x.is_contiguous());
    STD_TORCH_CHECK(out.is_contiguous());
    STD_TORCH_CHECK(w.is_contiguous());

    STD_TORCH_CHECK(x.size(0) == out.size(0) && x.size(1) == out.size(1));
    STD_TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size) || 
                w.size(-1) == x.size(-1) || w.size(0) == x.size(-1));
    STD_TORCH_CHECK(w.size(0) == out.size(-1) || w.size(1) == out.size(-1));

    const half* x_ptr = reinterpret_cast<const half*>(x.const_data_ptr<half_t>());
    half* out_ptr = reinterpret_cast<half*>(out.mutable_data_ptr<half_t>());

    size_t M = x.numel() / x.size(-1);
    size_t K = x.size(-1);

    if (qtype == GGMLQuantizationType::F16) {
        const half* w_ptr = reinterpret_cast<const half*>(w.const_data_ptr<half_t>());

        if (w.size(0) == x.size(-1)) {
            // X @ W
            size_t N = w.size(1);
            dispatch_f16_xw(
                M, K, N, x_ptr, w_ptr, out_ptr
            );
        } else if (w.size(-1) == x.size(-1)) {
            // X @ W.T
            size_t N = w.size(0);
            dispatch_f16_xwt(
                M, K, N, x_ptr, w_ptr, out_ptr
            );
        }
    } else {
        STD_TORCH_CHECK(is_dequant_qtype(qtype_int));
        const uint8_t* w_ptr = w.const_data_ptr<uint8_t>();
        
        STD_TORCH_CHECK(K%qblock_size==0);

        if (w.size(-1) == (x.size(-1)/qblock_size)*qtype_size) {
            // X @ W.T, W quantized and of shape [N, K_bytes]
            size_t N = w.size(0);
            dispatch_quant_xwt(
                qtype_int, qblock_size, qtype_size, M, K, N, x_ptr, w_ptr, out_ptr
            );
        } else {
            STD_TORCH_CHECK(false, "Unsupported dims for quant matmul");
        }
    }

    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}