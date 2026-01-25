#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernels/xw.cuh"

// basic compatibility checks here
// note that this is optimized for V100

namespace minfer::impl {

inline void dispatch_f16_xwt(
    size_t M, size_t K, size_t N,
    const half* __restrict__ x_ptr,
    const half* __restrict__ w_ptr,
    half* __restrict__ out_ptr
) {

}

// TODO: incomplete, finish
inline void dispatch_f16_xw(
    size_t M, size_t K, size_t N,
    const half* __restrict__ x_ptr,
    const half* __restrict__ w_ptr,
    half* __restrict__ out_ptr
) {
    auto device_index = torch::stable::accelerator::getCurrentDeviceIndex();

    cudaDeviceProp deviceProp;
    STD_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device_index));

    STD_TORCH_CHECK(deviceProp.major >= 8, "SM 8.0 or higher required to use tensor cores + async memcpy");

    // eventually: dispatch various configurations of 
    // these vals based on problem size
    constexpr unsigned int DIM_BM = 512;
    constexpr unsigned int DIM_BK = 32;
    constexpr unsigned int DIM_BN = 256;

    // Turing (and beyond) supports ldmatrix m16n8k16 instruction
    constexpr unsigned int K_PIPE_MAX = 2;
    constexpr unsigned int WARPS_M = 4;
    constexpr unsigned int TILES_K = 2;
    constexpr unsigned int WARPS_N = 2;

    constexpr unsigned int DIM_WM = (DIM_BM+WARPS_M-1) / WARPS_M;
    constexpr unsigned int DIM_WK = (DIM_BK+TILES_K-1) / TILES_K;
    constexpr unsigned int DIM_WN = (DIM_BN+WARPS_N-1) / WARPS_N;

    const unsigned int blocks_m = (M+DIM_BM-1)/DIM_BM;
    const unsigned int blocks_n = (N+DIM_BN-1)/DIM_BN;

    constexpr unsigned int SIZE_WARP = 32;
    constexpr unsigned int THRS_N = SIZE_WARP * WARPS_N;
    constexpr unsigned int THRS_M = WARPS_M;
    constexpr unsigned int NUM_THRS = THRS_M * THRS_N;
    constexpr unsigned int SHMEM_SZ = K_PIPE_MAX * (DIM_BM*DIM_BK + DIM_BK*DIM_BN) * sizeof(half);

    STD_TORCH_CHECK(SHMEM_SZ <= deviceProp.sharedMemPerBlockOptin, "Too much shmem (per block) requested");

    dim3 grid(blocks_n, blocks_m);
    dim3 block(THRS_N, THRS_M);

    STD_CUDA_CHECK(
        cudaFuncSetAttribute(
            xw_impl<
                DIM_BM,
                DIM_BK,
                DIM_BN,
                DIM_WM,
                DIM_WK,
                DIM_WN,
                TILES_K,
                K_PIPE_MAX,
                NUM_THRS
            >,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHMEM_SZ
        )
    );

    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)
    );
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    xw_impl<
        DIM_BM, DIM_BK, DIM_BN, 
        DIM_WM, DIM_WK, DIM_WN, 
        TILES_K, K_PIPE_MAX, NUM_THRS
    ><<<grid, block, SHMEM_SZ, stream>>>(
        M, K, N, x_ptr, w_ptr, out_ptr
    );
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