#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <torch/csrc/stable/c/shim.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernels/256x128x128.cuh"

// basic compatibility checks here
// note that this is optimized for V100

namespace {

    using namespace minfer::impl;

    inline void dispatch_f16_xwt(
        size_t M, size_t K, size_t N,
        const half* __restrict__ x_ptr,
        const half* __restrict__ w_ptr,
        half* __restrict__ out_ptr
    ) {

        // cudaDeviceProp deviceProp;
        // AT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, x.device().index()));

        // if (deviceProp.major < 7) {
        //     TORCH_CHECK(false, "SM 7.0 or higher required to use tensor cores");
        // }

        // if (deviceProp.sharedMemPerMultiprocessor < SHMEM_SZ) {
        //     TORCH_CHECK(false, "Not enough shared memory for performant kernel");
        // }

        // dim3 grid(deviceProp.multiProcessorCount);
        // dim3 block(THREADS_PER_BLOCK);

        // AT_CUDA_CHECK(
        //     cudaFuncSetAttribute(
        //         matmul_wmma_f16_cuda_impl<Config>,
        //         cudaFuncAttributeMaxDynamicSharedMemorySize,
        //         SHMEM_SZ
        //     )
        // );

        // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
        // matmul_wmma_f16_cuda_impl<Config><<<grid, block, SHMEM_SZ, stream>>>(
        //     M, K, N, x_ptr, w_ptr, out_ptr
        // );
    }

    // TODO: incomplete, finish
    inline void dispatch_f16_xw(
        size_t M, size_t K, size_t N,
        const half* __restrict__ x_ptr,
        const half* __restrict__ w_ptr,
        half* __restrict__ out_ptr
    ) {
        int device;
        cudaDeviceProp deviceProp;
        AT_CUDA_CHECK(cudaGetDevice(&device));
        AT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        TORCH_CHECK(deviceProp.major >= 7, "SM 7.0 or higher required to use tensor cores");

        // eventually: dispatch various configurations of 
        // these vals based on problem size
        constexpr unsigned int DIM_BM = 256;
        constexpr unsigned int DIM_BK = 128;
        constexpr unsigned int DIM_BN = 128;

        // Turing (and beyond) supports ldmatrix m16n8k16 instruction
        constexpr unsigned int WARPS_M = 4;
        constexpr unsigned int WARPS_K = 4; // this one is more like "k tiles per block" rather than warps per block in the K dimension, i think?
        constexpr unsigned int WARPS_N = 2;

        constexpr unsigned int DIM_WM = (DIM_BM+WARPS_M-1) / WARPS_M;
        constexpr unsigned int DIM_WK = (DIM_BK+WARPS_K-1) / WARPS_K;
        constexpr unsigned int DIM_WN = (DIM_BN+WARPS_N-1) / WARPS_N;

        const unsigned int blocks_m = (M+DIM_BM-1)/DIM_BM;
        const unsigned int blocks_n = (N+DIM_BN-1)/DIM_BN;

        constexpr unsigned int SIZE_WARP = 32;
        constexpr unsigned int THRS_N = SIZE_WARP * WARPS_N;
        constexpr unsigned int THRS_M = WARPS_M;
        constexpr unsigned int NUM_THRS = THRS_M * THRS_N;
        constexpr unsigned int SHMEM_SZ = (DIM_BM*DIM_BK + DIM_BK*DIM_BN) * sizeof(half);

        TORCH_CHECK(SHMEM_SZ <= deviceProp.sharedMemPerBlockOptin, "Too much shmem (per block) requested");

        dim3 grid(blocks_n, blocks_m);
        dim3 block(THRS_N, THRS_M);

        AT_CUDA_CHECK(
            cudaFuncSetAttribute(
                xw_256x128x128<
                    DIM_BM,
                    DIM_BK,
                    DIM_BN,
                    DIM_WM,
                    DIM_WK,
                    DIM_WN,
                    WARPS_K,
                    NUM_THRS
                >,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SHMEM_SZ
            )
        );

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        xw_256x128x128<
            DIM_BM, DIM_BK, DIM_BN, 
            DIM_WM, DIM_WK, DIM_WN, 
            WARPS_K, NUM_THRS
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

        // cudaDeviceProp deviceProp;
        // AT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, x.device().index()));

        // if (deviceProp.major < 7) {
        //     TORCH_CHECK(false, "SM 7.0 or higher required to use tensor cores");
        // }
        
        // if (deviceProp.sharedMemPerMultiprocessor < SHMEM_SZ) {
        //     TORCH_CHECK(false, "Not enough shared memory for performant kernel");
        // }

        // dim3 grid(deviceProp.multiProcessorCount);
        // dim3 block(THREADS_PER_BLOCK);

        // AT_CUDA_CHECK(
        //     cudaFuncSetAttribute(
        //         matmul_wmma_quant_cuda_impl<Config>,
        //         cudaFuncAttributeMaxDynamicSharedMemorySize,
        //         SHMEM_SZ
        //     )
        // );
        
        // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        // matmul_wmma_quant_cuda_impl<Config><<<grid, block, SHMEM_SZ, stream>>>(
        //     qtype_int, qblock_size, qtype_size,
        //     M, K, N, 
        //     x_ptr, w_ptr, out_ptr
        // );
    }

}

void matmul_cuda(
    int64_t qtype_int,
    int64_t qblock_size,
    int64_t qtype_size,
    const at::Tensor& x, // [B,L,in_dim]
    const at::Tensor& w, // [out_dim, in_dim (possibly bytes)]
    at::Tensor& out // [B,L,out_dim]
) {
    TORCH_CHECK(is_valid_qtype(qtype_int));
    auto qtype = static_cast<GGMLQuantizationType>(qtype_int);

    TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(w.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(x.device() == w.device() && x.device() == out.device());

    TORCH_CHECK(x.dtype() == at::kHalf);
    TORCH_CHECK(out.dtype() == at::kHalf);
    TORCH_CHECK(w.dtype() == at::kByte || w.dtype() == at::kHalf);

    TORCH_CHECK(x.dim() == 3);
    TORCH_CHECK(out.dim() == 3);
    TORCH_CHECK(w.dim() == 2);

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(w.is_contiguous());

    TORCH_CHECK(x.size(0) == out.size(0) && x.size(1) == out.size(1));
    TORCH_CHECK(w.size(-1) == ((x.size(-1) / qblock_size) * qtype_size) || 
                w.size(-1) == x.size(-1) || w.size(0) == x.size(-1));
    TORCH_CHECK(w.size(0) == out.size(-1) || w.size(1) == out.size(-1));

    const half* x_ptr = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());

    size_t M = x.numel() / x.size(-1);
    size_t K = x.size(-1);

    if (qtype == GGMLQuantizationType::F16) {
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());

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
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
        
        TORCH_CHECK(K%qblock_size==0);

        if (w.size(-1) == (x.size(-1)/qblock_size)*qtype_size) {
            // X @ W.T, W quantized and of shape [N, K_bytes]
            size_t N = w.size(0);
            dispatch_quant_xwt(
                qtype_int, qblock_size, qtype_size, M, K, N, x_ptr, w_ptr, out_ptr
            );
        } else {
            TORCH_CHECK(false, "Unsupported dims for quant matmul");
        }
    }

    AT_CUDA_CHECK(cudaPeekAtLastError());
}