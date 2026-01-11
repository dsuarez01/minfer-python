#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common/types.hpp"
#include "kernel.cuh"

template<typename Config>
inline void launch_wmma_f16_kernel(
    cudaDeviceProp& deviceProp,
    cudaStream_t stream,
    int64_t M, int64_t K, int64_t N,
    const half* x_ptr,
    const half* w_ptr,
    half* out_ptr
) {
    static constexpr auto THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    static constexpr auto SHMEM_SZ = Config::SHMEM_SZ;

    dim3 grid(deviceProp.multiProcessorCount);
    dim3 block(THREADS_PER_BLOCK);
    
    if (deviceProp.sharedMemPerMultiprocessor < SHMEM_SZ) {
        TORCH_CHECK(false, "Not enough shared memory for performant kernel");
    }

    AT_CUDA_CHECK(
        cudaFuncSetAttribute(
            matmul_wmma_f16_cuda_impl<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHMEM_SZ
        )
    );
    
    matmul_wmma_f16_cuda_impl<Config><<<grid, block, SHMEM_SZ, stream>>>(
        M, K, N, x_ptr, w_ptr, out_ptr
    );
}

template<typename Config>
inline void launch_wmma_quant_kernel(
    cudaDeviceProp& deviceProp,
    cudaStream_t stream,
    int64_t qtype_int, int64_t qblock_size, int64_t qtype_size,
    int64_t M, int64_t K, int64_t N,
    const half* x_ptr,
    const uint8_t* w_ptr,
    half* out_ptr
) {
    static constexpr auto THREADS_PER_BLOCK = Config::THREADS_PER_BLOCK;
    static constexpr auto SHMEM_SZ = Config::SHMEM_SZ;

    dim3 grid(deviceProp.multiProcessorCount);
    dim3 block(THREADS_PER_BLOCK);
    
    if (deviceProp.sharedMemPerMultiprocessor < SHMEM_SZ) {
        TORCH_CHECK(false, "Not enough shared memory for performant kernel");
    }

    AT_CUDA_CHECK(
        cudaFuncSetAttribute(
            matmul_wmma_quant_cuda_impl<Config>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SHMEM_SZ
        )
    );
    
    matmul_wmma_quant_cuda_impl<Config><<<grid, block, SHMEM_SZ, stream>>>(
        qtype_int, qblock_size, qtype_size,
        M, K, N, 
        x_ptr, w_ptr, out_ptr
    );
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

    int64_t M = x.numel() / x.size(-1);
    int64_t K = x.size(-1);
    
    cudaDeviceProp deviceProp;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, x.device().index()));

    if (deviceProp.major < 7) {
        TORCH_CHECK(false, "SM 7.0 or higher required to use tensor cores");
    }
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (qtype == GGMLQuantizationType::F16) {
        const half* w_ptr = reinterpret_cast<const half*>(w.data_ptr<at::Half>());

        if (w.size(-1) == x.size(-1)) {
            // X @ W.T
            int64_t N = w.size(0);
            launch_wmma_f16_kernel<Config_Matmul_F16_T>(
                deviceProp, stream, M, K, N, x_ptr, w_ptr, out_ptr
            );
        } else if (w.size(0) == x.size(-1)) {
            // X @ W
            int64_t N = w.size(1);
            launch_wmma_f16_kernel<Config_Matmul_F16>(
                deviceProp, stream, M, K, N, x_ptr, w_ptr, out_ptr
            );
        }
    } else {
        const uint8_t* w_ptr = w.data_ptr<uint8_t>();
        TORCH_CHECK(K%qblock_size==0);

        if (w.size(-1) == (x.size(-1)/qblock_size)*qtype_size) {
            // X @ W.T, W quantized
            int64_t N = w.size(0);
            launch_wmma_quant_kernel<Config_Matmul_Quant_T>(
                deviceProp, stream, qtype_int, qblock_size, qtype_size, M, K, N, x_ptr, w_ptr, out_ptr
            );
        } else {
            TORCH_CHECK(false, "Unsupported dims for quant matmul");
        }
    }

    AT_CUDA_CHECK(cudaGetLastError());
}