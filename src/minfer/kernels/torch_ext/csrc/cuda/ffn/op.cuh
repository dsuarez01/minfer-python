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
#include "kernel.cuh"

namespace minfer {

using namespace impl;

// elemwise multiplication of up proj tile with swiglu (tile of swish(gate(x))), then apply downproj
void ffn_cuda(
    int64_t up_qtype_int, int64_t gate_qtype_int, int64_t down_qtype_int,
    int64_t up_qblock_size, int64_t gate_qblock_size, int64_t down_qblock_size,
    int64_t up_qtype_size, int64_t gate_qtype_size, int64_t down_qtype_size,
    const torch::stable::Tensor& in, // [B,L,hidden_dim]
    const torch::stable::Tensor& ws_up, // [n_local_exps, mlp_dim, hidden_dim]
    const torch::stable::Tensor& ws_gate, // // [n_local_exps, mlp_dim, hidden_dim]
    const torch::stable::Tensor& ws_down, // [n_local_exps, hidden_dim, mlp_dim]
    torch::stable::Tensor& hb, // [n_local_exps,B,L,mlp_dim]
    torch::stable::Tensor& hb2, // [n_local_exps,B,L,mlp_dim]
    torch::stable::Tensor& out // [n_local_exps,B,L,hidden_dim]
) {
    // TORCH_CHECK(is_valid_qtype(up_qtype_int) && is_valid_qtype(gate_qtype_int) && is_valid_qtype(down_qtype_int));
    // auto up_qtype = static_cast<GGMLQuantizationType>(up_qtype_int);
    // auto gate_qtype = static_cast<GGMLQuantizationType>(gate_qtype_int);
    // auto down_qtype = static_cast<GGMLQuantizationType>(down_qtype_int);

    // //validation
    // TORCH_INTERNAL_ASSERT(in.device().type() == at::DeviceType::CUDA);
    // TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
    // TORCH_INTERNAL_ASSERT(hb.device().type() == at::DeviceType::CUDA);
    // TORCH_INTERNAL_ASSERT(hb2.device().type() == at::DeviceType::CUDA);
    // TORCH_INTERNAL_ASSERT(ws_up.device().type() == at::DeviceType::CUDA);
    // TORCH_INTERNAL_ASSERT(ws_gate.device().type() == at::DeviceType::CUDA);
    // TORCH_INTERNAL_ASSERT(ws_down.device().type() == at::DeviceType::CUDA);

    // //contiguity
    // TORCH_CHECK(
    //     in.is_contiguous() && out.is_contiguous() && hb.is_contiguous() && hb2.is_contiguous() && 
    //     ws_up.is_contiguous() && ws_gate.is_contiguous() && ws_down.is_contiguous()
    // );

    // //dtypes
    // TORCH_CHECK(
    //     (in.dtype() == at::kHalf) && (out.dtype() == at::kHalf) && 
    //     (hb.dtype() == at::kHalf) && (hb2.dtype() == at::kHalf) &&
    //     (ws_up.dtype() == at::kByte || ws_up.dtype() == at::kHalf) && 
    //     (ws_gate.dtype() == at::kByte || ws_gate.dtype() == at::kHalf) && 
    //     (ws_down.dtype() == at::kByte || ws_down.dtype() == at::kHalf)
    // );

    // //dims
    // TORCH_CHECK((in.dim() == 3) && (out.dim() == 4));
    // TORCH_CHECK((hb.dim() == 4) && (hb2.dim() == 4));
    // TORCH_CHECK((ws_up.dim() == 3) && (ws_gate.dim() == 3) && (ws_down.dim() == 3));

    // //checks between dimensions
    // TORCH_CHECK(in.numel() == out.numel() / out.size(0));
    // TORCH_CHECK(hb.sizes().equals(hb2.sizes()));
    // TORCH_CHECK(
    //     ws_up.size(0) == ws_gate.size(0) && ws_gate.size(0) == ws_down.size(0) && ws_up.size(1) == ws_gate.size(1)
    // );

    // const half* in_ptr = reinterpret_cast<const half*>(in.data_ptr<at::Half>());
    // half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    // half* hb_ptr = reinterpret_cast<half*>(hb.data_ptr<at::Half>());
    // half* hb2_ptr = reinterpret_cast<half*>(hb2.data_ptr<at::Half>());

    // int64_t B = in.size(0);
    // int64_t L = in.size(1);
    // int64_t hidden_dim = in.size(-1);
    // int64_t n_local_exps = out.size(0);
    // int64_t mlp_dim = hb.size(3);

    // int64_t M = B*L;
    // int64_t K_GU = hidden_dim;
    // int64_t N_GU = mlp_dim;
    // int64_t K_DOWN = mlp_dim;
    // int64_t N_DOWN = hidden_dim;
    
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // // assume: all of gate, up, down are dequantized (F16) or all are quantized i.e. not FP16
    // if (
    //     up_qtype == GGMLQuantizationType::F16 && 
    //     gate_qtype == GGMLQuantizationType::F16 && 
    //     down_qtype == GGMLQuantizationType::F16
    // ) {
    //     const half* ws_up_ptr = reinterpret_cast<const half*>(ws_up.data_ptr<at::Half>());
    //     const half* ws_gate_ptr = reinterpret_cast<const half*>(ws_gate.data_ptr<at::Half>());
    //     const half* ws_down_ptr = reinterpret_cast<const half*>(ws_down.data_ptr<at::Half>());

    //     constexpr int WARPS_PER_BLOCK = 4; // tune this as needed
    //     constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
    //     constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
    //     dim3 grid_swiglu(n_local_exps, (M+ROWS_M-1)/ROWS_M, (N_GU+WMMA_N-1)/WMMA_N);
    //     dim3 grid_down_exp((M+ROWS_M-1)/ROWS_M, (N_DOWN+WMMA_N-1)/WMMA_N);
    //     dim3 block(BLOCK_SIZE);

    //     swiglu
    //     swiglu_f16_cuda_impl<WARPS_PER_BLOCK><<<grid_swiglu, block, 0, stream>>>(
    //         M, K_GU, N_GU,
    //         in_ptr, ws_up_ptr, ws_gate_ptr,
    //         hb_ptr, hb2_ptr
    //     );

    //     launch down-proj kernel per exp
    //     for (int e=0; e<n_local_exps; e++) {
    //         half* hb2_exp = hb2_ptr + e*M*N_GU;
    //         half* out_exp = out_ptr + e*M*N_DOWN;
    //         const half* ws_down_exp = ws_down_ptr + e*N_DOWN*K_DOWN;
            
    //         matmul_wmma_f16_cuda_impl<<<grid_down_exp, block, 0, stream>>>(
    //             M, K_DOWN, N_DOWN,
    //             hb2_exp, ws_down_exp, out_exp
    //         );
    //     }
    // } else {
    //     STD_TORCH_CHECK(is_dequant_qtype(qtype_int));
    //     const uint8_t* ws_up_ptr = ws_up.data_ptr<uint8_t>();
    //     const uint8_t* ws_gate_ptr = ws_gate.data_ptr<uint8_t>();
    //     const uint8_t* ws_down_ptr = ws_down.data_ptr<uint8_t>();

    //     constexpr int WARPS_PER_BLOCK = 4; // tune this as needed
    //     constexpr int BLOCK_SIZE = WARPS_PER_BLOCK*32;
    //     constexpr int ROWS_M = WARPS_PER_BLOCK*WMMA_M;
    //     dim3 grid_swiglu(n_local_exps, (M+ROWS_M-1)/ROWS_M, (N_GU+WMMA_N-1)/WMMA_N);
    //     dim3 grid_down_exp((M+ROWS_M-1)/ROWS_M, (N_DOWN+WMMA_N-1)/WMMA_N);
    //     dim3 block(BLOCK_SIZE);

    //     swiglu
    //     swiglu_quant_cuda_impl<WARPS_PER_BLOCK><<<grid_swiglu, block, 0, stream>>>(
    //         up_qtype_int, gate_qtype_int,
    //         up_qblock_size, gate_qblock_size,
    //         up_qtype_size, gate_qtype_size,
    //         M, K_GU, N_GU,
    //         in_ptr, ws_up_ptr, ws_gate_ptr,
    //         hb_ptr, hb2_ptr
    //     );

    //     launch down-proj kernel per expert
    //     for (int e=0; e<n_local_exps; e++) {
    //         half* hb2_exp = hb2_ptr + e*M*N_GU;
    //         half* out_exp = out_ptr + e*M*N_DOWN;
    //         const uint8_t* ws_down_exp = ws_down_ptr + e*N_DOWN*(K_DOWN/down_qblock_size)*down_qtype_size;
            
    //         matmul_quant_cuda_impl<<<grid_down_exp, block, 0, stream>>>(
    //             down_qtype_int, down_qblock_size, down_qtype_size,
    //             M, K_DOWN, N_DOWN,
    //             hb2_exp, ws_down_exp, out_exp
    //         );
    //     }
    // }

    // STD_CUDA_KERNEL_LAUNCH_CHECK();
}

}