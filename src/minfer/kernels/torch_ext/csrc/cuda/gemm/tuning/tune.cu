#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "tune.cuh"
#include "kernels/gemm_f16.cuh"

inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
#define checkCudaErrors(call) checkCuda(call, __FILE__, __LINE__)

namespace minfer::tuning {

    using namespace minfer::impl;

    // uniform over [lo, hi]
    void rand_init(half* d_ptr, size_t n, float lo, float hi) {
        std::vector<half> h(n);
        for (size_t i = 0; i < n; ++i)
            h[i] = __float2half(lo + (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * (hi-lo));
        checkCudaErrors(cudaMemcpy(d_ptr, h.data(), n*sizeof(half), cudaMemcpyHostToDevice));
    }

    template<size_t IDX>
    void launch_kernel_impl(
        int M,
        int K,
        int N,
        float alpha,
        float beta,
        const half* d_A,
        const half* d_B,
        const half* d_C,
        half* d_D
    ) {
        constexpr auto& kc = ALL_KERNEL_CONFIGS[IDX];
        constexpr unsigned int BM = kc.BM, BK = kc.BK, BN = kc.BN;
        constexpr unsigned int WM = kc.WM, WK = kc.WK, WN = kc.WN;
        constexpr unsigned int MM = kc.MM, MK = kc.MK, MN = kc.MN;
        constexpr unsigned int K_PIPE_MAX = kc.K_PIPE_MAX;
        constexpr unsigned int USE_SYNC = kc.USE_SYNC;
        
        constexpr unsigned int WARPS_M = (BM+WM-1)/WM;
        constexpr unsigned int WARPS_N = (BN+WN-1)/WN;
        constexpr unsigned int TILES_K = (BK+WK-1)/WK;
        
        constexpr unsigned int WARP_SIZE = 32;
        constexpr unsigned int THRS_N = WARP_SIZE*WARPS_N;
        constexpr unsigned int THRS_M = WARPS_M;
        constexpr unsigned int NUM_THRS = THRS_M*THRS_N;
        constexpr unsigned int SHMEM_SZ = K_PIPE_MAX*(BM*BK+BK*BN)*sizeof(half);
        
        const unsigned int blocks_m = (M+BM-1)/BM;
        const unsigned int blocks_n = (N+BN-1)/BN;
        
        dim3 grid(blocks_n, blocks_m);
        dim3 block(THRS_N, THRS_M);
        
        if constexpr (USE_SYNC == 1) {
            checkCudaErrors(
                cudaFuncSetAttribute(
                    ab_sync_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, NUM_THRS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SHMEM_SZ
                )
            );
            
            ab_sync_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, NUM_THRS>
                <<<grid, block, SHMEM_SZ>>>(M, K, N, alpha, beta, d_A, d_B, d_C, d_D);
        } else {

            checkCudaErrors(
                cudaFuncSetAttribute(
                    ab_async_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, K_PIPE_MAX, NUM_THRS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SHMEM_SZ
                )
            );
            
            ab_async_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, K_PIPE_MAX, NUM_THRS>
                <<<grid, block, SHMEM_SZ>>>(M, K, N, alpha, beta, d_A, d_B, d_C, d_D);
        }
        
        checkCudaErrors(cudaGetLastError());
    }

    void launch_kernel(
        size_t idx, 
        int M, 
        int K, 
        int N, 
        float alpha, 
        float beta, 
        const half* d_A, 
        const half* d_B, 
        const half* d_C, 
        half* d_D
    ) {
        switch(idx) {
#define X(IDX) case IDX: return launch_kernel_impl<IDX>(M, K, N, alpha, beta, d_A, d_B, d_C, d_D);
KERNEL_CONFIG_INDICES
#undef X
            default: throw std::runtime_error("Invalid config idx");
        }
    }

    Result run_benchmark(const Config& config) {
        auto M = config.M, K = config.K, N = config.N;
        auto alpha = config.alpha, beta = config.beta;
    
        half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
        cudaEvent_t start = nullptr, stop = nullptr;

        try {
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
    
            checkCudaErrors(cudaMalloc(&d_A, M*K*sizeof(half)));
            checkCudaErrors(cudaMalloc(&d_B, K*N*sizeof(half)));
            checkCudaErrors(cudaMalloc(&d_C, M*N*sizeof(half)));
            checkCudaErrors(cudaMalloc(&d_D, M*N*sizeof(half)));
            
            rand_init(d_A, M*K, -1.0f, 1.0f);
            rand_init(d_B, K*N, -1.0f, 1.0f);
            rand_init(d_C, M*N, -1.0f, 1.0f);

            // measure timer overhead 
            std::vector<float> overhead_samples;
            for (int i = 0; i < 5; ++i) {
                checkCudaErrors(cudaEventRecord(start));
                // NOTE: just timer overhead
                checkCudaErrors(cudaEventRecord(stop));
                checkCudaErrors(cudaEventSynchronize(stop));

                float ms;
                checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
                overhead_samples.push_back(ms);
            }
            std::sort(overhead_samples.begin(), overhead_samples.end());
            float overhead_ms = overhead_samples[2]; // median

            // estimate block size (adaptive warm-up)
            // block size determined by when total time taken 
            // is large enough to be < 0.01% of timer overhead
            int block_size = 1;
            float elapsed_ms = 0.0f;

            while (true) {
                checkCudaErrors(cudaEventRecord(start));
                for (int i = 0; i < block_size; ++i) {
                    launch_kernel(config.config_idx, M, K, N, alpha, beta, d_A, d_B, d_C, d_D);
                }
                checkCudaErrors(cudaEventRecord(stop));
                checkCudaErrors(cudaEventSynchronize(stop));
                
                checkCudaErrors(cudaEventElapsedTime(&elapsed_ms, start, stop));
                
                float rel_overhead = overhead_ms / elapsed_ms;
                
                if (rel_overhead <= 1e-4 && elapsed_ms >= config.min_run_time_ms) {
                    break;
                }

                if (elapsed_ms > config.target_time_ms) {
                    break;
                }

                if (block_size >= INT_MAX / 10) {
                    break;
                }
                
                block_size *= 10;
            }
            
            // main benchmark
            std::vector<double> times;
            double total_time_ms = 0.0;
            
            while (total_time_ms < config.target_time_ms) {
                checkCudaErrors(cudaEventRecord(start));
                for (int i = 0; i < block_size; ++i) {
                    launch_kernel(config.config_idx, M, K, N, alpha, beta, d_A, d_B, d_C, d_D);
                }
                checkCudaErrors(cudaEventRecord(stop));
                checkCudaErrors(cudaEventSynchronize(stop));
                
                float block_ms;
                checkCudaErrors(cudaEventElapsedTime(&block_ms, start, stop));
                
                double per_iter_ms = block_ms / block_size;
                times.push_back(per_iter_ms);
                total_time_ms += block_ms;
            }

            // compute timing/throughput stats
            int total_iters = times.size() * block_size;
            std::sort(times.begin(), times.end());
            double median_ms = times[times.size()/2];
            double min_ms = times.front();
            double max_ms = times.back();

            double flops = 2.0*M*K*N + 2.0*M*N;
            float tflops = (flops / (median_ms * 1e-3)) / 1e12;

            // clean-up
            checkCudaErrors(cudaFree(d_A));
            checkCudaErrors(cudaFree(d_B));
            checkCudaErrors(cudaFree(d_C));
            checkCudaErrors(cudaFree(d_D));
            checkCudaErrors(cudaEventDestroy(start));
            checkCudaErrors(cudaEventDestroy(stop));
            
            return Result{
                config, 
                block_size,
                total_iters,
                median_ms, 
                min_ms, 
                max_ms, 
                tflops
            };
        } catch(...) {
            // cleanup on err
            if (d_A) cudaFree(d_A);
            if (d_B) cudaFree(d_B);
            if (d_C) cudaFree(d_C);
            if (d_D) cudaFree(d_D);
            if (start) cudaEventDestroy(start);
            if (stop) cudaEventDestroy(stop);
            throw;
        }
    }

    int tune_run(const std::string& mode, int job_id, size_t config_idx, size_t M, size_t K, size_t N, float alpha, float beta) {
        double target_time_ms = 200.0; // how long we want each case to run in total (in ms)
        double min_run_time_ms = target_time_ms / 1000.0; // min run time for one block (in ms)

        std::filesystem::create_directories("./logs");

        std::string write_path = "./logs/" + mode + "_tuning_results_job" + std::to_string(job_id) + ".csv";
        bool write_header = !std::filesystem::exists(write_path);

        std::ofstream results_file(write_path, std::ios::app);
        if (!results_file.is_open()) {
            std::cerr << "Error at mode: " << mode << std::endl;
            std::cerr << "ERROR: Could not open results CSV" << std::endl;
            return 1;
        }

        if (write_header) {
            results_file << "M,K,N,alpha,beta,config_idx,success,"
                         << "BM,BK,BN,WM,WK,WN,MM,MK,MN,K_PIPE_MAX,USE_SYNC,"
                         << "median_ms,min_ms,max_ms,tflops,block_size,target_time_ms,min_time_ms\n";
            results_file.flush();
        }

        auto config = Config{
            config_idx, 
            ALL_KERNEL_CONFIGS[config_idx], 
            M, K, N, 
            alpha, beta,
            target_time_ms, 
            min_run_time_ms
        };

        try {
            auto result = run_benchmark(config);

            auto& kc = result.config.kc;
            results_file << M << "," << K << "," << N << ","
                         << alpha << "," << beta << ","
                         << config_idx << ","
                         << 1 << ","
                         << kc.BM << "," << kc.BK << "," << kc.BN << ","
                         << kc.WM << "," << kc.WK << "," << kc.WN << ","
                         << kc.MM << "," << kc.MK << "," << kc.MN << ","
                         << kc.K_PIPE_MAX << "," << kc.USE_SYNC << ","
                         << result.median_time_ms << "," << result.min_time_ms << ","
                         << result.max_time_ms << "," << result.tflops << ","
                         << result.block_size << "," << config.target_time_ms << ","
                         << config.min_run_time_ms << "\n";
            results_file.flush();
        } catch (const std::exception& e) {
            std::cerr << "Error at mode: " << mode << std::endl;
            std::cerr << "\tProblem size M=" << M << " K=" << K << " N=" << N << std::endl;
            std::cerr << "\tScales alpha=" << alpha << " beta=" << beta << std::endl;
            std::cerr << "\tConfig " << config_idx << " failed: " << e.what() << std::endl;

              
            results_file << M << "," << K << "," << N << ","
                         << alpha << "," << beta << ","
                         << config_idx << ","
                         << 0 << ","
                         << -1 << "," << -1 << "," << -1 << ","  // BM, BK, BN
                         << -1 << "," << -1 << "," << -1 << ","  // WM, WK, WN
                         << -1 << "," << -1 << "," << -1 << ","  // MM, MK, MN
                         << -1 << "," << -1 << ","               // K_PIPE_MAX, USE_SYNC
                         << -1 << "," << -1 << ","               // median_ms, min_ms
                         << -1 << "," << -1 << ","               // max_ms, tflops
                         << -1 << ","                             // block_size
                         << config.target_time_ms << ","
                         << config.min_run_time_ms << "\n";
            results_file.flush();
        } catch (...) {
            std::cerr << "Error at mode: " << mode << std::endl;
            std::cerr << "\tProblem size M=" << M << " K=" << K << " N=" << N << std::endl;
            std::cerr << "\tScales alpha=" << alpha << " beta=" << beta << std::endl;
            std::cerr << "\tConfig " << config_idx << " failed w/ unknown err" << std::endl;

            results_file << M << "," << K << "," << N << ","
                         << alpha << "," << beta << ","
                         << config_idx << ","
                         << 0 << ","
                         << -1 << "," << -1 << "," << -1 << ","  // BM, BK, BN
                         << -1 << "," << -1 << "," << -1 << ","  // WM, WK, WN
                         << -1 << "," << -1 << "," << -1 << ","  // MM, MK, MN
                         << -1 << "," << -1 << ","               // K_PIPE_MAX, USE_SYNC
                         << -1 << "," << -1 << ","               // median_ms, min_ms
                         << -1 << "," << -1 << ","               // max_ms, tflops
                         << -1 << ","                             // block_size
                         << config.target_time_ms << ","
                         << config.min_run_time_ms << "\n";
            results_file.flush();
        }
        results_file.close();
        return 0;
    }
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " <mode> <job_id> <config_idx> <M> <K> <N> <alpha> <beta>\n";
        return 1;
    }

    srand(time(nullptr) ^ getpid());
    return minfer::tuning::tune_run(argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), std::stof(argv[7]), std::stof(argv[8]));
}
