#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <nvml.h>

#include <thread>
#include <atomic>
#include <array>
#include <vector>
#include <exception>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <filesystem>

#include "tune.cuh"
#include "kernels/xw.cuh"

inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
#define checkCudaErrors(call) checkCuda(call, __FILE__, __LINE__)

inline void checkNVML(nvmlReturn_t ret, const char* file, int line) {
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "NVML error at %s:%d: %s\n", file, line, nvmlErrorString(ret));
        throw std::runtime_error(nvmlErrorString(ret));
    }
}
#define checkNVMLErrors(call) checkNVML(call, __FILE__, __LINE__)

namespace minfer::tuning {

    using namespace minfer::impl;

    std::atomic<bool> sampling{false};
    std::vector<unsigned int> power_samples;

    void sample_power(nvmlDevice_t device, int sleep_us) {
        while (sampling) {
            unsigned int power_mw;
            // silently fails, but if taking enough measurements this shouldn't be an issue
            if (nvmlDeviceGetPowerUsage(device, &power_mw) == NVML_SUCCESS) {
                power_samples.push_back(power_mw);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        }
    }

    void flush_l2_cache(int l2_size) {
        if (l2_size > 0) {
            void* buffer;
            checkCudaErrors(cudaMalloc(&buffer, l2_size));
            checkCudaErrors(cudaMemsetAsync(buffer, 0, l2_size));
            checkCudaErrors(cudaFree(buffer));
        }
    }

    template<size_t IDX>
    void launch_kernel_impl(int M, int K, int N, const half* d_x, const half* d_w, half* d_out) {
        constexpr auto& kc = ALL_CONFIGS[IDX];
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
                    xw_sync_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, NUM_THRS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SHMEM_SZ
                )
            );
            
            xw_sync_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, NUM_THRS>
                <<<grid, block, SHMEM_SZ>>>(M, K, N, d_x, d_w, d_out);
        } else {

            checkCudaErrors(
                cudaFuncSetAttribute(
                    xw_async_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, K_PIPE_MAX, NUM_THRS>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    SHMEM_SZ
                )
            );
            
            xw_async_impl<BM, BK, BN, WM, WK, WN, MM, MK, MN, TILES_K, K_PIPE_MAX, NUM_THRS>
                <<<grid, block, SHMEM_SZ>>>(M, K, N, d_x, d_w, d_out);
        }
        
        checkCudaErrors(cudaGetLastError());
    }

    void launch_kernel(size_t idx, int M, int K, int N, const half* d_x, const half* d_w, half* d_out) {
        switch(idx) {
#define X(IDX) case IDX: return launch_kernel_impl<IDX>(M, K, N, d_x, d_w, d_out);
KERNEL_CONFIG_INDICES
#undef X
            default: throw std::runtime_error("Invalid config idx");
        }
    }

    Result run_benchmark(const Config& config) {
        auto M = config.M, K = config.K, N = config.N;

        half *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
        cudaEvent_t start = nullptr, stop = nullptr;

        try {
            checkCudaErrors(cudaEventCreate(&start));
            checkCudaErrors(cudaEventCreate(&stop));
        
            int dev_id, l2_cache_size;
            checkCudaErrors(cudaGetDevice(&dev_id));
            checkCudaErrors(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, dev_id));

            checkCudaErrors(cudaMalloc(&d_x, M*K*sizeof(half)));
            checkCudaErrors(cudaMalloc(&d_w, K*N*sizeof(half)));
            checkCudaErrors(cudaMalloc(&d_out, M*N*sizeof(half)));
            
            // estimate num iters for benchmark
            int est_iters = 10;
            float est_ms;
            checkCudaErrors(cudaEventRecord(start));
            for (int i = 0; i < est_iters; ++i) {
                launch_kernel(config.config_idx, M, K, N, d_x, d_w, d_out);
                flush_l2_cache(l2_cache_size);
            }
            checkCudaErrors(cudaEventRecord(stop));
            checkCudaErrors(cudaEventSynchronize(stop));
            checkCudaErrors(cudaEventElapsedTime(&est_ms, start, stop));
            int total_iters = std::max(10, (int)(config.target_time_ms / (est_ms/est_iters)));
            int warmup_iters = std::min(5, total_iters/10);

            // warm-up
            for (int i = 0; i < warmup_iters; ++i) {
                launch_kernel(config.config_idx, M, K, N, d_x, d_w, d_out);
            }
            checkCudaErrors(cudaDeviceSynchronize());
            flush_l2_cache(l2_cache_size);
            
            // main benchmark
            
            // (init power sampling)
            int sample_intvl_us = (int)((est_ms/est_iters*1000)/100); // 100 samples by default
            nvmlDevice_t nvml_device;
            checkNVMLErrors(nvmlDeviceGetHandleByIndex(dev_id, &nvml_device));
            unsigned int power_lim_mw;
            checkNVMLErrors(nvmlDeviceGetPowerManagementLimit(nvml_device, &power_lim_mw));
            float power_lim_watts = power_lim_mw / 1000.0f;
            power_samples.clear();
            power_samples.reserve(total_iters * 100);
            sampling = true;
            std::thread sampler(sample_power, nvml_device, sample_intvl_us);

            std::vector<double> times;
            for (int i = 0; i < total_iters; ++i) {
                checkCudaErrors(cudaEventRecord(start));
                launch_kernel(config.config_idx, M, K, N, d_x, d_w, d_out);
                checkCudaErrors(cudaEventRecord(stop));
                checkCudaErrors(cudaEventSynchronize(stop));
                
                float ms;
                checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
                times.push_back(ms);
                
                flush_l2_cache(l2_cache_size);
            }
            
            sampling = false;
            sampler.join();

            // compute timing/throughput stats
            std::sort(times.begin(), times.end());
            double median_ms = times[times.size()/2];
            double min_ms = times.front();
            double max_ms = times.back();

            double flops = 2.0*M*K*N;
            float tflops = (flops / (median_ms * 1e-3)) / 1e12;

            // compute power stats
            unsigned int sum_power_mw = 0u, max_power_mw = 0u;
            for (auto p : power_samples) {
                sum_power_mw += p;
                max_power_mw = std::max(max_power_mw, p);
            }
            float mean_power_watts = (sum_power_mw / 1000.0f) / power_samples.size();
            std::sort(power_samples.begin(), power_samples.end());
            float median_power_watts = power_samples[power_samples.size()/2] / 1000.0f;
            float max_power_watts = max_power_mw / 1000.0f;

            // clean-up
            checkCudaErrors(cudaFree(d_x));
            checkCudaErrors(cudaFree(d_w));
            checkCudaErrors(cudaFree(d_out));
            checkCudaErrors(cudaEventDestroy(start));
            checkCudaErrors(cudaEventDestroy(stop));
            
            return Result{
                config, 
                warmup_iters, total_iters, 
                median_ms, min_ms, max_ms,
                mean_power_watts, median_power_watts, max_power_watts,
                power_lim_watts

            };
        } catch(...) {
            // cleanup on error
            if (d_x) cudaFree(d_x);
            if (d_w) cudaFree(d_w);
            if (d_out) cudaFree(d_out);
            if (start) cudaEventDestroy(start);
            if (stop) cudaEventDestroy(stop);
            throw;
        }
    }
}

int main(int argc, char** argv) {

    using namespace minfer::tuning;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <job_id> <num_jobs>\n";
        return 1;
    }

    int job_id = atoi(argv[1]);
    int num_jobs = atoi(argv[2]);

    checkNVMLErrors(nvmlInit());

    // generate all of the problem sizes we want to test on (will do this in a second)
    std::vector<std::array<int, 3>> problems = {};
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192, 16384, 32768};

    for (int m : sizes) {
        for (int k : sizes) {
            for (int n : sizes) {
                problems.push_back({m, k, n});
            }
        }
    }

    double target_time_ms = 200.0; // how long we want each case to run in total

    std::filesystem::create_directories("./logs");
    std::ofstream results_file("./logs/tuning_results_job" + std::to_string(job_id) + ".csv");
    if (!results_file.is_open()) {
        std::cerr << "ERROR: Could not open ./logs/tuning_results_job" << job_id << ".csv\n";
        return 1;
    }
    results_file << "M,K,N,config_idx,BM,BK,BN,WM,WK,WN,MM,MK,MN,K_PIPE_MAX,USE_SYNC,"
                 << "median_ms,min_ms,max_ms,mean_power_w,median_power_w,max_power_w,power_limit_w,"
                 << "warmup_iters,iters\n";

    for (size_t i = job_id; i < problems.size(); i += num_jobs) {
        auto [M, K, N] = problems[i];
        std::cout << "Benchmarking M=" << M << " K=" << K << " N=" << N << std::endl;
        
        float best_tflops = 0.0f;
        for (size_t i = 0; i < NUM_CONFIGS; ++i) {
            auto config = Config{i, ALL_CONFIGS[i], M, K, N, target_time_ms};
            try {
                auto result = run_benchmark(config);
                
                // Write immediately
                auto& kc = result.config.kc;
                results_file << result.config.M << "," << result.config.K << "," << result.config.N << ","
                             << result.config.config_idx << ","
                             << kc.BM << "," << kc.BK << "," << kc.BN << ","
                             << kc.WM << "," << kc.WK << "," << kc.WN << ","
                             << kc.MM << "," << kc.MK << "," << kc.MN << ","
                             << kc.K_PIPE_MAX << "," << kc.USE_SYNC << ","
                             << result.median_time_ms << "," << result.min_time_ms << "," << result.max_time_ms << ","
                             << result.mean_power_watts << "," << result.median_power_watts << "," 
                             << result.max_power_watts << "," << result.power_limit_watts << ","
                             << result.warmup_iters << "," << result.iters << "\n";
                results_file.flush();
                
                double flops = 2.0 * result.config.M * result.config.K * result.config.N;
                float tflops = (flops / (result.median_time_ms * 1e-3)) / 1e12;
                best_tflops = std::max(best_tflops, tflops);
                if (i % 100 == 0) {
                    std::cout << "\tConfig " << i << "/" << NUM_CONFIGS 
                            << " - Current: " << tflops 
                            << " TFLOPS, Best: " << best_tflops << " TFLOPS" << std::endl;
                }
            } catch(const std::exception& e) {
                std::cerr << "\tConfig " << i << " failed: " << e.what() << std::endl;
            } catch(...) {
                std::cerr << "\tConfig " << i << " failed w/ unknown err" << std::endl;
            }
        }
        std::cout << "\tFinal best for this problem: " << best_tflops << " TFLOPS\n" << std::endl;
    }

    results_file.close();
    checkNVMLErrors(nvmlShutdown());

    return 0;
}