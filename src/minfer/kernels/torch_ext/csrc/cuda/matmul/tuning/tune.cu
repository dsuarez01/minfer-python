// NOTE: this is a standalone compile, generates massive binary

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <vector>
#include <exception>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <map>

#include "tune.cuh"

#define checkCudaErrors(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA err at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

namespace minfer::tuning {
    
    using namespace minfer::tuning;

    void flush_l2_cache(int l2_size) {
        if (l2_size > 0) {
            void* buffer;
            checkCudaErrors(cudaMalloc(&buffer, l2_size));
            checkCudaErrors(cudaMemset(buffer, 0, l2_size));
            checkCudaErrors(cudaFree(buffer));
        }
    }

    Result run_benchmark(const Config& config) {
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
        
        int dev_id, l2_cache_size;
        checkCudaErrors(cudaGetDevice(&dev_id));
        checkCudaErrors(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, dev_id));
        
        auto M = config.M, K = config.K, N = config.N;

        half *d_x, *d_w, *d_out;
        checkCudaErrors(cudaMalloc(&d_x, M*K*sizeof(half)));
        checkCudaErrors(cudaMalloc(&d_w, K*N*sizeof(half)));
        checkCudaErrors(cudaMalloc(&d_out, M*N*sizeof(half)));
        
        // estimate num iters for benchmark
        int est_iters = 10;
        checkCudaErrors(cudaEventRecord(start));
        for (int i = 0; i < est_iters; ++i) {
            // TODO: launch kernel here
            flush_l2_cache(l2_cache_size);
        }
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        
        double est_ms;
        checkCudaErrors(cudaEventElapsedTime(&est_ms, start, stop));
        int total_iters = std::max(10, (int)(config.target_time_ms / (est_ms/est_iters)));
        int warmup_iters = std::min(5, total_iters/10);
        
        // warm-up
        for (int i = 0; i < warmup_iters; ++i) {
            // TODO: launch kernel here
        }
        checkCudaErrors(cudaDeviceSynchronize());
        flush_l2_cache(l2_cache_size);
        
        // main benchmark
        std::vector<double> times;
        for (int i = 0; i < total_iters; ++i) {
            checkCudaErrors(cudaEventRecord(start));
            // TODO: launch kernel here
            checkCudaErrors(cudaEventRecord(stop));
            checkCudaErrors(cudaEventSynchronize(stop));
            
            double ms;
            checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
            times.push_back(ms);
            
            flush_l2_cache(l2_cache_size);
        }
        
        // compute stats
        std::sort(times.begin(), times.end());
        double median_ms = times[times.size() / 2];
        double min_ms = times.front();
        double max_ms = times.back();

        double flops = 2.0*M*K*N;
        float tflops = (flops / (median_ms * 1e-3)) / 1e12;

        // clean-up
        checkCudaErrors(cudaFree(d_x));
        checkCudaErrors(cudaFree(d_w));
        checkCudaErrors(cudaFree(d_out));
        checkCudaErrors(cudaEventDestroy(start));
        checkCudaErrors(cudaEventDestroy(stop));
        
        return Result{config, warmup_iters, total_iters, median_ms, min_ms, max_ms, tflops};
    }
}

int main(int argc, char** argv) {

    using namespace minfer::tuning;

    // argc handling

    // generate all of the problem sizes we want to test on (will do this in a second)
    std::vector<std::array<int, 3>> problems = {};
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};

    for (int m : sizes) {
        for (int k : sizes) {
            for (int n : sizes) {
                problems.push_back({m, k, n});
            }
        }
    }
    
    double target_time_ms = 200.0; // how long we want each case to run in total

    std::vector<Result> results = {};

    for (const auto& [M, K, N] : problems) {
        std::cout << "Benchmarking M=" << M << " K=" << K << " N=" << N << std::endl;
        
        for (size_t i = 0; i < NUM_CONFIGS; ++i) {
            auto config = Config{ALL_CONFIGS[i], M, K, N, target_time_ms};
            try {
                auto result = run_benchmark(config);
                results.push_back(result);
                if (i % 100 == 0) {
                    std::cout << "\tConfig " << i << "/" << NUM_CONFIGS << " - " << result.tflops << " TFLOPS" << std::endl;
                }
            } catch(const std::exception& e) {
                std::cerr << "\tConfig " << i << " failed: " << e.what() << std::endl;
            } catch(...) {
                std::cerr << "\tConfig " << i << " failed w/ unknown err" << std::endl;
            }
        }
    }

    // find best-performing config for each problem size
    std::map<std::array<int, 3>, size_t> best_configs;
    for (const auto& result : results) {
        auto key = std::array{result.config.M, result.config.K, result.config.N};
        
        if (best_configs.find(key) == best_configs.end() || result.tflops > results[best_configs[key]].tflops) {
            auto idx = &result - &results[0];
            best_configs[key] = idx;
        }
    }

    // write to ./lookup.cuh (CWD from which this will be run is cuda/matmul, not tuning)
    std::ofstream out("./lookup.cuh");
    out << "#pragma once\n\n";
    out << "namespace minfer::config {\n\n";
    out << "struct TunedEntry {\n";
    out << "\tsize_t M, K, N;\n";
    out << "\tunsigned int BM, BK, BN;\n";
    out << "\tunsigned int WM, WK, WN;\n";
    out << "\tunsigned int MM, MK, MN;\n";
    out << "\tunsigned int K_PIPE;\n";
    out << "\tunsigned int USE_SYNC;\n";
    out << "};\n\n";
    out << "constexpr TunedEntry TUNED_LOOKUP[] = {\n";

    for (const auto& [problem, result_idx] : best_configs) {
        auto [M, K, N] = problem;
        auto& kc = results[result_idx].config.kc;
        out << "\t{" << M << ", " << K << ", " << N << ", "
            << kc.BM << ", " << kc.BK << ", " << kc.BN << ", "
            << kc.WM << ", " << kc.WK << ", " << kc.WN << ", "
            << kc.MM << ", " << kc.MK << ", " << kc.MN << ", "
            << kc.K_PIPE << ", " << kc.USE_SYNC << "},\n";
    }

    out << "};\n\n";
    out << "}\n";

    return 0;
}