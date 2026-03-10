#pragma once

#include <cstddef>
#include <stdexcept>

namespace minfer::impl {

struct LookupEntry {
    size_t M, K, N;
    bool is_alpha_1, is_beta_0;
    unsigned int BM, BK, BN;
    unsigned int WM, WK, WN;
    unsigned int MM, MK, MN;
    unsigned int K_PIPE_MAX;
    unsigned int USE_SYNC;
    float tflops;
};

constexpr size_t LOOKUP_TABLE_SIZE = 24;
constexpr LookupEntry LOOKUP_TABLE[] = {
	{512, 512, 512, true, true, 128, 32, 128, 64, 16, 32, 16, 16, 8, 4, 0, 28.87f},  // idx=0
	{512, 512, 512, true, false, 128, 32, 128, 32, 16, 64, 16, 16, 8, 4, 0, 26.66f},  // idx=1
	{512, 512, 512, false, true, 128, 32, 128, 64, 16, 32, 16, 16, 8, 4, 0, 28.94f},  // idx=2
	{512, 512, 512, false, false, 128, 32, 128, 32, 16, 64, 16, 16, 8, 4, 0, 26.60f},  // idx=3
	{1024, 1024, 1024, true, true, 128, 32, 128, 64, 16, 32, 16, 16, 8, 4, 0, 130.77f},  // idx=4
	{1024, 1024, 1024, true, false, 128, 32, 128, 64, 16, 32, 16, 16, 8, 4, 0, 124.76f},  // idx=5
	{1024, 1024, 1024, false, true, 128, 32, 128, 64, 16, 32, 16, 16, 8, 4, 0, 130.63f},  // idx=6
	{1024, 1024, 1024, false, false, 128, 32, 128, 64, 16, 32, 16, 16, 8, 4, 0, 124.64f},  // idx=7
	{2048, 2048, 2048, true, true, 128, 32, 256, 64, 16, 64, 16, 16, 8, 4, 0, 240.80f},  // idx=8
	{2048, 2048, 2048, true, false, 128, 32, 256, 64, 16, 64, 16, 16, 8, 4, 0, 231.06f},  // idx=9
	{2048, 2048, 2048, false, true, 128, 32, 256, 64, 16, 64, 16, 16, 8, 4, 0, 240.50f},  // idx=10
	{2048, 2048, 2048, false, false, 128, 32, 256, 64, 16, 64, 16, 16, 8, 4, 0, 234.43f},  // idx=11
	{4096, 4096, 4096, true, true, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 230.47f},  // idx=12
	{4096, 4096, 4096, true, false, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 212.20f},  // idx=13
	{4096, 4096, 4096, false, true, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 224.66f},  // idx=14
	{4096, 4096, 4096, false, false, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 207.63f},  // idx=15
	{8192, 8192, 8192, true, true, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 194.94f},  // idx=16
	{8192, 8192, 8192, true, false, 128, 64, 256, 64, 64, 64, 16, 16, 8, 2, 0, 190.09f},  // idx=17
	{8192, 8192, 8192, false, true, 128, 64, 256, 64, 64, 64, 16, 16, 8, 2, 0, 198.21f},  // idx=18
	{8192, 8192, 8192, false, false, 128, 64, 256, 64, 64, 64, 16, 16, 8, 2, 0, 194.76f},  // idx=19
	{16384, 16384, 16384, true, true, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 196.95f},  // idx=20
	{16384, 16384, 16384, true, false, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 197.63f},  // idx=21
	{16384, 16384, 16384, false, true, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 194.41f},  // idx=22
	{16384, 16384, 16384, false, false, 128, 64, 256, 64, 32, 64, 16, 16, 8, 2, 0, 195.14f}  // idx=23

};

// hardcoded problem sizes for now, will refine dispatch soon
inline int find_config(size_t M, size_t K, size_t N, float alpha, float beta) {
    bool is_alpha_1 = (alpha == 1.0f);
    bool is_beta_0 = (beta == 0.0f);
    for (int i = 0; i < LOOKUP_TABLE_SIZE; ++i) {
        const auto& entry = LOOKUP_TABLE[i];
        if (entry.M == M && entry.K == K && entry.N == N &&
            entry.is_alpha_1 == is_alpha_1 && entry.is_beta_0 == is_beta_0) {
            return i;
        }
    }
    return -1;
}

#define LOOKUP_INDEX_CASES \
	X(0) \
	X(1) \
	X(2) \
	X(3) \
	X(4) \
	X(5) \
	X(6) \
	X(7) \
	X(8) \
	X(9) \
	X(10) \
	X(11) \
	X(12) \
	X(13) \
	X(14) \
	X(15) \
	X(16) \
	X(17) \
	X(18) \
	X(19) \
	X(20) \
	X(21) \
	X(22) \
	X(23) \


}
