# pyright: reportOperatorIssue=false

import textwrap

from z3 import Solver, Ints, Int, sat, Or, And, Int2BV, Implies
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA not enabled, or no GPU detected (need for device properties)"

def main():

    props = torch.cuda.get_device_properties(0)
    max_threads_per_block = props.max_threads_per_block
    max_shmem_per_block = props.shared_memory_per_block_optin
    max_regs_per_sm = props.regs_per_multiprocessor
    
    # estimated overhead in addition to register usage below...
    sync_overhead = 30
    async_overhead = 40

    BM, BK, BN = Ints("BM BK BN")
    WM, WK, WN = Ints("WM WK WN")
    MM, MK, MN = Ints("MM MK MN")
    K_PIPE_MAX = Int("K_PIPE_MAX")
    USE_SYNC = Int("USE_SYNC")
    NUM_THRS = (BM/WM)*(BN/WN)*32

    s = Solver()

    # constraint on block, warp, mma tile sizes
    # mma tile size constraints based on supported instr. types for sm_89 (see PTX manual)
    s.add(
        BM == 128,
        Or([BK == 32, BK == 64]),
        Or([BN == 128, BN == 256]),
        WM >= 32,
        Or([WK == BK, WK == BK/2]),
        WN >= 32,
        MM == 16,
        MK == 16,
        MN == 8,
    )

    # mma divides warp divides block tile sizes
    s.add(
        BM % WM == 0, 
        BK % WK == 0, 
        BN % WN == 0,
        WM % MM == 0, 
        WK % MK == 0, 
        WN % MN == 0,
    )

    # constraints on K_PIPE_MAX, USE_SYNC
    s.add(
        Or([K_PIPE_MAX == 2, K_PIPE_MAX == 4]), # weird bug with K_PIPE_MAX = 3
        Or([USE_SYNC == 0, USE_SYNC == 1]),
        Implies(USE_SYNC == 1, K_PIPE_MAX == 2), # use_sync only supports size-2 pipeline
    )
    
    # (register constrs.)
    sync_regs = (
        (BK/WK)*((WM/MM)*(WK/MK)*(MM/8)*(MK/8)+(WK/MK)*(WN/MN)*(MK/8)*(MN/8)) 
        + (WM/MM)*(WN/MN)*(MM/8)*(MN/8)
        + 4*(BM/(NUM_THRS/(BK/8))) 
        + 4*(BK/(NUM_THRS/(BN/8)))
    )
    async_regs = (
        (BK/WK)*((WM/MM)*(WK/MK)*(MM/8)*(MK/8)+(WK/MK)*(WN/MN)*(MK/8)*(MN/8)) 
        + (WM/MM)*(WN/MN)*(MM/8)*(MN/8)
    )

    s.add(
        Implies(USE_SYNC == 1, NUM_THRS * (sync_regs + sync_overhead) <= max_regs_per_sm),
        Implies(USE_SYNC == 0, NUM_THRS * (async_regs + async_overhead) <= max_regs_per_sm),
    )

    # last (truly) necessary constraints
    s.add(
        NUM_THRS <= max_threads_per_block,
        K_PIPE_MAX*(BM*BK+BK*BN)*2 <= max_shmem_per_block,
        BM*BN*2 <= K_PIPE_MAX*(BM*BK+BK*BN)*2,  # reuse shmem for output tile
        BM*(BK/8) >= NUM_THRS, # see toShmem fcns
        BK*(BN/8) >= NUM_THRS, # see toShmem fcns

    )

    # pruning
    s.add(
        BN/WN >= 2, # more warps along N block
        NUM_THRS>=256, # solns with <= 256 threads likely subopt
    )

    kcs = []

    print("Searching for valid kernel configs")
    while s.check() == sat:
        m = s.model()
        kc = (
            m[BM].as_long(), m[BK].as_long(), m[BN].as_long(), # type: ignore
            m[WM].as_long(), m[WK].as_long(), m[WN].as_long(), # type: ignore
            m[MM].as_long(), m[MK].as_long(), m[MN].as_long(), # type: ignore
            m[K_PIPE_MAX].as_long(), m[USE_SYNC].as_long()     # type: ignore
        )
        kcs.append(kc)
        
        # forces unique solns
        s.add(
            Or(
                BM != m[BM], BK != m[BK], BN != m[BN],
                WM != m[WM], WK != m[WK], WN != m[WN],
                MM != m[MM], MK != m[MK], MN != m[MN],
                K_PIPE_MAX != m[K_PIPE_MAX], USE_SYNC != m[USE_SYNC]
            )
        )
        
        if len(kcs) % 100 == 0:
            print(f"Found {len(kcs)} kernel configs so far")

    print(f"\nTotal valid kernel configs: {len(kcs)}")

    with open("./tune.cuh", "w") as f:
        
        f.write("#pragma once\n\n")
        f.write("#include <cstddef>\n\n")
        f.write("namespace minfer::tuning {\n\n")
        
        # define structs for config and results
        f.write("struct KernelConfig {\n")
        f.write("\tconst char* dtype;\n")
        f.write("\tunsigned int BM, BK, BN;\n")
        f.write("\tunsigned int WM, WK, WN;\n")
        f.write("\tunsigned int MM, MK, MN;\n")
        f.write("\tunsigned int K_PIPE_MAX;\n")
        f.write("\tunsigned int USE_SYNC;\n")
        f.write("};\n\n")
        
        f.write("struct Config {\n")
        f.write("\tsize_t config_idx;\n")
        f.write("\tKernelConfig kc;\n")
        f.write("\tsize_t M, K, N;\n")
        f.write("\tfloat alpha, beta;\n")
        f.write("\tdouble target_time_ms;\n")
        f.write("\tdouble min_run_time_ms;\n")
        f.write("};\n\n")

        f.write("struct Result {\n")
        f.write("\tConfig config;\n")
        f.write("\tint block_size;\n")
        f.write("\tint total_iters;\n")
        f.write("\tdouble median_time_ms;\n")
        f.write("\tdouble min_time_ms;\n")
        f.write("\tdouble max_time_ms;\n")
        f.write("\tfloat tflops;\n")
        f.write("};\n\n")
        
        f.write(f"constexpr size_t NUM_KERNEL_CONFIGS = {len(kcs)};\n\n")
        
        # write ALL_CONFIGS array
        f.write("constexpr KernelConfig ALL_KERNEL_CONFIGS[NUM_KERNEL_CONFIGS] = {\n")
        for bm, bk, bn, wm, wk, wn, mm, mk, mn, kpm, sync in kcs:
            f.write(f"\t{{\"float16\", {bm}, {bk}, {bn}, {wm}, {wk}, {wn}, {mm}, {mk}, {mn}, {kpm}, {sync}}},\n")
        f.write("};\n\n")
        
        # write X macro
        f.write("#define KERNEL_CONFIG_INDICES \\\n")
        for i in range(len(kcs)):
            f.write(f"\tX({i}) \\\n")
        f.write("\n")

        f.write("}\n")

if __name__ == "__main__":
    main()