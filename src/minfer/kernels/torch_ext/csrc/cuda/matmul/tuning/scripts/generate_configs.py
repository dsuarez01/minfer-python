# pyright: reportOperatorIssue=false

import itertools

from z3 import Solver, Ints, Int, sat, Or, Int2BV, Implies
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA not enabled, or no GPU detected (need for device properties)"

def main():

    props = torch.cuda.get_device_properties(0)
    max_threads_per_block = props.max_threads_per_block
    max_shmem_per_block = props.shared_memory_per_block_optin
    max_regs_per_thread = 255 if props.major >= 2 else 63
    min_regs_per_thread = 180 # for pruning

    BM, BK, BN = Ints("BM BK BN")
    WM, WK, WN = Ints("WM WK WN")
    MM, MK, MN = Ints("MM MK MN")
    K_PIPE_MAX = Int("K_PIPE_MAX")
    USE_SYNC = Int("USE_SYNC")

    s = Solver()

    # range constrs.
    s.add(BM >= WM, WM >= MM, MM > 0)
    s.add(BK >= WK, WK >= MK, MK > 0)
    s.add(BN >= WN, WN >= MN, MN > 0)
    s.add(K_PIPE_MAX >= 2)
    s.add(USE_SYNC >= 0, USE_SYNC <= 1)

    # mma divides warp divides block tiles
    s.add(BM % WM == 0, BK % WK == 0, BN % WN == 0)
    s.add(WM % MM == 0, WK % MK == 0, WN % MN == 0)
    
    # based on supported instr. types for sm_89 (see PTX manual)
    s.add(MM == 16)
    s.add(Or([MK == 8, MK == 16]))
    s.add(MN == 8)

    # powers of 2
    s.add(Int2BV(BM,16)&(Int2BV(BM,16)-1) == 0, BM > 0)
    s.add(Int2BV(BK,16)&(Int2BV(BK,16)-1) == 0, BK > 0)
    s.add(Int2BV(BN,16)&(Int2BV(BN,16)-1) == 0, BN > 0)

    # last correctness constrs.
    s.add(Implies(USE_SYNC == 1, K_PIPE_MAX == 2)) # use_sync only supports size-2 pipeline
    s.add((BM/WM)*(BN/WN)*32 <= max_threads_per_block)
    s.add(K_PIPE_MAX*(BM*BK+BK*BN)*2 <= max_shmem_per_block)
    s.add(BM*BN*2 <= K_PIPE_MAX*(BM*BK+BK*BN)*2) # reuse shmem for output tile
    s.add(BM*(BK/8) >= (BM/WM)*(BN/WN)*32) # see toShmem fcns
    s.add(BK*(BN/8) >= (BM/WM)*(BN/WN)*32) # see toShmem fcns
    s.add( # (register constrs.)
        Implies(
            USE_SYNC == 1, 
            (BK/WK)*((WM/MM)*(WK/MK)*(MM/8)*(MK/8)+(WK/MK)*(WN/MN)*(MK/8)*(MN/8)) 
            + (WM/MM)*(WN/MN)*(MM/8)*(MN/8)
            + 4*(BM/(((BM/WM)*(BN/WN)*32)/(BK/8))) 
            + 4*(BK/(((BM/WM)*(BN/WN)*32)/(BN/8))) 
            <= max_regs_per_thread
        )
    )
    s.add( # (register constrs.)
        Implies(
            USE_SYNC == 0, 
            (BK/WK)*((WM/MM)*(WK/MK)*(MM/8)*(MK/8)+(WK/MK)*(WN/MN)*(MK/8)*(MN/8)) 
            + (WM/MM)*(WN/MN)*(MM/8)*(MN/8) 
            <= max_regs_per_thread
        )
    )

    s.add(
        Implies(
            USE_SYNC == 1,
            (BK/WK)*((WM/MM)*(WK/MK)*(MM/8)*(MK/8)+(WK/MK)*(WN/MN)*(MK/8)*(MN/8)) 
            + (WM/MM)*(WN/MN)*(MM/8)*(MN/8)
            + 4*(BM/(((BM/WM)*(BN/WN)*32)/(BK/8))) 
            + 4*(BK/(((BM/WM)*(BN/WN)*32)/(BN/8))) 
            >= min_regs_per_thread
        )
    )
    s.add(
        Implies(
            USE_SYNC == 0,
            (BK/WK)*((WM/MM)*(WK/MK)*(MM/8)*(MK/8)+(WK/MK)*(WN/MN)*(MK/8)*(MN/8)) 
            + (WM/MM)*(WN/MN)*(MM/8)*(MN/8)
            >= min_regs_per_thread
        )
    )

    # pruning
    s.add(Or([BM == 32, BM == 64, BM == 128, BM == 256, BM == 512]))
    s.add(Or([BK == 32, BK == 64, BK == 128]))
    s.add(Or([BN == 32, BN == 64, BN == 128, BN == 256, BN == 512]))
    s.add(BM >= BK, BN >= BK) # arithmetic intensity prop. to BM*BN/(BM+BN)
    s.add(K_PIPE_MAX <= 4)
    s.add(BN/WN >= 2) # more parallelism along N dimension due to dispatch
    s.add((BM/WM)*(BN/WN)*32>=128) # solns w <= 128 threads likely subopt

    configs = []

    print("Searching for valid configs")
    while s.check() == sat:
        m = s.model()
        config = (
            m[BM].as_long(), m[BK].as_long(), m[BN].as_long(), # type: ignore
            m[WM].as_long(), m[WK].as_long(), m[WN].as_long(), # type: ignore
            m[MM].as_long(), m[MK].as_long(), m[MN].as_long(), # type: ignore
            m[K_PIPE_MAX].as_long(), m[USE_SYNC].as_long()     # type: ignore
        )
        configs.append(config)
        
        # forces unique solns
        s.add(
            Or(
                BM != m[BM], BK != m[BK], BN != m[BN],
                WM != m[WM], WK != m[WK], WN != m[WN],
                MM != m[MM], MK != m[MK], MN != m[MN],
                K_PIPE_MAX != m[K_PIPE_MAX], USE_SYNC != m[USE_SYNC]
            )
        )
        
        if len(configs) % 100 == 0:
            print(f"Found {len(configs)} configs so far")

    print(f"\nTotal valid configs: {len(configs)}")

    with open("./tune.cuh", "w") as f:
        
        # write ALL_CONFIGS array
        f.write("#pragma once\n\n")
        f.write("#include <cstddef>\n\n")
        f.write("namespace minfer::tuning {\n\n")
        
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
        f.write("\tint M, K, N;\n")
        f.write("\tdouble target_time_ms;\n")
        f.write("};\n\n")
        
        f.write("struct Result {\n")
        f.write("\tConfig config;\n")
        f.write("\tint warmup_iters;\n")
        f.write("\tint iters;\n")
        f.write("\tdouble median_time_ms;\n")
        f.write("\tdouble min_time_ms;\n")
        f.write("\tdouble max_time_ms;\n")
        f.write("\tfloat tflops;\n")
        f.write("};\n\n")
        
        f.write(f"constexpr size_t NUM_CONFIGS = {len(configs)};\n\n")
        
        f.write("constexpr KernelConfig ALL_CONFIGS[NUM_CONFIGS] = {\n")
        for cfg in configs:
            bm, bk, bn, wm, wk, wn, mm, mk, mn, kpm, sync = cfg
            f.write(f"\t{{\"float16\", {bm}, {bk}, {bn}, {wm}, {wk}, {wn}, {mm}, {mk}, {mn}, {kpm}, {sync}}},\n")
        f.write("};\n\n")
        
        # write X macro
        f.write("#define KERNEL_CONFIG_INDICES \\\n")
        for i in range(len(configs)):
            f.write(f"\tX({i}) \\\n")
        f.write("\n")

        f.write("}\n") # end namespace

if __name__ == "__main__":
    main()