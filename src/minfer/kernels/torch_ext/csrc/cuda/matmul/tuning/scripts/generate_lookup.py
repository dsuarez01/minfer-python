import numpy as np
import pandas as pd
import glob
import os

def main():
    csv_files = glob.glob(os.path.join("./logs", "tuning_results_job*.csv"))

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    df['tflops'] = (2*df['M']*df['K']*df['N'])/(df['median_ms']*1e-3)*1e-12
    df['tflops/watt'] = df['tflops']/df['median_power_w']

    best_configs_noeff = df.loc[df.groupby(['M', 'K', 'N'])['tflops'].idxmax()]
    best_configs_eff = df.loc[df.groupby(['M', 'K', 'N'])['tflops/watt'].idxmax()]

    selected_idxs = np.where((best_configs_eff['tflops']/best_configs_noeff['tflops']) >= 0.8, best_configs_eff.index, best_configs_noeff.index)

    # selection logic: use the config maxxing tflops/watt unless it causes >20% drop in throughput
    best_df = df.loc[selected_idxs].sort_values(['M', 'K', 'N']).reset_index(drop=True)
    
    with open("../lookup.cuh", 'w') as f:
        f.write("#pragma once\n\n")
        f.write("#include <cfloat>\n")
        f.write("#include <cstddef>\n")
        f.write("#include <stdexcept>\n\n")
        f.write("namespace minfer::impl {\n\n")
        
        f.write("struct LookupEntry {\n")
        f.write("\tsize_t M, K, N;\n")
        f.write("\tunsigned int BM, BK, BN;\n")
        f.write("\tunsigned int WM, WK, WN;\n")
        f.write("\tunsigned int MM, MK, MN;\n")
        f.write("\tunsigned int K_PIPE_MAX;\n")
        f.write("\tunsigned int USE_SYNC;\n")
        f.write("\tfloat tflops;\n")
        f.write("};\n\n")
        
        f.write(f"constexpr size_t LOOKUP_TABLE_SIZE = {len(best_df)};\n\n")
        f.write("constexpr LookupEntry LOOKUP_TABLE[] = {\n")
        
        for i, (idx, row) in enumerate(best_df.iterrows()):
            f.write(f"\t{{{int(row['M'])}, {int(row['K'])}, {int(row['N'])}, ")
            f.write(f"{int(row['BM'])}, {int(row['BK'])}, {int(row['BN'])}, ")
            f.write(f"{int(row['WM'])}, {int(row['WK'])}, {int(row['WN'])}, ")
            f.write(f"{int(row['MM'])}, {int(row['MK'])}, {int(row['MN'])}, ")
            f.write(f"{int(row['K_PIPE_MAX'])}, {int(row['USE_SYNC'])}, {row['tflops']:.2f}f}}")
            
            if i < len(best_df)-1:
                f.write(",")
            f.write(f"  // idx={i}\n")
        
        f.write("};\n\n")
        
        f.write("inline size_t find_nearest_config(size_t M, size_t K, size_t N) {\n")
        f.write("\tsize_t best_idx = 0;\n")
        f.write("\tdouble best_dist = DBL_MAX;\n")
        f.write("\t\n")
        f.write("\tfor (size_t i = 0; i < LOOKUP_TABLE_SIZE; ++i) {\n")
        f.write("\t\tconst auto& entry = LOOKUP_TABLE[i];\n")
        f.write("\t\t\n")
        f.write("\t\tif (entry.M > M || entry.K > K || entry.N > N) continue;\n")
        f.write("\t\t\n")
        f.write("\t\tdouble dm = (double)(M > entry.M ? M - entry.M : entry.M - M) / (double)entry.M;\n")
        f.write("\t\tdouble dk = (double)(K > entry.K ? K - entry.K : entry.K - K) / (double)entry.K;\n")
        f.write("\t\tdouble dn = (double)(N > entry.N ? N - entry.N : entry.N - N) / (double)entry.N;\n")
        f.write("\t\tdouble dist = dm*dm + dk*dk + dn*dn;\n")
        f.write("\t\t\n")
        f.write("\t\tif (dist < best_dist) {\n")
        f.write("\t\t\tbest_dist = dist;\n")
        f.write("\t\t\tbest_idx = i;\n")
        f.write("\t\t}\n")
        f.write("\t}\n")
        f.write("\t\n")
        f.write("\tif (best_dist == DBL_MAX) {\n")
        f.write("\t\tthrow std::runtime_error(\"No valid kernel config found: problem size too small\");\n")
        f.write("\t}\n")
        f.write("\t\n")
        f.write("\treturn best_idx;\n")
        f.write("}\n\n")

        f.write("#define LOOKUP_INDEX_CASES \\\n")
        for i in range(len(best_df)):
            f.write(f"\tX({i}) \\\n")
        f.write("\n")

        f.write("}\n")


if __name__ == "__main__":
    main()