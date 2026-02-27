import pandas as pd
import glob
import os

def main():
    csv_files = glob.glob(os.path.join("./logs", "cold_tuning_results_job*.csv"))
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df['tflops'] = (2*df['M']*df['K']*df['N']+2*df['M']*df['N'])/(df['median_ms']*1e-3)*1e-12
    best_df = (df.sort_values(['M', 'K', 'N', 'alpha', 'beta', 'tflops'], ascending=[True, True, True, True, True, False])
                 .drop_duplicates(subset=['M', 'K', 'N', 'alpha', 'beta'])
                 .reset_index(drop=True))

    with open("../lookup.cuh", 'w') as f:
        f.write(f"""#pragma once

#include <cfloat>
#include <cstddef>
#include <stdexcept>

namespace minfer::impl {{

struct LookupEntry {{
    size_t M, K, N;
    float alpha, beta;
    unsigned int BM, BK, BN;
    unsigned int WM, WK, WN;
    unsigned int MM, MK, MN;
    unsigned int K_PIPE_MAX;
    unsigned int USE_SYNC;
    float tflops;
}};

constexpr size_t LOOKUP_TABLE_SIZE = {len(best_df)};
constexpr LookupEntry LOOKUP_TABLE[] = {{
""")

        for i, (idx, row) in enumerate(best_df.iterrows()):
            comma = "," if i < len(best_df) - 1 else ""
            f.write(f"\t{{{int(row['M'])}, {int(row['K'])}, {int(row['N'])}, "
                    f"{row['alpha']:.6f}f, {row['beta']:.6f}f, "
                    f"{int(row['BM'])}, {int(row['BK'])}, {int(row['BN'])}, "
                    f"{int(row['WM'])}, {int(row['WK'])}, {int(row['WN'])}, "
                    f"{int(row['MM'])}, {int(row['MK'])}, {int(row['MN'])}, "
                    f"{int(row['K_PIPE_MAX'])}, {int(row['USE_SYNC'])}, {row['tflops']:.2f}f}}{comma}  // idx={i}\n")

        macro_cases = "".join(f"\tX({i}) \\\n" for i in range(len(best_df)))

        f.write(f"""
}};

inline size_t find_nearest_config(size_t M, size_t K, size_t N) {{
    size_t best_idx = 0;
    double best_dist = DBL_MAX;

    for (size_t i = 0; i < LOOKUP_TABLE_SIZE; ++i) {{
        const auto& entry = LOOKUP_TABLE[i];

        if (entry.M > M || entry.K > K || entry.N > N) continue;

        double dm = (double)(M > entry.M ? M - entry.M : entry.M - M) / (double)entry.M;
        double dk = (double)(K > entry.K ? K - entry.K : entry.K - K) / (double)entry.K;
        double dn = (double)(N > entry.N ? N - entry.N : entry.N - N) / (double)entry.N;
        double dist = dm*dm + dk*dk + dn*dn;

        if (dist < best_dist) {{
            best_dist = dist;
            best_idx = i;
        }}
    }}

    if (best_dist == DBL_MAX) {{
        throw std::runtime_error("No valid kernel config found: problem size too small");
    }}

    return best_idx;
}}

#define LOOKUP_INDEX_CASES \\
{macro_cases}

}}
""")

if __name__ == "__main__":
    main()