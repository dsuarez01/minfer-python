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

#include <cstddef>
#include <stdexcept>

namespace minfer::impl {{

struct LookupEntry {{
    size_t M, K, N;
    bool is_alpha_1, is_beta_0;
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
                    f"{'true' if row['alpha'] == 1 else 'false'}, {'true' if row['beta'] == 0 else 'false'}, "
                    f"{int(row['BM'])}, {int(row['BK'])}, {int(row['BN'])}, "
                    f"{int(row['WM'])}, {int(row['WK'])}, {int(row['WN'])}, "
                    f"{int(row['MM'])}, {int(row['MK'])}, {int(row['MN'])}, "
                    f"{int(row['K_PIPE_MAX'])}, {int(row['USE_SYNC'])}, {row['tflops']:.2f}f}}{comma}  // idx={i}\n")

        macro_cases = "".join(f"\tX({i}) \\\n" for i in range(len(best_df)))

        f.write(f"""
}};

// hardcoded problem sizes for now, will refine dispatch soon
inline int find_config(size_t M, size_t K, size_t N, float alpha, float beta) {{
    bool is_alpha_1 = (alpha == 1.0f);
    bool is_beta_0 = (beta == 0.0f);
    for (int i = 0; i < LOOKUP_TABLE_SIZE; ++i) {{
        const auto& entry = LOOKUP_TABLE[i];
        if (entry.M == M && entry.K == K && entry.N == N &&
            entry.is_alpha_1 == is_alpha_1 && entry.is_beta_0 == is_beta_0) {{
            return i;
        }}
    }}
    return -1;
}}

#define LOOKUP_INDEX_CASES \\
{macro_cases}

}}
""")

if __name__ == "__main__":
    main()