import pandas as pd
import sys

def main(job_id):
    fpath = f"./logs/warm_tuning_results_job{job_id}.csv"
    df = pd.read_csv(fpath)
    result = df.groupby(["M", "K", "N"], group_keys=False).apply(
        lambda g: g.sort_values("median_ms")
    )
    result.to_csv(fpath, index=False)

if __name__ == "__main__":
    main(sys.argv[1])