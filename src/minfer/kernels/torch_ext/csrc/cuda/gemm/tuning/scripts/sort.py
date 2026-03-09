import pandas as pd
import sys

def main(job_id):
    fpath = f"./logs/warm_tuning_results_job{job_id}.csv"
    df = pd.read_csv(fpath)
    df.sort_values(
        ["M", "K", "N", "alpha", "beta", "success", "median_ms"], 
        ascending=[True, True, True, True, True, False, True], inplace=True
    )
    df.to_csv(fpath, index=False)

if __name__ == "__main__":
    main(sys.argv[1])