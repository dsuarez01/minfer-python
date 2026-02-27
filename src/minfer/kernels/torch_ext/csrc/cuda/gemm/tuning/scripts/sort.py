import pandas as pd
import sys

def main(job_id):
    fpath = f"./logs/warm_tuning_results_job{job_id}.csv"
    df = pd.read_csv(fpath)
    df.sort_values(["M", "K", "N", "alpha", "beta", "median_ms"], inplace=True)
    df.to_csv(fpath, index=False)

if __name__ == "__main__":
    main(sys.argv[1])