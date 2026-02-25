#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --gpu-freq=high
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

# need to source env setup script at project root
source "$(git rev-parse --show-toplevel)/setup.sh"

JOB_ID=$SLURM_ARRAY_TASK_ID
NUM_JOBS=$SLURM_ARRAY_TASK_COUNT
NUM_CONFIGS=$(grep -oP 'NUM_KERNEL_CONFIGS\s*=\s*\K[0-9]+' tune.cuh)
ALPHA=2.0
BETA=3.0
SIZES=(512 1024 2048 4096 8192 16384 32768)
WARM_CSV="./logs/warm_tuning_results_job${JOB_ID}.csv"
COLD_CSV="./logs/cold_tuning_results_job${JOB_ID}.csv"

mkdir -p ./logs

# warm phase (benchmark all configs for this job/GPU's problem sizes at once)
declare -A warm_done
if [[ -f "$WARM_CSV" ]]; then
    while IFS=',' read -r M K N alpha beta config_idx rest; do
        warm_done["${M},${K},${N},${config_idx}"]=1
    done < <(tail -n +2 "$WARM_CSV")
fi

combo_index=0
for M in "${SIZES[@]}"; do
    for K in "${SIZES[@]}"; do
        for N in "${SIZES[@]}"; do
            if (( combo_index % NUM_JOBS == JOB_ID )); then
                for config_idx in $(seq 0 $((NUM_CONFIGS - 1))); do
                    if [[ -z "${warm_done[${M},${K},${N},${config_idx}]}" ]]; then
                        echo "Warm: starting M=$M K=$K N=$N config_idx=$config_idx"
                        ./tune warm $JOB_ID $config_idx $M $K $N $ALPHA $BETA
                    fi
                done
            fi
            (( combo_index++ ))
        done
    done
done

# sort each (M,K,N) group by median_ms ascending, overwrite in-place
python scripts/sort.py $JOB_ID

# cold phase (for (M,K,N) group, take top 5% of configs, benchmark each as separate process)
declare -A cold_done
if [[ -f "$COLD_CSV" ]]; then
    while IFS=',' read -r M K N alpha beta config_idx rest; do
        cold_done["${M},${K},${N},${config_idx}"]=1
    done < <(tail -n +2 "$COLD_CSV")
fi

while IFS=',' read -r M K N alpha beta config_idx rest; do
    if [[ -z "${cold_done[${M},${K},${N},${config_idx}]}" ]]; then
        sleep 6
        echo "Cold: starting M=$M K=$K N=$N alpha=$ALPHA beta=$BETA config_idx=$config_idx"
        ./tune cold $JOB_ID $config_idx $M $K $N $ALPHA $BETA
    fi
done < <(awk -F',' '
NR==1 { next }
{
    key = $1","$2","$3
    count[key]++
    lines[key][count[key]] = $0
}
END {
    for (k in count) {
        top = int(count[k]*0.05)
        if (top<1) top=1
        printf "(Cold) M,K,N=%s taking top %d/%d configs\n", k, top, count[k] > "/dev/stderr"
        for (i=1; i<=top; i++) print lines[k][i]
    }
}' "$WARM_CSV")