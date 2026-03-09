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
if [[ -z "$NUM_CONFIGS" ]]; then
    echo "ERROR: unable to parse NUM_KERNEL_CONFIGS from tune.cuh"
    exit 1
fi
ALPHAS=("1" "2")
BETAS=("0" "3")
SIZES=(512 1024 2048 4096 8192 16384 32768)
COLD_CSV="./logs/cold_tuning_results_job${JOB_ID}.csv"

mkdir -p ./logs

# cold phase (for (M,K,N,alpha,beta) group, benchmark each with cooldown)
declare -A cold_done
if [[ -f "$COLD_CSV" ]]; then
    while IFS=',' read -r M K N alpha beta config_idx rest; do
        cold_done["${M},${K},${N},${alpha},${beta},${config_idx}"]=1
    done < <(tail -n +2 "$COLD_CSV")
fi

combo_index=0
for S in "${SIZES[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        for beta in "${BETAS[@]}"; do
            if (( combo_index % NUM_JOBS == JOB_ID )); then
                for config_idx in $(seq 0 $((NUM_CONFIGS-1))); do
                    if [[ -z "${cold_done[${S},${S},${S},${alpha},${beta},${config_idx}]}" ]]; then
                        sleep 6
                        echo "Cold: M=$S K=$S N=$S alpha=$alpha beta=$beta config_idx=$config_idx"
                        ./tune cold $JOB_ID $config_idx $S $S $S $alpha $beta
                    fi
                done
            fi
            (( combo_index++ ))
        done
    done
done

# combo_index=0
# for M in "${SIZES[@]}"; do
#     for K in "${SIZES[@]}"; do
#         for N in "${SIZES[@]}"; do
#             for alpha in "${ALPHAS[@]}"; do
#                 for beta in "${BETAS[@]}"; do
#                     if (( combo_index % NUM_JOBS == JOB_ID )); then
#                         for config_idx in $(seq 0 $((NUM_CONFIGS-1))); do
#                             if [[ -z "${cold_done[${M},${K},${N},${alpha},${beta},${config_idx}]}" ]]; then
#                                 sleep 6
#                                 echo "Cold: M=$M K=$K N=$N alpha=$alpha beta=$beta config_idx=$config_idx"
#                                 ./tune cold $JOB_ID $config_idx $M $K $N $alpha $beta
#                             fi
#                         done
#                     fi
#                     (( combo_index++ ))
#                 done
#             done
#         done
#     done
# done