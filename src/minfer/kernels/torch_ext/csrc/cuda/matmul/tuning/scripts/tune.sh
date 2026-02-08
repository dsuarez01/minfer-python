#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --gpu-freq=high
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

./tune $SLURM_ARRAY_TASK_ID 8
