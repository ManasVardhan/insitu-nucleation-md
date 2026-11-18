#!/bin/bash
#SBATCH -A anakano_429
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -t 00:30:00
#SBATCH -o runs/smoke/slurm-%j.out
# Runs up to two smoke configurations side by side, one GPU each.
# Usage (repo root): sbatch -p gpu scripts/smoke.slurm "0.70 0" "0.70 4.0"
# Each argument is "TQ RSEED [HOLD]".
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
nvidia-smi -L
for cfg in "$@"; do
  bash scripts/run_smoke.sh $cfg &
done
wait
