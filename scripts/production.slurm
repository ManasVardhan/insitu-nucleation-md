#!/bin/bash
#SBATCH -A anakano_429
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH -c 16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o runs/production/slurm-%j.out
# T3.1 production run (repo root):
#   sbatch scripts/production.slurm [input:outdir ...]
# Default: runs/production/prod.in:runs/production/out. Several pairs
# run one after the other. Ranks are placed cyclically over the 2 nodes:
# world ranks 0..3 are SIM (2 per node, one A100 each), 4..7 are ANA
# (2 per node). SIM s sends to ANA 4+s, which lands on the same node.
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close OMP_PLACES=cores
ROOT=$(pwd)
nvidia-smi -L
[ $# -eq 0 ] && set -- runs/production/prod.in:runs/production/out
for pair in "$@"; do
  IN=${pair%%:*}; OUT=${pair##*:}; B=$(basename $IN)
  mkdir -p $OUT && cp $IN $OUT/ && cd $OUT
  NT=$(( $(awk '/^vproc/{print $2*$3*$4}' $B) + $(awk '/^InsituRanks/{print $2}' $B) ))
  echo "== $IN -> $OUT ($NT ranks) $(date)"
  srun -n $NT --distribution=cyclic --cpu-bind=cores $ROOT/src/pmd_insitu $B > run.log 2>&1
  grep -E "TIMING|INSITU" run.log
  du -sh . ; cd $ROOT
done
