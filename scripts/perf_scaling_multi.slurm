#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=2
#SBATCH -c 4
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-scaling-multi-%j.out
# T3.2 measurements 2-3 across nodes, one rank per GPU (repo root):
#   sbatch scripts/perf_scaling_multi.slurm
# 4 nodes x 2 V100 (change --gres for another type). Strong scaling at
# N = 1,048,576 and weak scaling at 262,144 atoms per GPU on 1, 2, 4, 8 GPUs
# (2 GPUs per node: 1 and 2 GPUs on one node, 4 on 2 nodes, 8 on 4 nodes).
source scripts/perf_lib.sh scaling_multi
export OMP_NUM_THREADS=1
PERF_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd+)
REPS=${REPS:-3}; SG=300
[ "${SMOKE:-0}" = 1 ] && { REPS=1; SG=10; }
I=$PERF_OUT/in; mkdir -p $I
mk_md $I/s1.in "1 1 1" "64 64 64" $SG
mk_md $I/s2.in "1 1 2" "64 64 32" $SG
mk_md $I/s4.in "2 2 1" "32 32 64" $SG
mk_md $I/s8.in "2 2 2" "32 32 32" $SG
mk_md $I/w1.in "1 1 1" "32 32 64" $SG
mk_md $I/w2.in "2 1 1" "32 32 64" $SG
mk_md $I/w4.in "2 2 1" "32 32 64" $SG
mk_md $I/w8.in "2 2 2" "32 32 64" $SG
for rep in $(seq 1 $REPS); do
  for P in 1 2 4 8; do
    NN=$(( P < 2 ? 1 : P/2 ))
    L="srun -N $NN -n $P --ntasks-per-node=$(( P < 2 ? 1 : 2 )) --cpu-bind=cores"
    run s_${P}_r$rep gpu_strong_1M pmd_cuda $rep $I/s$P.in $L
    run w_${P}_r$rep gpu_weak_262k pmd_cuda $rep $I/w$P.in $L
  done
done
echo done $(date)
