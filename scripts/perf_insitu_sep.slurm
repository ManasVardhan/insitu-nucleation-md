#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-insitu-sep-%j.out
#SBATCH -N 1 -n 8 -c 2 --gres=gpu:p100:2 --exclusive --mem=100G
#SBATCH hetjob
#SBATCH -A anakano_429 -p debug -N 1 -n 4 -c 2 --constraint=xeon-4116 --mem=32G
# T3.2 measurement 4b: ANA ranks on the same node as the SIM ranks vs on a
# separate CPU node (repo root): sbatch scripts/perf_insitu_sep.slurm
# Heterogeneous job: component 0 = GPU node (4 SIM ranks on 2 P100, and
# the 4 ANA ranks for the same-node case), component 1 = a CPU-only debug
# node (Xeon Silver 4116) for the 4 ANA ranks of the separate-node case.
# Same input as perf_insitu.slurm (K = 100, P_ana = 4, 2 threads per ANA).
source scripts/perf_lib.sh insitu_sep
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
RUN_TMO=${RUN_TMO:-600}
REPS=${REPS:-3}; SM=1000; SQ=500; SH=1500
if [ "${SMOKE:-0}" = 1 ]; then REPS=1; SM=100; SQ=50; SH=150; fi
I=$PERF_OUT/in; mkdir -p $I
mk_seeded $I/off.in 100 0 $SM $SQ $SH
mk_seeded $I/on.in 100 4 $SM $SQ $SH
N0=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0 | head -1)
N1=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1 | head -1)
echo "het group 0: $N0, group 1: $N1"
for rep in $(seq 1 $REPS); do
  PLACEMENT=same PERF_NODES=$N0 run off_r$rep off pmd_insitu $rep $I/off.in \
    srun --het-group=0 -n 4 -c 2 --cpu-bind=cores
  PLACEMENT=same PERF_NODES=$N0 run same_r$rep k100_same pmd_insitu $rep $I/on.in \
    srun --het-group=0 -n 8 -c 2 --cpu-bind=cores
  PLACEMENT=separate PERF_NODES=$N0+$N1 run sep_r$rep k100_sep pmd_insitu $rep $I/on.in \
    srun --het-group=0 -n 4 -c 2 --cpu-bind=cores $PERF_ROOT/src/pmd_insitu run.in : \
         --het-group=1 -n 4 -c 2 --cpu-bind=cores
  for t in same sep; do
    [ -f $PERF_OUT/${t}_r$rep/perf_summary.dat ] && \
      cp $PERF_OUT/${t}_r$rep/perf_summary.dat $PERF_ROOT/runs/perf/results/summary_${PERF_JOB}_${t}_r$rep.dat
  done
  grep -h "local rank\|INSITU_ANA rank" $PERF_OUT/sep_r$rep/run.log | head -8
done
echo done $(date)
