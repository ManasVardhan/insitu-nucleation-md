#!/bin/bash -l
#SBATCH -A anakano_429
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 10
#SBATCH -c 2
#SBATCH --gres=gpu:p100:2
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -t 00:30:00
#SBATCH -o runs/perf/slurm-insitu-%j.out
# T3.2 measurement 4: in situ overhead on one node (repo root):
#   sbatch scripts/perf_insitu.slurm "off:0:0" "k100:100:4" ...
# Each case is tag:K:P_ana (P_ana = 0: analysis off). The production layout:
# N = 1,048,576, 4 SIM ranks (vproc 2 2 1) on 2 P100, P_ana ANA ranks with
# 2 OpenMP threads each, every rank bound to 2 cores (as in the P100
# production run). Seeded protocol shortened to melt 1000, ramp 500, hold
# 1500 steps (3000 steps), no movie. Cases are interleaved over REPS reps.
source scripts/perf_lib.sh insitu
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
RUN_TMO=${RUN_TMO:-600}
REPS=${REPS:-3}; SM=1000; SQ=500; SH=1500
if [ "${SMOKE:-0}" = 1 ]; then REPS=1; SM=100; SQ=50; SH=150; fi
[ $# -eq 0 ] && set -- off:0:0 k100:100:4
I=$PERF_OUT/in; mkdir -p $I
for rep in $(seq 1 $REPS); do
  for c in "$@"; do
    IFS=: read tag K P <<< "$c"
    mk_seeded $I/$tag.in $K $P $SM $SQ $SH
    run ${tag}_r$rep $tag pmd_insitu $rep $I/$tag.in srun -n $((4+P)) -c 2 --cpu-bind=cores
    [ -f $PERF_OUT/${tag}_r$rep/perf_summary.dat ] && \
      cp $PERF_OUT/${tag}_r$rep/perf_summary.dat $PERF_ROOT/runs/perf/results/summary_${PERF_JOB}_${tag}_r$rep.dat
  done
done
echo done $(date)
