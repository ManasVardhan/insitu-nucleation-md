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
#SBATCH -o runs/perf/slurm-drift-%j.out
# Step time vs simulated time (repo root): sbatch scripts/perf_drift.slurm
# The P100 production run averaged 36-37 ms/step against 26-28 ms in the
# 4000-step smoke tests. This repeats the first 15000 steps of the
# production protocol (melt 5000, ramp 2000, hold 8000) with analysis off
# and on (K = 100, P_ana = 4), with WALL lines every 500 steps.
source scripts/perf_lib.sh drift
export OMP_NUM_THREADS=2 OMP_PROC_BIND=close OMP_PLACES=cores
SM=5000; SQ=2000; SH=8000
if [ "${SMOKE:-0}" = 1 ]; then SM=200; SQ=100; SH=200; fi
I=$PERF_OUT/in; mkdir -p $I
mk_seeded $I/off.in 100 0 $SM $SQ $SH
mk_seeded $I/on.in 100 4 $SM $SQ $SH
for c in off:0 on:4; do
  IFS=: read tag P <<< "$c"
  run $tag $tag pmd_insitu 1 $I/$tag.in srun -n $((4+P)) -c 2 --cpu-bind=cores
  grep '^WALL' $PERF_OUT/$tag/run.log > $PERF_ROOT/runs/perf/results/drift_${PERF_JOB}_$tag.wall
  [ -f $PERF_OUT/$tag/perf_summary.dat ] && \
    cp $PERF_OUT/$tag/perf_summary.dat $PERF_ROOT/runs/perf/results/summary_${PERF_JOB}_drift_$tag.dat
done
echo done $(date)
