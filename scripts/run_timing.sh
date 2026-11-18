#!/bin/bash
# Preliminary CPU vs GPU step time vs N, 2 ranks (vproc 1 1 2), one node.
# Run inside an allocation, from the repo root:
#   salloc -p debug -A anakano_429 -N 1 -n 2 --gres=gpu:2 -t 00:30:00 \
#     bash scripts/run_timing.sh
set -e
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
ROOT=$(pwd)
make -C src
mkdir -p runs/timing && cd runs/timing
for k in 8 12 16 24 32; do
  printf "vproc 1 1 2\nInitUcell $k $k $((k/2))\nDensity 0.95\nInitTemp 1.0\nStepLimit 200\nStepAvg 200\n" > t$k.in
  for b in pmd pmd_cuda; do
    srun -n 2 $ROOT/src/$b t$k.in | grep TIMING | sed "s/^/$b /"
  done
done | tee timing.txt
