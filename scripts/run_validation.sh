#!/bin/bash
# Phase 1 validation runs on CARC Discovery. Run inside an allocation:
#   salloc -p debug -A anakano_429 -N 1 -n 4 --gres=gpu:2 -t 00:40:00 \
#     bash scripts/run_validation.sh
# (from the repo root). Outputs go to runs/validation/out/.
set -e
module load gcc/13.3.0 openmpi/5.0.5 cuda/12.6.3
ROOT=$(pwd)
make -C src
mkdir -p runs/validation/out && cd runs/validation/out

# GPU vs CPU forces, 2 ranks and 4 ranks (4 ranks share the 2 GPUs)
srun -n 2 $ROOT/src/pmd_cuda ../force_check.in  > force_check.log
srun -n 4 $ROOT/src/pmd_cuda ../force_check4.in > force_check4.log

# NVE drift, g(r), thermostat: GPU and CPU builds
srun -n 2 $ROOT/src/pmd_cuda ../nve.in > nve_gpu.log
mkdir -p cpu && cd cpu
srun -n 2 $ROOT/src/pmd ../../nve.in > ../nve_cpu.log
