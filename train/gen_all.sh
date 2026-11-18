#!/bin/sh
# Generate all labeled snapshots into train/data/ (gitignored).
#   train state points: T in {0.45,0.55,0.65,0.75} x rho in {0.90,0.95,1.00}
#   test  state points: T in {0.50,0.60,0.70}      x rho in {0.925,0.975}
#   slab (evaluation):  T in {0.50,0.55,0.60,0.65} at rho = 0.95
# Each bulk run: 9^3 FCC cells (2916 atoms), 4 snapshots 500 steps apart.
# Each slab run: 8x8x16 cells (4096 atoms), 3 snapshots 50 steps apart.
set -e
cd "$(dirname "$0")"
make -s gen_data
mkdir -p data/runs
JOBS=${JOBS:-8}

{
  seed=100
  for T in 0.45 0.55 0.65 0.75; do for rho in 0.90 0.95 1.00; do
    for set in crystal liquid; do seed=$((seed+1)); echo "train $set $T $rho 9 4 $seed"; done
  done; done
  for T in 0.50 0.60 0.70; do for rho in 0.925 0.975; do
    for set in crystal liquid; do seed=$((seed+1)); echo "test $set $T $rho 9 4 $seed"; done
  done; done
  for T in 0.50 0.55 0.60 0.65; do seed=$((seed+1)); echo "slab slab $T 0.95 8 3 $seed"; done
} | xargs -P "$JOBS" -L 1 sh -c \
  './gen_data "$1" "$2" "$3" "$4" "$5" "$6" > "data/runs/$0_$1_$2_$3.csv" 2> "data/runs/$0_$1_$2_$3.log"'

HDR="set,T,rho,snap,id,label,z,dist,qbar4,qbar6,q4,q6,coord"
for split in train test slab; do
  { echo "$HDR"; cat data/runs/${split}_*.csv; } > data/$split.csv
done
cat data/runs/*.log > data/diagnostics.log
wc -l data/train.csv data/test.csv data/slab.csv
