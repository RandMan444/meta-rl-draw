#!/bin/bash
# Learning-curve test: the retrieval agent vs the same agent without retrieval, 2 seeds each.
set -e
cd "$(dirname "$0")"
OUT=${1:-runs/original}
mkdir -p $OUT
THREADS=4 python3 arm_meta.py $OUT/keys --episodes 0 > $OUT/keys.out 2>&1
for seed in 0 1; do
  THREADS=1 python3 arm_meta.py $OUT/nn-s$seed --keys $OUT/keys/keys.pt --seed $seed > $OUT/nn-s$seed.out 2>&1 &
  THREADS=1 python3 arm_meta.py $OUT/nonn-s$seed --no-nn --keys $OUT/keys/keys.pt --seed $seed > $OUT/nonn-s$seed.out 2>&1 &
done
wait
echo done
