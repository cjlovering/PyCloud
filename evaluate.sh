#!/bin/bash
mkdir -p results

RUNS=10
PROCESSES=10
for i in {1..$PROCESSES} do
  for j in $RUNS; do
    python3 local_hogwild/main.py --capture-results True --epochs 1 --num-processes $i >> results/local_hogwild_$i.csv
  done
done
