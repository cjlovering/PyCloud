#!/bin/bash
mkdir -p results

RUNS=10
PROCESSES=10

for i in {1..$RUNS}; do
  for j in {1..$PROCESSES}; do
    python3 hogwild/main.py --capture-results True --epochs 10 --num-processes $i >> results/$i.csv
  done
done
