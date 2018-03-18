#!/bin/bash
mkdir -p results

RUNS=10
PROCESSES=10
EPOCHS=10

for e in {1..$EPOCHS} do
  for i in {1..$PROCESSES} do
    for j in $RUNS; do
      python3 local_hogwild/main.py --capture-results True --epochs $e --num-processes $i >> results/local_hogwil_$e_$i.csv
    done
  done
done
