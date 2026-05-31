#!/bin/bash
# run_classic.sh — All classic control SARSA experiments

for env in cartpole acrobot mountaincar; do
  for mode in raw pixels; do
    echo "=========================================="
    echo "  $env $mode"
    echo "=========================================="
    python train_sarsa.py --env $env --mode $mode --max-steps 1000000
  done
done