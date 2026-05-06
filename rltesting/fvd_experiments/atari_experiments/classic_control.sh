#!/bin/bash
# Classic Control experiments
# Run: bash run_classic.sh
set -e

echo "============================================"
echo "  Classic Control Experiments"
echo "============================================"

for ENV in cartpole acrobot mountaincar; do
    echo ""
    echo "========== $ENV =========="

    echo "--- SARSA raw state ---"
    python train_sarsa.py --env $ENV --mode raw

    echo "--- Autoencoder ---"
    python train_autoencoder.py --env $ENV --force

    echo "--- SARSA pixels ---"
    python train_sarsa.py --env $ENV --mode pixels

    echo "--- PPO state ---"
    python train_ppo.py --env $ENV --mode ram

    echo "--- PPO pixels ---"
    python train_ppo.py --env $ENV --mode pixels

    echo "========== $ENV done =========="
done

echo ""
echo "============================================"
echo "  All classic control experiments complete"
echo "============================================"