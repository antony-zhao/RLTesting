#!/bin/bash
# Atari experiments (standard conv encoder, weighted loss from config)
# Run: bash run_atari.sh
set -e

echo "============================================"
echo "  Atari Experiments"
echo "============================================"

for ENV in boxing pong bowling assault; do
    echo ""
    echo "========== $ENV =========="

    echo "--- Autoencoder ---"
    python train_autoencoder.py --env $ENV --force --impala

    echo "--- SARSA pixels ---"
    python train_sarsa.py --env $ENV

    echo "--- PPO pixels ---"
    python train_ppo.py --env $ENV --mode pixels

    echo "========== $ENV done =========="
done

echo ""
echo "============================================"
echo "  All Atari experiments complete"
echo "============================================"
