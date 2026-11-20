#!/bin/bash
# GPU smoke test, run this before any training. Submit from the repo root: sbatch cluster/smoke.sbatch
#   1. Sut's grasps survive GPU physics, and simulator throughput
#   2. the classical controller on the GPU backend (compare with its CPU numbers in the README)
#SBATCH --job-name=peg-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source cluster/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

python scripts/gpu_smoke.py --starts start_states/eval/states.pt --num-envs 1024

python evaluate.py --policy classical --starts start_states/eval/states.pt --limit 100 \
    --num-envs 100 --sim-backend physx_cuda --noise-mm 0 2 4 6 \
    --out results/classical_v1_gpu100.json

echo "smoke test finished"
