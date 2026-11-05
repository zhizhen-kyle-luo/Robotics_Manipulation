#!/bin/bash
# GPU smoke test, run this before any training. Submit from the repo root: sbatch cluster/smoke.sbatch
#   1. Sut's grasps survive GPU physics, and simulator throughput
#   2. the classical controller on the GPU backend (compare with its CPU numbers in the README)
#   3. PPO at the real scale (2048 train slots, 500 eval slots), then a resume from its checkpoint
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

rm -rf runs/smoke_ppo
PPO=(--num-envs 2048 --num-eval-envs 500 --eval-freq 2 --no-capture-video --exp-name smoke_ppo
     --starts start_states/train/states.pt --eval-starts start_states/eval/states.pt)
python ppo.py "${PPO[@]}" --total-timesteps 131072
python ppo.py "${PPO[@]}" --total-timesteps 196608 --checkpoint runs/smoke_ppo/ckpt_3.pt
echo "smoke test finished"
