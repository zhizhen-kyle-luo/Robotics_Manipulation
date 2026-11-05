#!/bin/bash
# Re-tune PPO from scratch for pd_ee_delta_pose before the 75M-step runs: 15M steps, dense reward, one seed,
# over the two settings the grasped start states put in question.
#   logstd-init: the joint-space baseline's -0.5 means noise of about 6 cm per step here
#   gripper-closed: an untrained policy drops the peg within 50 steps of every episode
#     sbatch --array=1-4 cluster/ppo_tune.sbatch
# Compare with: python scripts/curves.py runs/ppo_tune_*
#SBATCH --job-name=ppo-tune
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
source cluster/env.sh

case "$SLURM_ARRAY_TASK_ID" in
    1) NAME=baseline;            FLAGS=() ;;
    2) NAME=closed;              FLAGS=(--gripper-closed) ;;
    3) NAME=lowstd;              FLAGS=(--logstd-init -1.5) ;;
    4) NAME=lowstd_closed;       FLAGS=(--logstd-init -1.5 --gripper-closed) ;;
    *) echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

python ppo.py --seed 1 --exp-name "ppo_tune_$NAME" --no-capture-video --total-timesteps 15000000 \
    --starts start_states/train/states.pt --eval-starts start_states/eval/states.pt --num-eval-envs 500 \
    "${FLAGS[@]}"
