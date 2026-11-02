#!/usr/bin/env bash
# Full Diffusion Policy runs on PegInsertionSide-v1, 136 pd_ee_delta_pose demos, seeds 0 1 2.
# Each seed runs in its own detached tmux session (dp_s<seed>), spread round-robin over the visible GPUs.
# Logs: runs/logs/dp_full136_s<seed>.log; checkpoints: runs/dp_full136_s<seed>/checkpoints/
#
# Usage: ./launch_dp.sh [extra train.py args, e.g. --track]
# Resume a seed: RESUME=runs/dp_full136_s0/checkpoints/50000.pt SEEDS=0 ./launch_dp.sh --track
set -euo pipefail
cd "$(dirname "$0")"

# absolute path, but do not resolve the venv symlink (that would drop the venv)
PYTHON=${PYTHON:-$(cd .. && pwd)/.venv/bin/python}
DATA=${DATA:-$HOME/Robotics_Manipulation/data/dp_training}
SEEDS=${SEEDS:-"0 1 2"}
WANDB_PROJECT=${WANDB_PROJECT:-dp_peginsertion_full136}
NUM_GPUS=$(nvidia-smi -L | wc -l)

if [[ -n "$(git status --porcelain -- .)" ]]; then
  echo "diffusion_policy/ has uncommitted changes; commit first so runs map to an exact code version" >&2
  exit 1
fi

mkdir -p runs/logs
i=0
for seed in $SEEDS; do
  exp=dp_full136_s${seed}
  session=dp_s${seed}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists, skipping seed $seed" >&2
    i=$((i + 1)); continue
  fi
  resume_args=""
  if [[ -n "${RESUME:-}" ]]; then resume_args="--resume $(printf %q "$RESUME")"; fi
  cmd="CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) $PYTHON train.py --env-id PegInsertionSide-v1 \
    --demo-path $DATA/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
    --control-mode pd_ee_delta_pose --sim-backend physx_cpu \
    --num-demos 136 --max_episode_steps 300 --total_iters 100000 \
    --num_eval_envs 8 --save_freq 5000 \
    --norm-stats-path $DATA/norm_stats.json \
    --seed $seed --exp-name $exp --wandb_project_name $WANDB_PROJECT \
    $resume_args $* 2>&1 | tee -a runs/logs/$exp.log"
  tmux new-session -d -s "$session" -c "$PWD" "$cmd; echo '[exited]'; exec bash"
  echo "seed $seed -> tmux $session, GPU $((i % NUM_GPUS)), log runs/logs/$exp.log"
  i=$((i + 1))
done
