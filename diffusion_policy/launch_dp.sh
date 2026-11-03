#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

# absolute path, but do not resolve the venv symlink (that would drop the venv)
PYTHON=${PYTHON:-$(cd .. && pwd)/.venv/bin/python}
DATA=${DATA:-$HOME/Robotics_Manipulation/data/dp_training}
SEEDS=${SEEDS:-"0 1 2"}
NUM_GPUS=$(nvidia-smi -L | wc -l)
DEMO_PATH=$DATA/trajectory.state.pd_ee_delta_pose.physx_cpu.h5
# always train on every trajectory in the demo file
NUM_DEMOS=$("$PYTHON" -c 'import h5py, sys; print(sum(k.startswith("traj_") for k in h5py.File(sys.argv[1], "r")))' "$DEMO_PATH")

if [[ -n "$(git status --porcelain -- .)" ]]; then
  echo "diffusion_policy/ has uncommitted changes; commit first so runs map to an exact code version" >&2
  exit 1
fi

mkdir -p runs/logs
i=0
for seed in $SEEDS; do
  exp=dp_full${NUM_DEMOS}_s${seed}
  session=dp${NUM_DEMOS}_s${seed}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session $session already exists, skipping seed $seed" >&2
    i=$((i + 1)); continue
  fi
  resume_args=""
  if [[ -n "${RESUME:-}" ]]; then resume_args="--resume $(printf %q "$RESUME")"; fi
  cmd="CUDA_VISIBLE_DEVICES=$((i % NUM_GPUS)) $PYTHON train.py --env-id PegInsertionSide-v1 \
    --demo-path $DEMO_PATH \
    --control-mode pd_ee_delta_pose --sim-backend physx_cpu \
    --num-demos $NUM_DEMOS --max_episode_steps 300 --total_iters 100000 \
    --num_eval_envs 8 --save_freq 5000 \
    --norm-stats-path $DATA/norm_stats.json \
    --seed $seed --exp-name $exp \
    $resume_args $* 2>&1 | tee -a runs/logs/$exp.log"
  tmux new-session -d -s "$session" -c "$PWD" "$cmd; echo '[exited]'; exec bash"
  echo "seed $seed -> tmux $session, GPU $((i % NUM_GPUS)), log runs/logs/$exp.log"
  i=$((i + 1))
done
