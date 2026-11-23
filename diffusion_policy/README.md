# Diffusion Policy: PegInsertionSide-v1

Diffusion Policy baseline from ManiSkill v3.0.1, trained on `pd_ee_delta_pose` state demos with raw
observations, as the baseline does. Run every command from this directory (`diffusion_policy/`) with the repo venv
(`../.venv/bin/python`).

Data used by default:
- demos: `../data/dp_training/trajectory.state.pd_ee_delta_pose.physx_cpu.h5`
- norm stats: `../data/dp_training/norm_stats.json` (for the residual's input only, `ppo.py --obs-norm-stats`)
- held-out grasped start states: `../start_states/eval/states.pt`

## 1. Norm stats

The residual z-scores its observations with these and clips them (`residual.py` `ObsNormalizer`). The DP is
not trained with them: joint velocities are near zero in the demos, so their z-scores reach the hundreds on a
policy's own motion, and a normalized DP fell to about 5 percent success against about 60 raw. Recompute after
changing the demo file.

```bash
../.venv/bin/python compute_norm.py \
  --demo-path ../data/dp_training/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --output-dir ../data/dp_training
```

| option | default | meaning |
|---|---|---|
| `--demo-path` | required | demo `.h5` file |
| `--output-dir` | required | where `norm_stats.json` is written |
| `--min-std` | `0.01` | std floor for near-constant dims |
| `--include-actions` | off | also write action stats |

## 2. Training

### All seeds: `launch_dp.sh`

```bash
./launch_dp.sh                            # clean DP, seeds 0 1 2
TAG=noise ./launch_dp.sh --hole-noise     # noise-augmented DP, seeds 0 1 2
```

Starts one tmux session per seed (`dp_<TAG>_s<seed>`, also the run name). Training uses every successful demo
in the file, one per seed (649 of the 1000 in the current file; `train.py` logs the count). `--hole-noise` gives
each demo an observed hole pose error (up to 6 mm in the hole plane and 1 deg), constant over the demo and redrawn
every pass over the data. It refuses to start if `diffusion_policy/` has uncommitted changes.

| env variable | default | meaning |
|---|---|---|
| `SEEDS` | `"0 1 2"` | seeds to launch |
| `TAG` | `clean` | run name part |
| `DATA` | `~/Robotics_Manipulation/data/dp_training` | folder with the demo `.h5` and `norm_stats.json` |
| `PYTHON` | `../.venv/bin/python` | python interpreter |
| `RESUME` | unset | checkpoint to resume from |

Any extra arguments are passed to `train.py`:

```bash
SEEDS="3 4" ./launch_dp.sh --total_iters 50000
RESUME=runs/dp_clean_s0/checkpoints/50000.pt SEEDS=0 ./launch_dp.sh
```

- Attach: `tmux attach -t dp_clean_s0` (detach with `Ctrl-b d`)
- Log: `runs/logs/<exp>.log`
- Checkpoints: `runs/<exp>/checkpoints/` (`<iter>.pt`, `best_eval_success_once.pt`, `best_eval_success_at_end.pt`, `final.pt`)
- Curves: `../.venv/bin/tensorboard --logdir runs --port 6006`

### One run: `train.py`

```bash
../.venv/bin/python train.py --env-id PegInsertionSide-v1 \
  --demo-path ../data/dp_training/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode pd_ee_delta_pose --sim-backend physx_cpu \
  --max_episode_steps 300 --total_iters 100000 \
  --num_eval_envs 8 --save_freq 5000 \
  --seed 0 --exp-name dp_clean_s0
```

| option | default | meaning |
|---|---|---|
| `--env-id` | `PegInsertionSide-v0` | task (use `PegInsertionSide-v1`) |
| `--demo-path` | | demo `.h5` file |
| `--control-mode` | `pd_joint_delta_pos` | must match the demos (`pd_ee_delta_pose`) |
| `--sim-backend` | `physx_cpu` | |
| `--num-demos` | all | number of trajectories to load, from the successful one-per-seed demos |
| `--hole-noise` | off | perturb the observed hole pose per demo (noise-augmented DP) |
| `--norm-stats-path` | none | not used: z-scores observations inside the DP, which broke it (see section 1) |
| `--resume` | none | continue from a checkpoint (`runs/<exp>/checkpoints/<tag>.pt`) |
| `--seed` | `1` | |
| `--exp-name` | auto | run folder name under `runs/` |
| `--total_iters` | `1000000` | training iterations |
| `--batch_size` | `1024` | |
| `--lr` | `1e-4` | |
| `--max_episode_steps` | env default | episode length during training-time eval |
| `--eval_freq` | `5000` | iterations between evals |
| `--num_eval_episodes` | `100` | episodes per eval |
| `--num_eval_envs` | `10` | parallel eval envs |
| `--save_freq` | none | iterations between numbered checkpoints |
| `--log_freq` | `1000` | iterations between loss logs |
| `--obs_horizon` / `--act_horizon` / `--pred_horizon` | `2` / `8` / `16` | policy horizons |
| `--track` | off | log to wandb (`--wandb_project_name`, `--wandb_entity`) |
| `--capture_video` | on | save eval videos |

All options: `../.venv/bin/python train.py --help`

### Export for the residual and `evaluate.py`

Run from the repo root. The export keeps the EMA weights and adds the config
`residual.DiffusionBase` reads; pass any architecture flag the run changed.

```bash
for tag in clean noise; do for s in 0 1 2; do
  uv run python scripts/export_dp_base.py diffusion_policy/runs/dp_${tag}_s$s/checkpoints/final.pt base_ckpts/dp_${tag}_s$s.pt
  uv run python evaluate.py --policy dp_policy:make_dp --policy-kwargs "{\"checkpoint\": \"base_ckpts/dp_${tag}_s$s.pt\"}" \
    --starts start_states/eval/states.pt --num-envs 100 --out results/dp_${tag}_s$s.json
done; done
uv run python ppo.py --base-checkpoint base_ckpts/dp_clean_s0.pt --obs-norm-stats data/dp_training/norm_stats.json ...   # residual on one of them
uv run python evaluate.py --policy dp_policy:make_residual --policy-kwargs '{"checkpoint": "runs/<run>/final_ckpt.pt"}' \
  --starts start_states/eval/states.pt --num-envs 100 --out results/residual_<run>.json
```

## 3. Normalization tests: `test_normalization.py`

```bash
../.venv/bin/python test_normalization.py
../.venv/bin/python test_normalization.py --ckpt runs/dp_full1000_s0/checkpoints/final.pt
```

| option | default | meaning |
|---|---|---|
| `--ckpt` | none | also check that these checkpoints hold the same norm stats as `norm_stats.json` |
