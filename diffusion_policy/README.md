# Diffusion Policy: PegInsertionSide-v1

Diffusion Policy baseline from ManiSkill v3.0.1, trained on `pd_ee_delta_pose` state demos with observation
normalization. Run every command from this directory (`diffusion_policy/`) with the repo venv
(`../.venv/bin/python`).

Data used by default:
- demos: `../data/dp_training/trajectory.state.pd_ee_delta_pose.physx_cpu.h5`
- norm stats: `../data/dp_training/norm_stats.json`
- held-out grasped start states: `../start_states/eval/states.pt`

## 1. Norm stats

Recompute after changing the demo file.

```bash
../.venv/bin/python ../compute_norm.py \
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
./launch_dp.sh
```

Starts one tmux session per seed (`dp<N>_s<seed>`, N = number of trajectories in the demo file) and trains
on every trajectory in the file. It refuses to start if `diffusion_policy/` has uncommitted changes.

| env variable | default | meaning |
|---|---|---|
| `SEEDS` | `"0 1 2"` | seeds to launch |
| `DATA` | `~/Robotics_Manipulation/data/dp_training` | folder with the demo `.h5` and `norm_stats.json` |
| `PYTHON` | `../.venv/bin/python` | python interpreter |
| `RESUME` | unset | checkpoint to resume from |

Any extra arguments are passed to `train.py`:

```bash
SEEDS="3 4" ./launch_dp.sh --total_iters 50000
RESUME=runs/dp_full1000_s0/checkpoints/50000.pt SEEDS=0 ./launch_dp.sh
```

- Attach: `tmux attach -t dp1000_s0` (detach with `Ctrl-b d`)
- Log: `runs/logs/<exp>.log`
- Checkpoints: `runs/<exp>/checkpoints/` (`<iter>.pt`, `best_eval_success_once.pt`, `best_eval_success_at_end.pt`, `final.pt`)
- Curves: `../.venv/bin/tensorboard --logdir runs --port 6006`

### One run: `train.py`

```bash
../.venv/bin/python train.py --env-id PegInsertionSide-v1 \
  --demo-path ../data/dp_training/trajectory.state.pd_ee_delta_pose.physx_cpu.h5 \
  --control-mode pd_ee_delta_pose --sim-backend physx_cpu \
  --num-demos 1000 --max_episode_steps 300 --total_iters 100000 \
  --num_eval_envs 8 --save_freq 5000 \
  --norm-stats-path ../data/dp_training/norm_stats.json \
  --seed 0 --exp-name dp_full1000_s0
```

| option | default | meaning |
|---|---|---|
| `--env-id` | `PegInsertionSide-v0` | task (use `PegInsertionSide-v1`) |
| `--demo-path` | | demo `.h5` file |
| `--control-mode` | `pd_joint_delta_pos` | must match the demos (`pd_ee_delta_pose`) |
| `--sim-backend` | `physx_cpu` | |
| `--num-demos` | all | number of trajectories to load |
| `--norm-stats-path` | none | normalize observations with this `norm_stats.json`; without it, the original baseline (no normalization) |
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

## 3. Evaluation from grasped start states: `eval_start_states.py`

```bash
../.venv/bin/python eval_start_states.py --run runs/dp_full1000_s0
```

```bash
# all seeds
for s in 0 1 2; do ../.venv/bin/python eval_start_states.py --run runs/dp_full1000_s$s; done
# full-task sanity check from normal resets
../.venv/bin/python eval_start_states.py --run runs/dp_full1000_s0 --full-task 50 --max-episode-steps 300
# videos (run the evaluation first, episodes are picked from its csv)
../.venv/bin/python eval_start_states.py --run runs/dp_full1000_s0 --num-videos 10
```

| option | default | meaning |
|---|---|---|
| `--run` | required | run folder, e.g. `runs/dp_full1000_s0` |
| `--ckpt` | `final.pt` | checkpoint in `<run>/checkpoints/` (EMA weights are used) |
| `--start-states` | `../start_states/eval/states.pt` | start-state file |
| `--num-states` | all | evaluate only the first N states |
| `--full-task` | off | run N normal resets instead of start states |
| `--full-task-seed-base` | `2000000` | seeds for `--full-task` |
| `--max-episode-steps` | `200` | episode length (use `300` with `--full-task`) |
| `--hole-offset-mm` | `0` | shift the observed hole position by this many mm in a random direction |
| `--settle-steps` | `0` | idle steps after restoring a state, before the policy acts |
| `--num-envs` | `50` | envs run in parallel |
| `--num-videos` | `0` | record N episodes (half successes, half failures) |
| `--seed` | `0` | policy noise and hole-offset directions |
| `--out-dir` | `../results/dp_eval` | output folder |

Output in `--out-dir`: `<name>.csv` (one row per episode), `<name>_summary.json` (success rates with 95% Wilson
intervals), `videos/<name>/`. `<name>` is the run name, plus `__<ckpt>` if not `final.pt`, `__fulltask`, and
`__hole<mm>mm`.

## 4. Failure modes: `failure_modes.py`

```bash
../.venv/bin/python failure_modes.py --run runs/dp_full1000_s0
```

Takes the same `--run`, `--ckpt`, `--start-states`, `--num-states`, `--max-episode-steps`, `--hole-offset-mm`,
`--settle-steps`, `--num-envs`, `--seed` and `--out-dir` options as `eval_start_states.py`.

Output in `--out-dir`: `<name>__failmodes.csv` (a `failure_mode` per state) and `<name>__failmodes_summary.json`
(counts per mode). Modes: `misaligned`, `jammed`, `entered_too_shallow`, `never_reached_face`, `dropped`.

## 5. Plots: `plot_dp_eval.py`

```bash
../.venv/bin/python plot_dp_eval.py --runs dp_full1000_s0 dp_full1000_s1 dp_full1000_s2
```

| option | default | meaning |
|---|---|---|
| `--runs` | `dp_full136_s0 s1 s2` | result names (csv files) in `--out-dir` |
| `--out-dir` | `../results/dp_eval` | folder with the csvs; PNGs are written here |
| `--metric` | `success_once` | or `success_at_end` |

Writes `success_rates.png`, `grasped_vs_fulltask.png` (if `__fulltask` csvs exist) and `steps_to_success.png`.

## 6. Normalization tests: `test_normalization.py`

```bash
../.venv/bin/python test_normalization.py
../.venv/bin/python test_normalization.py --ckpt runs/dp_full1000_s0/checkpoints/final.pt
```

| option | default | meaning |
|---|---|---|
| `--ckpt` | none | also check that these checkpoints hold the same norm stats as `norm_stats.json` |
