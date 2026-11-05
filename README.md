# Robotics_Manipulation
pose error contribution

## Setup

```bash
brew install molten-vk vulkan-loader  # macOS only: ManiSkill always builds a Vulkan renderer
uv sync
uv run pytest tests
```

## Residual PPO

```bash
uv run python ppo.py --base hold                               # clean residual (4), sparse reward by default
uv run python ppo.py --base hold --hole-offset-range 0 0.004   # noise-trained residual (4')
uv run python ppo.py                                           # PPO from scratch (3), dense reward by default
uv run python ppo.py <original flags> --checkpoint runs/<run>/ckpt_<i>.pt --exp-name <run>  # resume: repeat every original flag
```

Local smoke run: add `--sim-backend physx_cpu --num-envs 1 --num-eval-envs 1 --num-minibatches 1 --no-capture-video`.

If the tests fail on macOS with `ErrorIncompatibleDriver` or `failed to find a rendering device`, point SAPIEN at MoltenVK:
`export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`
(`evaluate.py` sets these itself).

## Evaluation and classical controller (Kevin)

```bash
uv run python evaluate.py --policy classical --starts start_states/eval/states.pt --out results/classical.json
uv run python evaluate.py --policy my_module:make_policy --policy-kwargs '{"checkpoint": "..."}' --starts start_states/eval/states.pt --num-envs 100
```

- `evaluate.py` runs each start state once per hole offset (default 0, 2, 4, 6 mm). It reports `success_once` (primary) and `success_at_end` with Wilson 95% intervals, plus per-episode results in the JSON. Offset directions are seeded per (start index, magnitude), so every method sees identical offsets. A policy needs `reset()` and `act(obs)`, where `obs` is the batched `state_dict`; `common.flatten_state_dict(obs, use_torch=True)` gives the exact 43-dim `obs_mode="state"` vector.
- `classical.py` is controller (1). It aligns the peg 2 cm in front of the observed hole, matching the square peg to the nearest quarter roll. It then pushes along the hole axis under the stiffness controller. On a stall at the rim (no progress for 4 steps) it spiral searches in the hole plane, pausing whenever the peg starts to drop. An integral term cancels gravity sag. Once the head is 5 mm past the success depth it stops pushing and holds, because servoing on against the walls walks the arm back and pulls the peg out. It is pure feedback on the observation, so the hole offset reaches it exactly as it reaches the learned methods.
- `ClassicalBase` wraps it in `residual.py`'s base interface (`reset(mask)`, `act(flat_obs)`) to replace `HoldBase`.
- `start_states.py` loads start-state files: Sut's list of records from `generate_start_states.py`, or the dict that `save()` writes. The seed rebuilds the peg and hole geometry on reconfigure; `reset_to` checks that against the saved box height, which the task sets to the peg half length.
- `scripts/fetch_start_states.sh` puts Sut's 1000 train and 500 held-out states in `start_states/` (they are in `data_setup`'s history, not at its tip).
- `data/interim_test.pt` holds 50 stand-in states for the tests, made by `scripts/make_interim_start_states.py`. Nothing reported uses them.

## PPO from scratch (Kevin)

- `ppo.py --starts <file>` trains from saved start states (`start_pool.py`). Geometry is fixed per env slot at reconfigure, so a partial reset puts a slot back into its own state. With at least as many slots as states the pool is dealt evenly once (1000 states in 2048 slots); with fewer, every `--starts-reshuffle-freq` iterations all slots are rebuilt from a new batch.
- `--max-episode-steps` defaults to 200, matching the Diffusion Policy evaluation, and evaluation runs for exactly one episode.
- `--logstd-init` and `--gripper-closed` are the settings `cluster/ppo_tune.sbatch` compares. The start states hold the peg on the table about 40 cm from the hole. From there an untrained policy (noise 0.61 on every action) has dropped the peg in every episode by step 50, reward per step 0.035 against 0.277 for holding still. With the gripper forced closed, 30% still hold it at step 200; with noise 0.22 as well, 100% do. PPO learns to keep the peg by itself in under 100k steps (one CPU env, training return 8 to about 59), so the gripper stays free unless the sweep says otherwise.
- `ppo_policy.py` scores a checkpoint with `evaluate.py`: `--policy ppo_policy:make_policy --policy-kwargs '{"checkpoint": "runs/<run>/final_ckpt.pt"}'`.
- `scripts/curves.py` prints runs side by side and writes the learning curves to CSV.

Classical controller on Sut's 500 held-out states (CPU sim, 200 steps):

| Hole offset | `success_once` (95% CI) | `success_at_end` | Mean steps to success |
|---|---|---|---|
| 0 mm | 0.990 [0.977, 0.996] | 0.972 | 34 |
| 2 mm | 0.992 [0.980, 0.997] | 0.974 | 35 |
| 4 mm | 0.976 [0.959, 0.986] | 0.950 | 44 |
| 6 mm | 0.968 [0.949, 0.980] | 0.934 | 50 |

## Colab

`colab/ppo_colab.ipynb` runs the smoke test, the tuning runs, the 75M-step runs and the final evaluation on a Colab GPU ([open in Colab](https://colab.research.google.com/github/zhizhen-kyle-luo/Robotics_Manipulation/blob/kevin/ppo-start-states/colab/ppo_colab.ipynb)). Runs, logs and results go to Google Drive. `ppo_runs.py` resumes each run from its newest checkpoint after a disconnect and skips finished ones.

## Cluster (ORCD Engaging)

```bash
ssh <kerb>@orcd-login.mit.edu
git clone git@github.com:zhizhen-kyle-luo/Robotics_Manipulation.git && cd Robotics_Manipulation
git switch kevin/ppo-start-states
bash cluster/setup_engaging.sh                  # once, on the login node: uv env, start states, Vulkan ICD
sbatch cluster/smoke.sbatch                     # read logs/peg-smoke-<jobid>.out before training anything
sbatch --array=1-4 cluster/ppo_tune.sbatch      # 15M steps over the two PPO settings in question
python scripts/curves.py runs/ppo_tune_*        # after activating .venv: compare them
PPO_FLAGS="<chosen flags>" sbatch --array=1-4 cluster/ppo_scratch.sbatch  # 75M steps: dense seeds 1-3, sparse seed 1
```

Resubmitting `ppo_scratch.sbatch` with the same `PPO_FLAGS` resumes each run from its newest checkpoint. All jobs default to the `mit_normal_gpu` partition; use `sbatch -p <partition>` for a lab's own GPUs.

What to read in the smoke log:
- **Hold test.** `grasped` should be close to 1. The states were generated on CPU physics and training runs on GPU physics; if the grasps do not survive the transfer, stop and regenerate the states on the GPU.
- **Classical on the GPU backend.** It should be close to the CPU table above. ManiSkill's GPU delta-pose controller is a one-step Jacobian solve rather than a converged IK, in the same frame and with the same sign.
- **`SPS`.** 75M steps / SPS is the run time.
- **`torch cannot see a GPU`.** The locked torch wheels are built for CUDA 13.2 and need NVIDIA driver 580 or newer; the log prints the node's driver version.
