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
uv run python ppo.py --base dp --starts start_states/train/states.pt   # clean residual (4), ResiP's settings (the Args defaults in ppo.py)
uv run python ppo.py --base dp --starts start_states/train/states.pt \
    --hole-offset-range 0 0.006 --hole-rot-range-deg 0 1                # noise-trained residual (4'): 0-6 mm, 0-1 deg
uv run python ppo.py <original flags> --checkpoint runs/<run>/ckpt_<i>.pt --exp-name <run>  # resume: repeat every original flag
```

Local smoke run: add `--sim-backend physx_cpu --num-envs 1 --num-eval-envs 1 --no-capture-video`.

- `ppo.py --starts <file>` trains from saved start states (`start_pool.py`). Geometry is fixed per env slot at reconfigure, so a partial reset puts a slot back into its own state. With at least as many slots as states the pool is dealt evenly once; with fewer, every full reset (one per iteration) rebuilds all slots from a new batch.
- `--max-episode-steps` defaults to 200, matching the Diffusion Policy evaluation, and evaluation runs for exactly one episode.
- The in-training evaluation runs on 500 envs (`--num-eval-envs`) with no hole pose error, on `--eval-starts`, which defaults to `--starts`. Keep it on the training states: the held-out `start_states/eval` states are scored only by `evaluate.py` at the end. `runs/<run>/best_ckpt.pt` is the checkpoint with the best in-training `success_once`, as ResiP keeps it; `final_ckpt.pt` is the last one.

If the tests fail on macOS with `ErrorIncompatibleDriver` or `failed to find a rendering device`, point SAPIEN at MoltenVK:
`export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`
(`evaluate.py` sets these itself).

## Evaluation and classical controller (Kevin)

```bash
uv run python evaluate.py --policy classical --starts start_states/eval/states.pt --out results/classical.json
uv run python evaluate.py --policy my_module:make_policy --policy-kwargs '{"checkpoint": "..."}' --starts start_states/eval/states.pt --num-envs 100
uv run python evaluate.py --policy classical --starts start_states/eval/states.pt --rot-deg 1 --out results/classical_rot1.json --log-steps
```

- `evaluate.py` runs each start state once per hole offset (default 0, 2, 4, 6 mm). It reports `success_once` (primary) and `success_at_end` with Wilson 95% intervals, plus per-episode results in the JSON. Offset directions are seeded per (start index, magnitude, `--seed`), so every method sees identical offsets. `--rot-deg r` adds an r degree error to the observed hole orientation, about an axis seeded the same way. Each episode records `start_idx`, the realised `offset_yz`, `rot_axis_angle` and `seed`. `--log-steps` also writes per-step traces to `<out>_steps.npz`: peg head position in the true hole frame, roll error to the nearest quarter turn, TCP pose, `is_grasped`, peg-box contact force, success, the executed action, and the policy's `last_base_action`, `last_residual_action`, `phase` and `spiral_count` when it exposes them. A policy needs `reset()` and `act(obs)`, where `obs` is the batched `state_dict`; `common.flatten_state_dict(obs, use_torch=True)` gives the exact 43-dim `obs_mode="state"` vector.
- `classical.py` is controller (1). It aligns the peg 2 cm in front of the observed hole, matching the square peg to the nearest quarter roll. It then pushes along the hole axis under the stiffness controller. On a stall at the rim (no progress for 4 steps) it spiral searches in the hole plane, pausing whenever the peg starts to drop. An integral term cancels gravity sag. Once the head is 5 mm past the success depth it stops pushing and holds, because servoing on against the walls walks the arm back and pulls the peg out. It is pure feedback on the observation, so the hole offset reaches it exactly as it reaches the learned methods.
- `ClassicalBase` wraps it in `residual.py`'s base interface (`reset(mask)`, `act(flat_obs)`).
- `start_states.py` loads start-state files: Sut's list of records from `generate_start_states.py`, or the dict that `save()` writes. The seed rebuilds the peg and hole geometry on reconfigure; `reset_to` checks that against the saved box height, which the task sets to the peg half length.
- `start_states/` holds Sut's 1000 train and 500 held-out states from `generate_start_states.py`.
- `data/interim_test.pt` holds 50 stand-in states for the tests, made by `scripts/make_interim_start_states.py`. Nothing reported uses them.

Classical controller on Sut's 500 held-out states (CPU sim, 200 steps):

| Hole offset | `success_once` (95% CI) | `success_at_end` | Mean steps to success |
|---|---|---|---|
| 0 mm | 0.990 [0.977, 0.996] | 0.972 | 34 |
| 2 mm | 0.992 [0.980, 0.997] | 0.974 | 35 |
| 4 mm | 0.976 [0.959, 0.986] | 0.950 | 44 |
| 6 mm | 0.968 [0.949, 0.980] | 0.934 | 50 |

## Cluster (ORCD Engaging)

```bash
ssh <kerb>@orcd-login.mit.edu
git clone git@github.com:zhizhen-kyle-luo/Robotics_Manipulation.git && cd Robotics_Manipulation
bash cluster/setup_engaging.sh                  # once, on the login node: uv env, Vulkan ICD
sbatch cluster/smoke.sbatch                     # read logs/peg-smoke-<jobid>.out before training anything
```

The smoke job defaults to the `mit_normal_gpu` partition; use `sbatch -p <partition>` for a lab's own GPUs.

What to read in the smoke log:
- **Hold test.** `grasped` should be close to 1. The states were generated on CPU physics and training runs on GPU physics; if the grasps do not survive the transfer, stop and regenerate the states on the GPU.
- **Classical on the GPU backend.** It should be close to the CPU table above. ManiSkill's GPU delta-pose controller is a one-step Jacobian solve rather than a converged IK, in the same frame and with the same sign.
- **`SPS`.** 75M steps / SPS is the run time.
- **`torch cannot see a GPU`.** The locked torch wheels are built for CUDA 13.2 and need NVIDIA driver 580 or newer; the log prints the node's driver version.
