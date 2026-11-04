# Robotics_Manipulation

Peg insertion with ManiSkill 3: pose-error contribution, motion-planned demos, grasped start states, and a
Diffusion Policy baseline on `PegInsertionSide-v1`.

## Requirements

### System

| requirement | tested with | notes |
|---|---|---|
| Linux | Ubuntu 22.04 | |
| NVIDIA GPU | RTX 4090 | training and evaluation use CUDA |
| NVIDIA driver | 580.95 | the torch 2.15 wheels are built for CUDA 13.2 and need a driver >= 580 |
| Vulkan | 1.3 / 1.4 | SAPIEN rendering; needed for videos (`vulkaninfo --summary` should list the GPU) |
| [uv](https://docs.astral.sh/uv/) | 0.11 | installs Python 3.11 and all packages |
| git | | `launch_dp.sh` refuses to train on uncommitted code |
| tmux | 3.2 | `launch_dp.sh` runs each seed in a tmux session |
| disk | | ~2 GB per Diffusion Policy training run (checkpoints every 5000 iterations) |

### Python (installed by uv)

Python 3.11 (`.python-version`), from `pyproject.toml` / `uv.lock`:

| package | version |
|---|---|
| mani-skill | 3.0.1 |
| sapien | 3.0.3 |
| gymnasium | 1.4.0 |
| torch | 2.15.0 (CUDA 13.2) |
| mplib | 0.2.1 (overrides mani-skill's `mplib==0.1.1` pin, which segfaults) |

Diffusion Policy additionally needs `diffusers`, `tensorboard` and `wandb`, installed from `diffusion_policy/`
(step 2 below).

## Setup

```bash
# 1. Python environment (.venv)
uv sync

# 2. Diffusion Policy package and its dependencies
uv pip install -e diffusion_policy

# 3. ManiSkill demos for PegInsertionSide-v1 (to ~/.maniskill/demos/)
uv run python -m mani_skill.utils.download_demo PegInsertionSide-v1
```

`uv sync` removes packages that are not in `uv.lock`, so rerun step 2 after every `uv sync`.

Check the install:

```bash
uv run python -c "import torch, mani_skill, diffusers; print(torch.cuda.is_available(), mani_skill.__version__)"
```

## Data

Not all data is in git. Expected locations:

| path | contents |
|---|---|
| `data/dp_training/` | Diffusion Policy demos (`trajectory.state.pd_ee_delta_pose.physx_cpu.h5`) and `norm_stats.json` |
| `start_states/train/`, `start_states/eval/` | grasped start states (`states.pt`) from `generate_start_states.py` |

## Run

```bash
uv run peg_insertion.py
```

Diffusion Policy training, evaluation, failure modes and plots: see [`diffusion_policy/README.md`](diffusion_policy/README.md).
