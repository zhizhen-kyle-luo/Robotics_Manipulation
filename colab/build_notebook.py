"""Generates ppo_colab.ipynb. Edit the cells here and rerun rather than editing the notebook's JSON:

    python colab/build_notebook.py colab/ppo_colab.ipynb   (needs nbformat)
"""

import sys
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# PPO from scratch (3) on Colab

[Open in Colab](https://colab.research.google.com/github/zhizhen-kyle-luo/Robotics_Manipulation/blob/kevin/ppo-start-states/colab/ppo_colab.ipynb)

Runs PPO from scratch on `PegInsertionSide-v1` from Sut's grasped start states: a smoke test, a short tuning run, then the 75M-step runs (dense reward on seeds 1–3, and a one-seed sparse-reward sanity check). Everything goes to Google Drive, so a disconnect costs at most one checkpoint interval (~0.8M steps).

**Runtime:** the notebook uses a GPU when the runtime has one and falls back to CPU physics otherwise. The setup and smoke test run on either; training needs a GPU (Runtime → Change runtime type → GPU; an A100 or L4 is much faster than a T4).

**After a disconnect:** rerun sections 1 and 2, then rerun the cell that was running. It resumes from its newest checkpoint and skips runs that are already finished.

**Long-running cells (start these early):**

| Cell | Steps | Time |
|---|---|---|
| 4. Tuning | 2 × 10M | 20M / SPS |
| 6. Main runs | 4 × 75M (seed 1 continues the tuning run, so 65M more for it) | ~290M / SPS |
| 7. Final evaluation | 500 states × 4 offsets per run | ~minutes per run |

Section 3 prints SPS and turns it into hours."""))

cells.append(md("## 1. Settings, Google Drive and runtime"))
cells.append(code("""DRIVE_DIR = "/content/drive/MyDrive/6.4210_peg_ppo"  # runs, checkpoints, logs and results are kept here
BRANCH = "kevin/ppo-start-states"
REPO = "/content/Robotics_Manipulation"

from google.colab import drive
drive.mount("/content/drive")

import torch
GPU = torch.cuda.is_available()
# ManiSkill's GPU physics runs thousands of envs at once; its CPU physics runs one
SIM = "physx_cuda" if GPU else "physx_cpu"
print(f"runtime: {torch.cuda.get_device_name(0) if GPU else 'CPU only'} -> simulating with {SIM}")"""))

cells.append(md("""## 2. Setup (~3 min)

**Packages.** Colab's own torch is kept. ManiSkill 3.0.1 pins `mplib==0.1.1`, a motion-planning library with no build for Colab's Python, so ManiSkill is installed without its pins and its other dependencies are listed by hand. Nothing here uses mplib; Sut's branch overrides the same pin. CPU physics also needs `pin` (Pinocchio) for the arm's inverse kinematics.

**Vulkan.** SAPIEN, ManiSkill's simulator, opens a Vulkan device even when nothing is drawn. Without one it stops with `failed to find a rendering device`. The cell checks first and only acts if that check fails. On a GPU it registers Colab's own NVIDIA driver with Vulkan, using the files from ManiSkill's install docs and Colab quickstart. On a CPU it installs Mesa's software Vulkan driver.

Every step stops the notebook with the command's output if it fails, and every step is safe to repeat. If Colab asks to restart the session after installing, restart and run sections 1 and 2 again."""))
cells.append(code("""import json, os, pathlib, subprocess, sys

def sh(cmd):
    \"\"\"Run a shell command; stop with its output if it fails.\"\"\"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed: {cmd}\\n{result.stdout[-3000:]}\\n{result.stderr[-3000:]}")
    return result.stdout

print("installing packages...")
sh(f"{sys.executable} -m pip -q install --no-deps mani_skill==3.0.1")
sh(f"{sys.executable} -m pip -q install sapien==3.0.3 pytorch_kinematics==0.7.6 dacite gymnasium h5py trimesh transforms3d 'imageio[ffmpeg]' "
   "GitPython tabulate defusedxml nvidia-ml-py 'tyro>=0.8.5' huggingface_hub scipy tqdm pyyaml" + ("" if GPU else " pin"))
sh(f"{sys.executable} -c 'import mani_skill.envs, tyro, sapien'")

def vulkan_ok():
    result = subprocess.run([sys.executable, "-c", "import sapien; sapien.render.RenderSystem()"],
                            capture_output=True, text=True)
    return result.returncode == 0, result.stderr[-1500:]

ok, err = vulkan_ok()
if not ok and GPU:
    driver_files = {
        "/usr/share/vulkan/icd.d/nvidia_icd.json": {
            "file_format_version": "1.0.0",
            "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.2.155"}},
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json": {
            "file_format_version": "1.0.0",
            "ICD": {"library_path": "libEGL_nvidia.so.0"}},
        # needed on some GPUs, such as the A100, per ManiSkill's docs
        "/etc/vulkan/implicit_layer.d/nvidia_layers.json": {
            "file_format_version": "1.0.0",
            "layer": {
                "name": "VK_LAYER_NV_optimus", "type": "INSTANCE", "library_path": "libGLX_nvidia.so.0",
                "api_version": "1.2.155", "implementation_version": "1", "description": "NVIDIA Optimus layer",
                "functions": {"vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
                              "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"},
                "enable_environment": {"__NV_PRIME_RENDER_OFFLOAD": "1"},
                "disable_environment": {"DISABLE_LAYER_NV_OPTIMUS_1": ""}}},
    }
    for path, content in driver_files.items():
        if not os.path.exists(path):
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(path).write_text(json.dumps(content, indent=2))
    ok, err = vulkan_ok()
if not ok:
    sh("apt-get -qq update > /dev/null; apt-get -qq install -y --no-install-recommends libvulkan1"
       + ("" if GPU else " mesa-vulkan-drivers"))
    ok, err = vulkan_ok()
if not ok:
    raise RuntimeError("SAPIEN still cannot open a Vulkan device:\\n" + err)
print("Vulkan OK")

if not os.path.exists(REPO):
    sh(f"git clone -q -b {BRANCH} https://github.com/zhizhen-kyle-luo/Robotics_Manipulation.git {REPO}")
else:
    sh(f"git -C {REPO} pull -q --ff-only")
os.chdir(REPO)
sys.path.insert(0, REPO)
print(sh("bash scripts/fetch_start_states.sh 2>&1"))

# keep runs, logs and results on Drive so they survive a disconnect
for d in ["runs", "logs", "results"]:
    os.makedirs(f"{DRIVE_DIR}/{d}", exist_ok=True)
    if not os.path.islink(d):
        os.symlink(f"{DRIVE_DIR}/{d}", d)

import ppo_runs

# every PPO run trains on the 1000 train states and is scored on the 500 held-out states
STARTS = ["--starts", "start_states/train/states.pt", "--eval-starts", "start_states/eval/states.pt"]
if GPU:
    # ppo.py's defaults: 2048 train envs, 16 steps each per iteration
    STARTS += ["--sim-backend", "physx_cuda", "--num-eval-envs", "500"]
    BATCH, EVAL_ENVS = ppo_runs.BATCH_SIZE, 100
    print(sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv"))
else:
    # CPU physics simulates one env at a time: enough to check the pipeline, not to train
    STARTS += ["--sim-backend", "physx_cpu", "--num-envs", "1", "--num-eval-envs", "1", "--num-steps", "64",
               "--num-minibatches", "1"]
    BATCH, EVAL_ENVS = 64, 1
print("setup done")"""))

cells.append(md("""## 3. Smoke test (~15 min on a GPU)

Read before training anything:
- **hold test** (GPU only): `grasped` should be close to 1. Sut made the grasps on CPU physics; training runs on GPU physics. If they don't survive, stop: the states need regenerating on the GPU.
- **SPS** from `gpu_smoke.py` is pure simulation; the estimate at the end uses PPO's own SPS."""))
cells.append(code("""if GPU:
    !{sys.executable} scripts/gpu_smoke.py --starts start_states/eval/states.pt --num-envs 1024
else:
    print("CPU runtime: skipping the GPU hold test and throughput check")"""))
cells.append(md("""The classical controller on this runtime's physics (first 100 held-out states on a GPU, first 3 on a CPU). On CPU physics, all 500 states, it scored `success_once` 0.990 / 0.992 / 0.976 / 0.968 at 0 / 2 / 4 / 6 mm. GPU numbers far below that would mean the GPU controller behaves differently, which also matters for the Diffusion Policy base."""))
cells.append(code("""n = 100 if GPU else 3
!{sys.executable} evaluate.py --policy classical --starts start_states/eval/states.pt --limit {n} --num-envs {EVAL_ENVS} \\
    --sim-backend {SIM} --noise-mm 0 2 4 6 --out results/classical_{SIM}_{n}.json"""))
cells.append(md("PPO at full scale (on a GPU: 2048 train envs, 500 eval envs) for 4 iterations, then resumed to 6. The second call has to say `resuming from runs/smoke_ppo/final_ckpt.pt`."))
cells.append(code("""import re, shutil
shutil.rmtree("runs/smoke_ppo", ignore_errors=True)
ppo_runs.run_ppo("smoke_ppo", STARTS + ["--eval-freq", "2"], 4 * BATCH, sps_every=1, batch_size=BATCH)
ppo_runs.run_ppo("smoke_ppo", STARTS + ["--eval-freq", "2"], 6 * BATCH, sps_every=1, batch_size=BATCH)

sps = [int(x) for x in re.findall(r"^SPS: (\\d+)", open("logs/smoke_ppo.log").read(), re.M)]
print(f"\\nPPO SPS {sps[-1]:,} (pessimistic: the smoke run evaluates every 2 iterations, the real runs every 25)")
if GPU:
    for label, steps in [("tuning, 2 x 10M", 20e6), ("main runs, ~290M", 290e6)]:
        print(f"  {label}: {steps / sps[-1] / 3600:.1f} h")"""))

cells.append(md("""## 4. Tuning (long: 2 × 10M steps, GPU only)

ManiSkill's PPO baseline was tuned for joint-space control; here every method uses end-effector deltas, where the baseline's initial noise (log std −0.5) means about 6 cm per step. This compares it with less noise. Both use dense reward and seed 1, and the winner's run is reused as the first 10M steps of its 75M seed-1 run.

Forcing the gripper closed (`--gripper-closed`) is left out: on CPU, PPO learned to keep the peg within 100k steps by itself. Add `"closed": ["--gripper-closed"]` to `CONFIGS` to test it anyway."""))
cells.append(code("""TUNE_STEPS = 10_000_000
CONFIGS = {"base": [], "lowstd": ["--logstd-init", "-1.5"]}

def run_name(config, reward, seed):
    return f"ppo_{config}_{'dense' if reward == 'normalized_dense' else 'sparse'}_seed{seed}"

assert GPU, "training needs a GPU runtime: Runtime -> Change runtime type -> GPU"
for config, flags in CONFIGS.items():
    ppo_runs.run_ppo(run_name(config, "normalized_dense", 1),
                     STARTS + ["--seed", "1", "--reward-mode", "normalized_dense"] + flags, TUNE_STEPS)"""))

cells.append(md("""## 5. Compare and choose

Pick the config with the higher `eval success_once`. If both are still at 0 (likely this early: the ManiSkill paper reports PPO struggling on this task), pick the higher `eval return` and `train/return`. Set `CHOSEN` in the next section."""))
cells.append(code("""!{sys.executable} scripts/curves.py runs/ppo_*_dense_seed1
%load_ext tensorboard
%tensorboard --logdir runs"""))

cells.append(md("""## 6. Main runs (long: 75M steps each, GPU only)

Each cell is one run. Rerunning a cell resumes it, and a finished run is skipped. Seed 1 continues its tuning run. If Colab time runs short, the proposal's cut order is to drop to fewer PPO seeds, so run these top to bottom."""))
cells.append(code("""CHOSEN = "base"  # from section 5
TOTAL = 75_000_000

def main_run(reward, seed):
    assert GPU, "training needs a GPU runtime: Runtime -> Change runtime type -> GPU"
    ppo_runs.run_ppo(run_name(CHOSEN, reward, seed),
                     STARTS + ["--seed", str(seed), "--reward-mode", reward] + CONFIGS[CHOSEN], TOTAL)"""))
cells.append(code("main_run(\"normalized_dense\", 1)"))
cells.append(code("main_run(\"normalized_dense\", 2)"))
cells.append(code("main_run(\"normalized_dense\", 3)"))
cells.append(code("main_run(\"sparse\", 1)  # Kyle's sanity check: one seed with sparse reward"))

cells.append(md("""## 7. Final evaluation and learning curves

Scores each finished run's final checkpoint on all 500 held-out states at 0 / 2 / 4 / 6 mm, and the classical controller on the same physics so the comparison is like for like. Results and the learning-curve CSV land in `results/` on Drive."""))
cells.append(code("""for reward, seed in [("normalized_dense", 1), ("normalized_dense", 2), ("normalized_dense", 3), ("sparse", 1)]:
    name = run_name(CHOSEN, reward, seed)
    if not ppo_runs.finished(f"runs/{name}", TOTAL):
        print(f"{name}: not finished, skipped")
        continue
    kwargs = json.dumps({"checkpoint": f"runs/{name}/final_ckpt.pt"})
    print(name)
    !{sys.executable} evaluate.py --policy ppo_policy:make_policy --policy-kwargs '{kwargs}' --starts start_states/eval/states.pt \\
        --num-envs {EVAL_ENVS} --sim-backend {SIM} --out results/{name}.json

print("classical")
!{sys.executable} evaluate.py --policy classical --starts start_states/eval/states.pt --num-envs {EVAL_ENVS} --sim-backend {SIM} \\
    --out results/classical_{SIM}_500.json
!{sys.executable} scripts/curves.py runs/ppo_{CHOSEN}_* --csv results/ppo_curves.csv"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "A100"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
nbf.write(nb, sys.argv[1])
print("wrote", sys.argv[1], len(cells), "cells")
