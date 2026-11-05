"""Start or resume ppo.py runs from Python, for notebooks that disconnect (Colab).

run_ppo(name, flags, total_timesteps) resumes runs/<name> from its newest checkpoint, or returns at
once if it already reached total_timesteps. ppo.py's learning rate is constant, so continuing a
finished run to a larger total gives the same run as one launched with that total: a 10M tuning
run continues into a 75M run.
"""

import re
import subprocess
import sys
from pathlib import Path

import torch

BATCH_SIZE = 2048 * 16  # ppo.py's default num_envs * num_steps
# lines worth showing from ppo.py's output; everything goes to the log file
SHOW = ("eval_success_once", "eval_success_at_end", "eval_return", "resuming", "Traceback", "Error")


def checkpoint_iteration(path):
    """Iteration a checkpoint resumes at: in the name for ckpt_<i>.pt, stored inside for final_ckpt.pt."""
    match = re.fullmatch(r"ckpt_(\d+)\.pt", Path(path).name)
    if match:
        return int(match.group(1))
    return torch.load(path, map_location="cpu", weights_only=True)["iteration"]


def latest_checkpoint(run_dir):
    """Newest checkpoint that loads; a disconnect can leave the last write cut off."""
    paths = sorted(Path(run_dir).glob("*ckpt*.pt"), key=checkpoint_iteration_safe, reverse=True)
    for path in paths:
        try:
            torch.load(path, map_location="cpu", weights_only=True)
            return path
        except Exception:
            print(f"skipping unreadable checkpoint {path}")
    return None


def checkpoint_iteration_safe(path):
    try:
        return checkpoint_iteration(path)
    except Exception:
        return -1


def finished(run_dir, total_timesteps, batch_size=BATCH_SIZE):
    final = Path(run_dir) / "final_ckpt.pt"
    if not final.exists():
        return False
    try:
        done = torch.load(final, map_location="cpu", weights_only=True)["global_step"]
    except Exception:
        return False
    return done >= (total_timesteps // batch_size) * batch_size


def ppo_command(name, flags, total_timesteps, checkpoint=None):
    cmd = [sys.executable, "ppo.py", "--exp-name", name, "--total-timesteps", str(total_timesteps),
           "--no-capture-video", *flags]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    return cmd


def run_ppo(name, flags, total_timesteps, log_dir="logs", sps_every=50, batch_size=BATCH_SIZE):
    """Run ppo.py to total_timesteps, resuming if possible. Shows eval lines and SPS, logs everything."""
    run_dir = Path("runs") / name
    if finished(run_dir, total_timesteps, batch_size):
        print(f"{name}: already at {total_timesteps:,} steps")
        return
    checkpoint = latest_checkpoint(run_dir) if run_dir.exists() else None
    cmd = ppo_command(name, flags, total_timesteps, checkpoint)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{name}.log"
    print(f"{name}: {'resuming from ' + str(checkpoint) if checkpoint else 'starting'}; full log in {log_path}")
    epochs = 0
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if line.startswith("Epoch:"):
                epochs += 1
                last_epoch = line.strip()
            if line.startswith("SPS:") and epochs % sps_every == 0:
                print(f"  {last_epoch}  {line.strip()}", flush=True)
            elif line.startswith(SHOW):
                print("  " + line.strip(), flush=True)
        code = proc.wait()
    if code != 0:
        tail = log_path.read_text().splitlines()[-30:]
        raise RuntimeError(f"{name} exited with {code}; last lines of {log_path}:\n" + "\n".join(tail))
    print(f"{name}: done")
