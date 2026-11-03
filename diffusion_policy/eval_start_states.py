"""Evaluate a trained Diffusion Policy checkpoint from saved grasped start states (or default full-task resets).

Shared evaluation script: does not modify train.py or any checkpoint. It reuses train.Agent and runs the
policy exactly like training-time eval (FrameStack obs_horizon history, CPUGymWrapper with
ignore_terminations, act_horizon actions executed per predicted chunk, EMA weights).

Start-state restore (see reset_to_state):
  PegInsertionSide samples peg/hole geometry from the seed at reconfiguration, and set_state_dict only
  restores poses/velocities. So each start state is restored with env.reset(seed=record["seed"]) on an env
  with reconfiguration_freq=1 (rebuilds the identical geometry), then set_state_dict(record["state_dict"]),
  then the FrameStack history is refilled with copies of the restored observation (otherwise the first
  policy input would contain the pre-grasp reset frame).

Usage (from diffusion_policy/):
  # grasped start states, final EMA weights
  python eval_start_states.py --run runs/dp_full136_s0 --ckpt final.pt \
      --start-states ../start_states/eval/states.pt --max-episode-steps 200 --out-dir ../results/dp_eval
  # sanity check: 50 default full-task resets with the training-eval horizon
  python eval_start_states.py --run runs/dp_full136_s0 --full-task 50 --max-episode-steps 300
  # record videos of selected episodes (picked from a previous results csv: successes and failures)
  python eval_start_states.py --run runs/dp_full136_s0 --num-videos 10

Outputs (in --out-dir):
  <name>.csv           one row per episode: state_id, seed, success_once, success_at_end,
                       steps_to_first_success (-1 if never), episode_len, hole_offset_mm
  <name>_summary.json  rates with 95% Wilson intervals, steps-to-success stats, config, git commit
  videos/<run>/        with --num-videos
  where <name> = <run>[__<ckpt tag> if not final][__fulltask][__hole<mm>mm if offset != 0]
"""

import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from gymnasium import spaces

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode

import train

# flat "state" observation layout of PegInsertionSide-v1 (43 dims):
# qpos 0:9, qvel 9:18, tcp_pose 18:25, peg_pose 25:32, peg_half_size 32:35, box_hole_pose 35:42, box_hole_radius 42
OBS_DIM = 43
BOX_HOLE_POS = slice(35, 38)


@dataclass
class EvalArgs:
    run: str
    """training run directory, e.g. runs/dp_full136_s0 (checkpoints/ is read from it)"""
    ckpt: str = "final.pt"
    """checkpoint file inside <run>/checkpoints/. EMA weights (ema_agent) are always used."""
    start_states: str = "../start_states/eval/states.pt"
    """list of start-state records (seed, state_dict, ...) saved by generate_start_states.py"""
    num_states: Optional[int] = None
    """evaluate only the first N start states (default: all)"""
    full_task: Optional[int] = None
    """instead of start states, run this many default full-task resets (seeds full_task_seed_base + i)"""
    full_task_seed_base: int = 2_000_000
    """seeds for --full-task; disjoint from demo/train (< 1e6) and held-out start-state (1e6..) seeds"""
    max_episode_steps: int = 200
    """episode length. 200 for insertion-only starts (~2.3x the demos' mean 86 steps close->success);
    use 300 for --full-task to match training-time eval"""
    hole_offset_mm: float = 0.0
    """shift the OBSERVED box_hole_pose position by this many mm in a random 3D direction (fixed per
    episode), applied to the raw observation before normalization. 0 = no perturbation"""
    settle_steps: int = 0
    """steps with zero arm action and closed gripper after restoring a start state, before the policy
    acts (not counted in episode length). 0 = policy acts directly from the restored state"""
    num_envs: int = 50
    """envs stepped in lockstep in this process; policy calls are batched over them"""
    num_videos: int = 0
    """record this many episodes to <out-dir>/videos/<run>/ instead of a full evaluation. Episodes are
    picked from the existing results csv (half successes, half failures where possible)"""
    seed: int = 0
    """seed for policy sampling noise and hole-offset directions"""
    out_dir: str = "../results/dp_eval"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    env_id: str = "PegInsertionSide-v1"


def load_agent(ckpt_path, obs_horizon, act_dim, device):
    """Build train.Agent and load the EMA weights; norm stats come from the checkpoint buffers."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["ema_agent"]
    obs_dim = OBS_DIM if "obs_mean" not in sd else sd["obs_mean"].shape[0]
    # Agent only reads the space shapes from env
    fake_env = type("Spaces", (), dict(
        single_observation_space=spaces.Box(-np.inf, np.inf, (obs_horizon, obs_dim), np.float32),
        single_action_space=spaces.Box(-1, 1, (act_dim,), np.float32)))()
    # train.Args defaults are the architecture used for training; a mismatch fails the strict load below
    agent = train.Agent(fake_env, train.Args(obs_horizon=obs_horizon))
    if "obs_mean" in sd:
        agent.normalize_obs = True
        agent.register_buffer("obs_mean", torch.zeros_like(sd["obs_mean"]))
        agent.register_buffer("obs_std", torch.ones_like(sd["obs_std"]))
    agent.load_state_dict(sd)
    agent.to(device).eval()
    return agent, ckpt.get("iteration")


def make_env(args, max_episode_steps, obs_horizon, video_dir=None):
    env = gym.make(args.env_id, reconfiguration_freq=1, control_mode=args.control_mode, reward_mode="sparse",
                   obs_mode="state", sim_backend=args.sim_backend, max_episode_steps=max_episode_steps,
                   render_mode="rgb_array" if video_dir else None,
                   human_render_camera_configs=dict(shader_pack="default"))
    if video_dir:
        env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True,
                            save_on_reset=False, source_type="diffusion_policy",
                            source_desc="diffusion_policy start-state evaluation")
    env = FrameStack(env, num_stack=obs_horizon)
    env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
    return env


def find_wrapper(env, cls):
    while env is not None:
        if isinstance(env, cls):
            return env
        env = getattr(env, "env", None)
    return None


def to_np(x):
    x = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return x


def reset_to_state(env, record, settle_steps=0):
    """Reset with the record's seed (rebuilds geometry), restore its state_dict, and refill the FrameStack
    history with copies of the restored observation. Returns the stacked obs (obs_horizon, obs_dim)."""
    env.reset(seed=record["seed"])
    base = env.unwrapped
    base.set_state_dict(record["state_dict"])
    for _ in range(settle_steps):
        # step the base env directly so TimeLimit / metrics / FrameStack do not count it
        base.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
    raw = base.get_obs()  # (1, obs_dim)
    fs = find_wrapper(env, FrameStack)
    fs.frames.clear()
    for _ in range(fs.num_stack):
        fs.frames.append(raw)
    rec = find_wrapper(env, RecordEpisode)
    if rec is not None:
        # replace the reset frame (pre-grasp scene) with the restored state
        rec.render_images = [rec.capture_image()]
    return to_np(fs.observation(None))[0]


def offset_vector(rng, mm):
    if mm == 0:
        return np.zeros(3)
    d = rng.normal(size=3)
    return d / np.linalg.norm(d) * mm / 1000.0


def run_batch(agent, envs, obs, offsets, max_steps, device, act_horizon):
    """Step all envs in lockstep for max_steps, executing act_horizon actions per chunk like evaluate.py.
    Returns per-env dicts with success_once, success_at_end, steps_to_first_success, episode_len."""
    n = len(envs)
    first_success = np.full(n, -1)
    success = np.zeros(n, dtype=bool)
    t = 0
    obs = np.stack(obs).astype(np.float32)  # (n, obs_horizon, obs_dim)
    while t < max_steps:
        o = obs.copy()
        o[:, :, BOX_HOLE_POS] += offsets[:, None, :]  # perturb the observed hole position only
        with torch.no_grad():
            action_seq = agent.get_action(torch.from_numpy(o).to(device)).cpu().numpy()
        for i in range(action_seq.shape[1]):
            for j, env in enumerate(envs):
                ob, _, _, trunc, info = env.step(action_seq[j, i])
                obs[j] = ob
                success[j] = bool(info["success"])
                if success[j] and first_success[j] < 0:
                    first_success[j] = t + 1
            t += 1
            if t >= max_steps:
                break
    return [dict(success_once=int(first_success[j] >= 0), success_at_end=int(success[j]),
                 steps_to_first_success=int(first_success[j]), episode_len=t) for j in range(n)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def git_commit():
    d = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=d, capture_output=True,
                           text=True).stdout.strip() != ""
    return out or "unknown", dirty


def result_name(args):
    run = os.path.basename(os.path.normpath(args.run))
    name = run
    if args.ckpt != "final.pt":
        name += "__" + os.path.splitext(args.ckpt)[0]
    if args.full_task:
        name += "__fulltask"
    if args.hole_offset_mm:
        name += f"__hole{args.hole_offset_mm:g}mm"
    return run, name


def episodes_to_run(args):
    """List of (state_id, seed, record or None)."""
    if args.full_task:
        return [(i, args.full_task_seed_base + i, None) for i in range(args.full_task)]
    records = torch.load(args.start_states, weights_only=False)
    if args.num_states is not None:
        records = records[:args.num_states]
    return [(i, r["seed"], r) for i, r in enumerate(records)]


def evaluate(args, agent, device, obs_horizon, act_horizon, episodes, video_dir=None, video_names=None):
    rng = np.random.default_rng(args.seed)
    rows = []
    batch = 1 if video_dir else args.num_envs
    envs = [make_env(args, args.max_episode_steps, obs_horizon, video_dir) for _ in range(min(batch, len(episodes)))]
    t0 = time.time()
    for b in range(0, len(episodes), batch):
        eps = episodes[b:b + batch]
        obs, offsets = [], []
        for env, (sid, seed, rec) in zip(envs, eps):
            if rec is None:
                ob, _ = env.reset(seed=seed)
                obs.append(to_np(ob))
            else:
                obs.append(reset_to_state(env, rec, args.settle_steps))
            offsets.append(offset_vector(rng, args.hole_offset_mm))
        res = run_batch(agent, envs[:len(eps)], obs, np.array(offsets), args.max_episode_steps, device, act_horizon)
        for (sid, seed, _), r, off in zip(eps, res, offsets):
            rows.append(dict(state_id=sid, seed=seed, **r, hole_offset_mm=args.hole_offset_mm,
                             hole_offset_dir=" ".join(f"{v:.4f}" for v in (off / max(np.linalg.norm(off), 1e-12)))
                             if args.hole_offset_mm else ""))
        if video_dir:
            rec = find_wrapper(envs[0], RecordEpisode)
            r = rows[-1]
            outcome = "success" if r["success_once"] else "fail"
            rec.flush_video(name=f"{video_names[b]}_{outcome}")
        done = len(rows)
        print(f"  {done}/{len(episodes)} episodes, success_once so far {np.mean([r['success_once'] for r in rows]):.3f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    for env in envs:
        env.close()
    return rows


def summarize(rows, args, ckpt_iter):
    n = len(rows)
    out = dict(n_episodes=n)
    for k in ["success_once", "success_at_end"]:
        s = sum(r[k] for r in rows)
        lo, hi = wilson(s, n)
        out[k] = dict(rate=s / n, successes=s, wilson95=[lo, hi])
    steps = [r["steps_to_first_success"] for r in rows if r["steps_to_first_success"] >= 0]
    out["steps_to_first_success"] = dict(
        mean=float(np.mean(steps)) if steps else None, median=float(np.median(steps)) if steps else None,
        min=int(min(steps)) if steps else None, max=int(max(steps)) if steps else None)
    commit, dirty = git_commit()
    out["config"] = dict(**asdict(args), ckpt_iteration=ckpt_iter, git_commit=commit, git_dirty=dirty,
                         mode="full_task" if args.full_task else "start_states",
                         timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


def main(args: EvalArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    targs = train.Args()
    obs_horizon, act_horizon = targs.obs_horizon, targs.act_horizon
    ckpt_path = os.path.join(args.run, "checkpoints", args.ckpt)
    agent, ckpt_iter = load_agent(ckpt_path, obs_horizon, 7, device)
    print(f"Loaded {ckpt_path} (iteration {ckpt_iter}), EMA weights, obs normalization: {agent.normalize_obs}")

    run, name = result_name(args)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{name}.csv")
    episodes = episodes_to_run(args)

    if args.num_videos:
        # pick episodes from the existing results: half successes, half failures where possible
        assert os.path.exists(csv_path), f"run the evaluation first, {csv_path} is needed to pick episodes"
        with open(csv_path) as f:
            prev = list(csv.DictReader(f))
        succ = [int(r["state_id"]) for r in prev if r["success_once"] == "1"]
        fail = [int(r["state_id"]) for r in prev if r["success_once"] == "0"]
        k_s = min(len(succ), args.num_videos // 2)
        ids = succ[:k_s] + fail[:args.num_videos - k_s]
        ids += succ[k_s:k_s + args.num_videos - len(ids)]
        chosen = [episodes[i] for i in ids]
        video_dir = os.path.join(args.out_dir, "videos", name)
        prev_by_id = {int(r["state_id"]): r for r in prev}
        names = [f"state{sid:03d}_seed{seed}_eval{'S' if prev_by_id[sid]['success_once'] == '1' else 'F'}"
                 for sid, seed, _ in chosen]
        print(f"Recording {len(chosen)} episodes ({k_s} were successes in the evaluation) to {video_dir}")
        rows = evaluate(args, agent, device, obs_horizon, act_horizon, chosen, video_dir, names)
        with open(os.path.join(video_dir, "episodes.json"), "w") as f:
            json.dump(dict(zip(names, rows)), f, indent=2)
        return

    print(f"Evaluating {len(episodes)} {'full-task resets' if args.full_task else 'start states'}, "
          f"max_episode_steps {args.max_episode_steps}, hole offset {args.hole_offset_mm} mm")
    rows = evaluate(args, agent, device, obs_horizon, act_horizon, episodes)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, args, ckpt_iter)
    with open(os.path.join(args.out_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    so, se = summary["success_once"], summary["success_at_end"]
    print(f"{name}: success_once {so['rate']:.3f} [{so['wilson95'][0]:.3f}, {so['wilson95'][1]:.3f}], "
          f"success_at_end {se['rate']:.3f} [{se['wilson95'][0]:.3f}, {se['wilson95'][1]:.3f}] -> {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
