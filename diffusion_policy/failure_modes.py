"""Classify why a Diffusion Policy checkpoint fails from saved grasped start states.

Diagnostic companion to eval_start_states.py (same agent loading, start-state restore and chunked
execution; does not modify train.py or any checkpoint). Each episode is rerun while tracking the peg head
position in the hole frame (info["peg_head_pos_at_hole"]: x along the hole axis, 0 at the box mid-depth,
y/z lateral) and whether the peg is still grasped. Failed episodes get one failure_mode:

  dropped              peg not grasped at the end of the episode
  never_reached_face   peg head stayed > 30 mm short of the box face (wanders off / stalls)
  entered_too_shallow  peg head went > 5 mm past the box face into the hole, but never deep enough
  jammed               reached the face with lateral offset within the clearance, but did not go in
  misaligned           reached the face but never got within the clearance laterally (stuck at rim/face)

The box face is at x = -peg_half_length in the hole frame; clearance = hole radius - peg radius (3 mm).

Usage (from diffusion_policy/):
  python failure_modes.py --run runs/dp_full1000_s0
  python failure_modes.py --run runs/dp_full1000_s0 --ckpt best_eval_success_at_end.pt --num-states 100

Outputs (in --out-dir):
  <name>__failmodes.csv           one row per episode: state_id, seed, success_once, failure_mode, and the
                                  geometry behind it (min_lateral_mm, max_depth_past_face_mm, ...)
  <name>__failmodes_summary.json  failure-mode counts and fractions, config, git commit
  where <name> is the same as eval_start_states.py
"""

import csv
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import tyro

import train
from eval_start_states import (BOX_HOLE_POS, git_commit, load_agent, make_env, offset_vector, reset_to_state,
                               result_name)

MODES = ["misaligned", "entered_too_shallow", "jammed", "never_reached_face", "dropped"]
FACE_FAR_MM = 30.0  # further than this short of the face: never reached it
ENTERED_MM = 5.0  # further than this past the face: entered the hole


@dataclass
class FailArgs:
    run: str
    """training run directory, e.g. runs/dp_full1000_s0 (checkpoints/ is read from it)"""
    ckpt: str = "final.pt"
    """checkpoint file inside <run>/checkpoints/. EMA weights (ema_agent) are always used."""
    start_states: str = "../start_states/eval/states.pt"
    num_states: Optional[int] = None
    """classify only the first N start states (default: all)"""
    max_episode_steps: int = 200
    hole_offset_mm: float = 0.0
    """same as eval_start_states.py: shift the observed hole position in a random direction"""
    settle_steps: int = 0
    num_envs: int = 50
    seed: int = 0
    out_dir: str = "../results/dp_eval"
    control_mode: str = "pd_ee_delta_pose"
    sim_backend: str = "physx_cpu"
    env_id: str = "PegInsertionSide-v1"
    # result_name() expects this field
    full_task: Optional[int] = None


def classify(r):
    if r["success_once"]:
        return ""
    if not r["grasp_end"]:
        return "dropped"
    if r["max_depth_past_face_mm"] < -FACE_FAR_MM:
        return "never_reached_face"
    if r["max_depth_past_face_mm"] > ENTERED_MM:
        return "entered_too_shallow"
    if r["min_lateral_mm"] <= r["clearance_mm"]:
        return "jammed"
    return "misaligned"


def run_batch(agent, envs, obs, offsets, max_steps, device, act_horizon):
    """Like eval_start_states.run_batch, but also tracks peg-head-in-hole geometry and grasp."""
    n = len(envs)
    first_success = np.full(n, -1)
    grasp = np.ones(n, dtype=bool)
    lost_grasp = np.zeros(n, dtype=bool)
    min_lat = np.full(n, np.inf)
    max_x = np.full(n, -np.inf)
    t = 0
    obs = np.stack(obs).astype(np.float32)
    while t < max_steps:
        o = obs.copy()
        o[:, :, BOX_HOLE_POS] += offsets[:, None, :]
        with torch.no_grad():
            action_seq = agent.get_action(torch.from_numpy(o).to(device)).cpu().numpy()
        for i in range(action_seq.shape[1]):
            for j, env in enumerate(envs):
                ob, _, _, _, info = env.step(action_seq[j, i])
                obs[j] = ob
                p = np.asarray(info["peg_head_pos_at_hole"]).reshape(-1)
                min_lat[j] = min(min_lat[j], np.hypot(p[1], p[2]))
                max_x[j] = max(max_x[j], p[0])
                if bool(info["success"]) and first_success[j] < 0:
                    first_success[j] = t + 1
                base = env.unwrapped
                grasp[j] = bool(base.agent.is_grasping(base.peg).item())
                # is_grasping reads False right after a state restore, so ignore the first steps
                if t > 5 and not grasp[j]:
                    lost_grasp[j] = True
            t += 1
            if t >= max_steps:
                break
    rows = []
    for j, env in enumerate(envs):
        base = env.unwrapped
        peg_half_len = base.peg_half_sizes[0, 0].item()
        hole_r, peg_r = base.box_hole_radii[0].item(), base.peg_half_sizes[0, 1].item()
        rows.append(dict(success_once=int(first_success[j] >= 0), steps_to_first_success=int(first_success[j]),
                         grasp_end=int(grasp[j]), lost_grasp=int(lost_grasp[j]),
                         min_lateral_mm=round(1000 * min_lat[j], 2),
                         max_depth_past_face_mm=round(1000 * (max_x[j] + peg_half_len), 2),
                         clearance_mm=round(1000 * (hole_r - peg_r), 2),
                         hole_r_mm=round(1000 * hole_r, 2), peg_r_mm=round(1000 * peg_r, 2)))
    return rows


def main(args: FailArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    targs = train.Args()
    obs_horizon, act_horizon = targs.obs_horizon, targs.act_horizon
    ckpt_path = os.path.join(args.run, "checkpoints", args.ckpt)
    agent, ckpt_iter = load_agent(ckpt_path, obs_horizon, 7, device)
    print(f"Loaded {ckpt_path} (iteration {ckpt_iter}), EMA weights")

    records = torch.load(args.start_states, weights_only=False)
    if args.num_states is not None:
        records = records[:args.num_states]
    rng = np.random.default_rng(args.seed)
    envs = [make_env(args, args.max_episode_steps, obs_horizon) for _ in range(min(args.num_envs, len(records)))]
    rows, t0 = [], time.time()
    for b in range(0, len(records), args.num_envs):
        batch = records[b:b + args.num_envs]
        obs = [reset_to_state(env, rec, args.settle_steps) for env, rec in zip(envs, batch)]
        offsets = np.array([offset_vector(rng, args.hole_offset_mm) for _ in batch])
        res = run_batch(agent, envs[:len(batch)], obs, offsets, args.max_episode_steps, device, act_horizon)
        for k, (rec, r) in enumerate(zip(batch, res)):
            rows.append(dict(state_id=b + k, seed=rec["seed"], **r))
            rows[-1]["failure_mode"] = classify(rows[-1])
        print(f"  {len(rows)}/{len(records)} episodes ({time.time() - t0:.0f}s)", flush=True)
    for env in envs:
        env.close()

    _, name = result_name(args)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{name}__failmodes.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fails = [r for r in rows if not r["success_once"]]
    counts = Counter(r["failure_mode"] for r in fails)
    mis = [r["min_lateral_mm"] for r in fails if r["failure_mode"] == "misaligned"]
    commit, dirty = git_commit()
    summary = dict(
        n_episodes=len(rows), n_success=len(rows) - len(fails), n_failures=len(fails),
        failure_modes={m: dict(count=counts[m], fraction_of_failures=counts[m] / len(fails) if fails else None)
                       for m in MODES},
        misaligned_min_lateral_mm=dict(median=float(np.median(mis)), p10=float(np.percentile(mis, 10)),
                                       p90=float(np.percentile(mis, 90))) if mis else None,
        lost_grasp_any_time=sum(r["lost_grasp"] for r in rows),
        thresholds=dict(face_far_mm=FACE_FAR_MM, entered_mm=ENTERED_MM, jammed="min_lateral_mm <= clearance_mm"),
        config=dict(**asdict(args), ckpt_iteration=ckpt_iter, git_commit=commit, git_dirty=dirty,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")))
    with open(os.path.join(args.out_dir, f"{name}__failmodes_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{name}: {len(rows) - len(fails)}/{len(rows)} succeeded, {len(fails)} failures -> {csv_path}")
    for m in MODES:
        if counts[m]:
            print(f"  {m:20s} {counts[m]:4d}  ({100 * counts[m] / len(fails):.0f}% of failures)")
    if mis:
        print(f"  misaligned: median closest lateral offset {np.median(mis):.1f} mm "
              f"(clearance {np.median([r['clearance_mm'] for r in fails]):.1f} mm)")


if __name__ == "__main__":
    main(tyro.cli(FailArgs))
