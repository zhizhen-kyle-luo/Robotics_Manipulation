"""Phase 3/4: reproducible start-state generation for RL training/evaluation.

Runs peg_insertion_planner.solve_episode(..., stop_after_grasp=True) across a
range of seeds, keeps only states where the grasp is physically verified
(see solve_episode's own verification: is_grasping() right after closing,
is_grasping() again after a short hold, bounded peg-TCP drift, finite
poses), and saves them as reusable RL start states.

What gets saved, and why (see verify_start_states.py for the empirical
checks behind this):
  - `seed`: NOT sufficient on its own to reproduce the grasped state. This
    task reconfigures its scene on every reset() (num_envs=1 ->
    reconfiguration_freq=1), and peg/hole *dimensions* (peg_half_sizes,
    box_hole_offsets, box_hole_radii) are sampled from the seed's RNG only
    at that reconfiguration -- they are not part of get_state_dict() at
    all (baked into actor construction, not pose/velocity state). So the
    seed IS still required: it is what makes a later `env.reset(seed=...)`
    rebuild the identical peg/hole geometry before state_dict is applied.
  - `state_dict` (BaseEnv.get_state_dict()): actor/articulation
    pose+velocity and controller state. Verified empirically to exactly
    reproduce peg pose, TCP pose, and robot qpos when applied after a
    matching reset(seed=...) on a completely separate fresh env instance
    (max abs diff 0.0 across 7 compared quantities).
  - What state_dict/set_state_dict do NOT capture: contact-force /
    is_grasping() state. That is a derived, per-physics-step quantity, not
    saved simulator state -- verified empirically that is_grasping()
    reports False immediately after set_state_dict() even though every
    pose/qpos value is bit-exact, and recovers to True after exactly one
    further env.step() with the gripper still commanded closed. Restoring
    code (verify_start_states.py) always does that one settle-step before
    trusting is_grasping() on a restored state.
  - There is no separate "attachment" object to save: this task's grasp is
    a real dynamic friction grasp (verified in test_grasp_state.py), not a
    kinematic attachment, so the full physical state IS the peg pose +
    robot qpos already saved -- nothing else represents "holding" it.

Seed allocation (train vs eval are never allowed to overlap):
  - Train seeds walk upward from TRAIN_SEED_BASE (0).
  - Eval seeds walk upward from EVAL_SEED_BASE (1,000,000) -- chosen with a
    large safety margin over any plausible training pool size.
  - Before generating, a static assertion checks the configured
    (count * max_span_multiplier) for train cannot reach EVAL_SEED_BASE;
    generation refuses to start rather than risk crossing into eval's
    reserved range.
  - Retry policy: solve_episode's moves are deterministic given a seed (no
    RRT/randomized sampling is used -- only plan_screw), so re-attempting
    a failed seed would not change its outcome. A failed seed is therefore
    recorded once and skipped permanently; the walk simply continues to
    the next seed in that split's range until `target_count` accepted
    states are collected or `max_span_multiplier * target_count` seeds
    have been attempted (in which case generation stops and reports the
    shortfall explicitly rather than looping or guessing).

Output layout (under --out-dir, default "start_states/"):
  <out-dir>/<split>/states.pt              -- list of accepted state records
  <out-dir>/<split>/manifest.jsonl         -- one line per ATTEMPTED seed
  <out-dir>/<split>/generation_config.json -- args + software versions used

By default, an existing states.pt for a split is never silently overwritten
(pass --overwrite to allow it) -- this is the main guard against a training
run accidentally clobbering the held-out eval set.
"""
import argparse
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch

from peg_insertion_planner import ENV_ID, PLANNING_MAX_EPISODE_STEPS, solve_episode

TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 1_000_000


def _to_cpu(obj):
    """Recursively move a (possibly nested dict/list of) tensors to CPU, so
    saved states load on a machine without the GPU they were generated on."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _software_versions() -> dict:
    import mani_skill
    import mplib
    import sapien

    return dict(
        python=sys.version,
        mani_skill=mani_skill.__version__,
        sapien=sapien.__version__,
        mplib=mplib.__version__,
        gymnasium=gym.__version__,
        torch=torch.__version__,
    )


def generate_split(env, split: str, seed_base: int, target_count: int,
                    max_span_multiplier: int, manifest_path: str,
                    progress_every: int = 25):
    accepted = []
    attempted = 0
    max_span = target_count * max_span_multiplier
    seed = seed_base
    t0 = time.time()

    with open(manifest_path, "a") as manifest_f:
        while len(accepted) < target_count and (seed - seed_base) < max_span:
            attempted += 1
            try:
                result = solve_episode(env, seed=seed, debug=False, vis=False, stop_after_grasp=True)
                crashed = False
            except Exception as exc:  # noqa: BLE001 -- a crash is a documented failure, not a hard stop
                result = {"failure_reason": f"crash: {exc!r}", "physically_grasped": False}
                crashed = True

            accepted_now = (not crashed) and bool(result.get("physically_grasped"))
            manifest_entry = dict(
                seed=seed, split=split, attempt_index=attempted,
                status="accepted" if accepted_now else "failed",
                failure_reason=result.get("failure_reason"),
                is_grasping=result.get("is_grasping"),
                is_grasping_after_hold=result.get("is_grasping_after_hold"),
                grasp_drift_m=result.get("grasp_drift_m"),
            )
            manifest_f.write(json.dumps(manifest_entry) + "\n")
            manifest_f.flush()

            if accepted_now:
                base_env = env.unwrapped
                accepted.append(dict(
                    seed=seed, split=split, env_id=ENV_ID,
                    state_dict=_to_cpu(base_env.get_state_dict()),
                    peg_pose=dict(p=result["peg_pose"]["p"], q=result["peg_pose"]["q"]),
                    tcp_pose=dict(p=result["tcp_pose"]["p"], q=result["tcp_pose"]["q"]),
                    is_grasping=result["is_grasping"],
                    is_grasping_after_hold=result["is_grasping_after_hold"],
                    grasp_drift_m=result["grasp_drift_m"],
                ))
                if len(accepted) % progress_every == 0 or len(accepted) == target_count:
                    elapsed = time.time() - t0
                    print(f"[{split}] accepted {len(accepted)}/{target_count} "
                          f"(attempted {attempted}, seed={seed}, {elapsed:.1f}s elapsed)")
            seed += 1

    return accepted, attempted


def run_split(args, split: str, seed_base: int, target_count: int, out_root: str):
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)
    states_path = os.path.join(split_dir, "states.pt")
    manifest_path = os.path.join(split_dir, "manifest.jsonl")
    config_path = os.path.join(split_dir, "generation_config.json")

    if os.path.exists(states_path) and not args.overwrite:
        print(f"[{split}] {states_path} already exists -- skipping "
              f"(pass --overwrite to regenerate and replace it).")
        return

    # Start each split's manifest fresh on a real (re)generation -- the
    # states.pt existence check above is what actually prevents accidental
    # overwrites; once we get here we are deliberately (re)generating.
    open(manifest_path, "w").close()

    env = gym.make(
        ENV_ID, obs_mode="none", control_mode="pd_joint_pos", reward_mode="dense",
        render_mode=None, max_episode_steps=PLANNING_MAX_EPISODE_STEPS,
    )
    try:
        accepted, attempted = generate_split(
            env, split, seed_base, target_count, args.max_span_multiplier, manifest_path,
        )
    finally:
        env.close()

    torch.save(accepted, states_path)
    with open(config_path, "w") as f:
        json.dump(dict(
            split=split, env_id=ENV_ID, seed_base=seed_base, target_count=target_count,
            max_span_multiplier=args.max_span_multiplier, attempted=attempted,
            accepted=len(accepted), max_episode_steps=PLANNING_MAX_EPISODE_STEPS,
            control_mode="pd_joint_pos", timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            software_versions=_software_versions(),
        ), f, indent=2)

    print(f"[{split}] done: {len(accepted)}/{target_count} accepted "
          f"after {attempted} attempts ({len(accepted)/attempted:.1%} yield). "
          f"Saved to {states_path}")
    if len(accepted) < target_count:
        print(f"[{split}] WARNING: short by {target_count - len(accepted)} states "
              f"-- exhausted the {args.max_span_multiplier}x seed-span budget "
              f"({attempted} attempts). See {manifest_path} for per-seed failure reasons.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--num-train", type=int, default=1000,
                         help="training-pool size (configurable, no hard-coded count)")
    parser.add_argument("--num-eval", type=int, default=500,
                         help="held-out evaluation-pool size (brief default: exactly 500)")
    parser.add_argument("--max-span-multiplier", type=int, default=5,
                         help="give up on a split after target_count * this many seed attempts")
    parser.add_argument("--out-dir", type=str, default="start_states")
    parser.add_argument("--overwrite", action="store_true",
                         help="allow regenerating (overwriting) a split that already has states.pt")
    args = parser.parse_args()

    # Non-overlap guard, checked BEFORE any generation starts: the highest
    # seed the training walk could ever reach must stay below EVAL_SEED_BASE.
    train_worst_case_max_seed = TRAIN_SEED_BASE + args.num_train * args.max_span_multiplier
    if train_worst_case_max_seed >= EVAL_SEED_BASE:
        print(
            f"Refusing to start: training could attempt seeds up to "
            f"{train_worst_case_max_seed} (num_train={args.num_train} * "
            f"max_span_multiplier={args.max_span_multiplier}), which reaches "
            f"EVAL_SEED_BASE={EVAL_SEED_BASE}. Reduce --num-train / "
            f"--max-span-multiplier, or raise EVAL_SEED_BASE in this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.split in ("eval", "both"):
        run_split(args, "eval", EVAL_SEED_BASE, args.num_eval, args.out_dir)
    if args.split in ("train", "both"):
        run_split(args, "train", TRAIN_SEED_BASE, args.num_train, args.out_dir)


if __name__ == "__main__":
    main()
