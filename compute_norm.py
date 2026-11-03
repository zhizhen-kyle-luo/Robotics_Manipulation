"""Compute normalization statistics for the insertion-only demos.

Same layout and output format as openpi's scripts/compute_norm_stats.py: load the
dataset, accumulate per-key statistics, and save {key: {mean, std, q01, q99}} to
norm_stats.json. What changed:
  - Data comes from a ManiSkill trajectory .h5 file (traj_0, traj_1, ... each with
    "obs" and "actions"), not from openpi's config / LeRobot / RLDS loaders.
  - Statistics are computed exactly over all frames (100 demos fit in memory), so
    openpi's RunningStats and max_frames subsampling are not needed.
  - A minimum std floor, because insertion-only demos have near-constant dimensions
    (closed gripper, per-episode-constant peg_half_size and box_hole_radius).

Usage:
  python compute_norm_stats.py --demo-path path/to/insertion_only.state.pd_ee_delta_pose.h5 \
      --output-dir runs/norm_stats
"""

import json
import pathlib

import h5py
import numpy as np
import tyro


def load_frames(demo_path: str) -> dict[str, np.ndarray]:
    """Return {"state": (N, obs_dim), "actions": (N, act_dim)} stacked over all demos."""
    states = []
    actions = []
    with h5py.File(demo_path, "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        if len(traj_keys) == 0:
            raise ValueError(f"No traj_* groups found in {demo_path}")
        for traj_key in traj_keys:
            traj = f[traj_key]
            # ManiSkill stores obs as one flat array only when the demo was replayed with
            # obs mode "state". With "state_dict" it is a nested group, which we do not handle.
            if isinstance(traj["obs"], h5py.Dataset):
                obs = np.asarray(traj["obs"], dtype=np.float64)
            else:
                raise ValueError(
                    f"{traj_key}/obs is a group, not an array. Replay the demos with -o state "
                    "so the observation is the flat state vector the policy sees."
                )
            act = np.asarray(traj["actions"], dtype=np.float64)
            # ManiSkill saves T+1 observations for T actions (the final obs has no action).
            # Drop the last obs so states and actions are paired frame by frame.
            if obs.shape[0] == act.shape[0] + 1:
                obs = obs[:-1]
            elif obs.shape[0] != act.shape[0]:
                raise ValueError(
                    f"{traj_key}: {obs.shape[0]} obs vs {act.shape[0]} actions, expected T+1 vs T"
                )
            states.append(obs)
            actions.append(act)
    return {"state": np.concatenate(states, axis=0), "actions": np.concatenate(actions, axis=0)}


def compute_stats(x: np.ndarray, key: str, min_std: float) -> dict[str, list[float]]:
    """Per-dimension mean, std, 1st and 99th percentile, matching openpi's NormStats fields."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    # Assumption (ours, not openpi's): dimensions that barely vary in the demos, like the
    # closed gripper, would be divided by ~0 and blow up. Floor std so they stay bounded.
    small = std < min_std
    if small.any():
        print(f"[{key}] clamping std to {min_std} for dims {np.flatnonzero(small).tolist()}")
        std = np.where(small, min_std, std)
    else:
        print(f"[{key}] no dims below min_std")
    q01 = np.quantile(x, 0.01, axis=0)
    q99 = np.quantile(x, 0.99, axis=0)
    return {"mean": mean.tolist(), "std": std.tolist(), "q01": q01.tolist(), "q99": q99.tolist()}


def main(demo_path: str, output_dir: str, min_std: float = 1e-2, include_actions: bool = False):
    """
    Args:
        demo_path: insertion-only demo .h5 (state obs, pd_ee_delta_pose actions).
        output_dir: directory to write norm_stats.json into.
        min_std: std floor for near-constant dims. 1e-2 is our choice, not from a paper.
        include_actions: pd_ee_delta_pose actions are already in [-1, 1], so by default we
            only normalize the state. Set True to also save action stats (e.g. for the
            residual's base-action input).
    """
    frames = load_frames(demo_path)
    print(f"Loaded {frames['state'].shape[0]} frames, state dim {frames['state'].shape[1]}, "
          f"action dim {frames['actions'].shape[1]}")

    if include_actions:
        keys = ["state", "actions"]
    else:
        keys = ["state"]

    norm_stats = {}
    for key in keys:
        norm_stats[key] = compute_stats(frames[key], key, min_std)

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "norm_stats.json"
    print(f"Writing stats to: {out_file}")
    # Same top-level structure openpi's normalize.save writes: {"norm_stats": {key: {...}}}.
    with open(out_file, "w") as f:
        json.dump({"norm_stats": norm_stats}, f, indent=2)


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """z-score with the saved stats. Use the same function at train and eval time."""
    return (x - np.asarray(stats["mean"])) / np.asarray(stats["std"])


if __name__ == "__main__":
    tyro.cli(main)