"""Checks for the observation normalization added to train.py.

Run from this directory: python test_normalization.py [--ckpt runs/<exp>/checkpoints/<tag>.pt]
  1. demo observations normalized with norm_stats.json have mean ~0, std ~1 per dim
  2. Agent mean/std buffers survive a save / reload of the state_dict (and the EMA agent gets them too)
  3. without --norm-stats-path the Agent has no extra buffers (original baseline)
  4. optionally: a checkpoint written by train.py holds the same buffers as norm_stats.json
"""

import argparse
import json
import os
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from gymnasium import spaces

import train

REPO = os.path.expanduser("~/Robotics_Manipulation")
DEMO_PATH = f"{REPO}/data/dp_training/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
NORM_STATS_PATH = f"{REPO}/data/dp_training/norm_stats.json"
OBS_DIM, ACT_DIM, OBS_HORIZON = 43, 7, 2


def load_stats(path=NORM_STATS_PATH):
    with open(path) as f:
        s = json.load(f)["norm_stats"]["state"]
    return np.asarray(s["mean"]), np.asarray(s["std"])


def make_agent(norm_stats_path):
    # stand-in for the vectorized eval env: Agent only reads the space shapes
    env = SimpleNamespace(
        single_observation_space=spaces.Box(-np.inf, np.inf, (OBS_HORIZON, OBS_DIM), np.float32),
        single_action_space=spaces.Box(-1, 1, (ACT_DIM,), np.float32),
    )
    args = train.Args(norm_stats_path=norm_stats_path, obs_horizon=OBS_HORIZON)
    return train.Agent(env, args)


def test_normalized_demo_obs():
    mean, std = load_stats()
    # same frames compute_norm.py used: all obs except the terminal one of each episode
    with h5py.File(DEMO_PATH, "r") as f:
        obs = np.concatenate([f[k]["obs"][:-1] for k in f if k.startswith("traj_")]).astype(np.float64)
    assert obs.shape[1] == OBS_DIM

    z = (obs - mean) / std
    z_mean, z_std = z.mean(0), z.std(0)
    clamped = np.isclose(std, 1e-2) & (obs.std(0) < 1e-2)
    print(f"clamped dims (std floored to 1e-2): {np.flatnonzero(clamped).tolist()}")
    assert np.abs(z_mean).max() < 1e-6, f"max |mean| {np.abs(z_mean).max()}"
    assert np.allclose(z_std[~clamped], 1.0, atol=1e-6), f"std of unclamped dims: {z_std[~clamped]}"
    # near-constant dims stay bounded instead of exploding
    assert (z_std[clamped] <= 1.0).all()
    print(f"normalized obs: max |mean| {np.abs(z_mean).max():.2e}, "
          f"unclamped std in [{z_std[~clamped].min():.6f}, {z_std[~clamped].max():.6f}], "
          f"clamped std max {z_std[clamped].max():.3f}")

    # the Agent's normalize gives the same result as the numpy reference
    agent = make_agent(NORM_STATS_PATH)
    x = torch.tensor(obs[:64].reshape(32, OBS_HORIZON, OBS_DIM), dtype=torch.float32)
    ref = z[:64].reshape(32, OBS_HORIZON, OBS_DIM)
    assert np.allclose(agent.normalize(x).numpy(), ref, atol=1e-4)


def test_buffers_roundtrip():
    mean, std = load_stats()
    agent = make_agent(NORM_STATS_PATH)
    assert "obs_mean" in agent.state_dict() and "obs_std" in agent.state_dict()

    with tempfile.TemporaryDirectory() as d:
        # a fresh agent built from *different* stats must end up with the saved ones after loading
        other_stats = f"{d}/other.json"
        with open(other_stats, "w") as f:
            json.dump({"norm_stats": {"state": {"mean": [0.0] * OBS_DIM, "std": [1.0] * OBS_DIM}}}, f)
        ckpt_path = f"{d}/ckpt.pt"
        torch.save({"agent": agent.state_dict(), "ema_agent": make_agent(NORM_STATS_PATH).state_dict()}, ckpt_path)

        ckpt = torch.load(ckpt_path, weights_only=False)
        for key in ["agent", "ema_agent"]:
            restored = make_agent(other_stats)
            restored.load_state_dict(ckpt[key])
            assert torch.equal(restored.obs_mean, agent.obs_mean) and torch.equal(restored.obs_std, agent.obs_std)
            assert np.allclose(restored.obs_mean.numpy(), mean) and np.allclose(restored.obs_std.numpy(), std)

        # resuming a normalized checkpoint without --norm-stats-path must fail, not run unnormalized
        try:
            make_agent(None).load_state_dict(ckpt["agent"])
            raise AssertionError("loading normalized ckpt into unnormalized agent should fail")
        except RuntimeError:
            pass
    print("mean/std buffers identical after save/reload (agent and ema_agent)")


def test_no_norm_is_baseline():
    agent = make_agent(None)
    assert not any(k.startswith("obs_") for k in agent.state_dict())
    x = torch.randn(4, OBS_HORIZON, OBS_DIM)
    assert agent.normalize(x) is x
    print("no --norm-stats-path: no extra buffers, observations passed through unchanged")


def check_train_ckpt(path):
    mean, std = load_stats()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ["agent", "ema_agent"]:
        sd = ckpt[key]
        assert np.allclose(sd["obs_mean"].numpy(), mean) and np.allclose(sd["obs_std"].numpy(), std), key
        restored = make_agent(NORM_STATS_PATH)
        restored.load_state_dict(sd)
    assert torch.equal(ckpt["agent"]["obs_mean"], ckpt["ema_agent"]["obs_mean"])
    assert torch.equal(ckpt["agent"]["obs_std"], ckpt["ema_agent"]["obs_std"])
    print(f"{path}: agent and ema_agent buffers match norm_stats.json (iteration {ckpt['iteration']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="*", default=[])
    cli = parser.parse_args()
    test_normalized_demo_obs()
    test_buffers_roundtrip()
    test_no_norm_is_baseline()
    for p in cli.ckpt:
        check_train_ckpt(p)
    print("all checks passed")
