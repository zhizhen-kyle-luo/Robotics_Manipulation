import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from mani_skill.utils.structs.pose import Pose

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "diffusion_policy"))  # train.py imports its helpers as diffusion_policy.*

import train  # noqa: E402
from diffusion_policy.utils import HOLE_POSE, clean_traj_keys, perturb_hole_pose, sample_hole_noise  # noqa: E402
from dp_policy import make_dp, make_residual  # noqa: E402
from ppo import ResidualAgent  # noqa: E402
from residual import OBS_CLIP, DiffusionBase, ObsNormalizer, ResidualWrapper  # noqa: E402

OBS_DIM, ACT_DIM, NUM_ENVS = 43, 7, 3
TINY = dict(obs_horizon=2, act_horizon=8, pred_horizon=16, diffusion_step_embed_dim=8, unet_dims=[8, 16], n_groups=2)
ACTION_SPACE = gym.spaces.Box(-1, 1, (ACT_DIM,), np.float32)


def write_demo_json(path, episodes):
    path.write_text(json.dumps({"episodes": [dict(episode_id=i, episode_seed=seed, success=ok)
                                             for i, (seed, ok) in enumerate(episodes)]}))


def test_clean_demos_keep_the_first_success_of_each_seed(tmp_path):
    # seed 5: failed then succeeded, seed 7: two successes, seed 9: failed only
    write_demo_json(tmp_path / "d.json", [(5, False), (7, True), (5, True), (7, True), (9, False), (5, True)])
    assert clean_traj_keys(tmp_path / "d.json") == ["traj_1", "traj_2"]


@pytest.mark.skipif("DP_DEMO_JSON" not in os.environ, reason="set DP_DEMO_JSON to the DP training demo json")
def test_clean_demos_on_the_real_file():
    assert len(clean_traj_keys(os.environ["DP_DEMO_JSON"])) == 649


def test_hole_noise_is_a_bounded_hole_frame_error():
    torch.manual_seed(0)
    noise = sample_hole_noise(1000, "cpu")
    assert torch.all(noise[:, 0] == 0) and torch.linalg.norm(noise[:, :3], dim=1).max() <= 0.006
    angle = 2 * torch.acos(noise[:, 3].clamp(max=1))
    assert angle.max() <= np.deg2rad(1.0) + 1e-6 and angle.max() > np.deg2rad(0.9)
    # a demo's steps share one error; everything but the hole pose is untouched
    obs = torch.randn(2, 4, OBS_DIM)
    hole = Pose.create_from_pq(p=torch.randn(1, 3), q=torch.nn.functional.normalize(torch.randn(1, 4), dim=-1))
    obs[..., HOLE_POSE] = hole.raw_pose
    out = perturb_hole_pose(obs, noise[:2, None])
    rest = [i for i in range(OBS_DIM) if not HOLE_POSE.start <= i < HOLE_POSE.stop]
    assert torch.equal(out[..., rest], obs[..., rest])
    for e in range(2):
        for t in range(4):
            err = (hole.inv() * Pose.create(out[e, t:t + 1, HOLE_POSE])).raw_pose[0]
            assert torch.allclose(err, noise[e], atol=1e-5)


def tiny_agent(tmp_path, normalized):
    stats = None
    if normalized:
        stats = tmp_path / "norm_stats.json"
        stats.write_text(json.dumps({"norm_stats": {"state": {"mean": torch.randn(OBS_DIM).tolist(),
                                                               "std": (torch.rand(OBS_DIM) + 0.5).tolist()}}}))
    env = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (2, OBS_DIM), np.float32),
                          single_action_space=ACTION_SPACE)
    return train.Agent(env, train.Args(norm_stats_path=stats and str(stats), **TINY)).eval()


def export(tmp_path, agent):
    """a train.py-style checkpoint of agent run through scripts/export_dp_base.py"""
    torch.save({"agent": agent.state_dict(), "ema_agent": agent.state_dict(), "rng": {"numpy": np.random.get_state()}},
               tmp_path / "train.pt")
    flags = ["--diffusion-step-embed-dim", "8", "--unet-dims", "8", "16", "--n-groups", "2"]
    subprocess.run([sys.executable, str(REPO / "scripts/export_dp_base.py"), str(tmp_path / "train.pt"),
                    str(tmp_path / "base.pt"), *flags], check=True)
    return tmp_path / "base.pt"


@pytest.mark.parametrize("normalized", [True, False])
def test_diffusion_base_matches_train_agent(tmp_path, normalized):
    agent = tiny_agent(tmp_path, normalized)
    base = DiffusionBase(export(tmp_path, agent), NUM_ENVS, "cpu")
    obs = torch.randn(NUM_ENVS, OBS_DIM) * 3
    torch.manual_seed(0)
    want = agent.get_action(obs[:, None].repeat(1, 2, 1))  # FrameStack repeats the first obs
    torch.manual_seed(0)
    got = torch.stack([base.act(obs) for _ in range(TINY["act_horizon"])], 1)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.equal(base.history[:, -1], obs)  # the history keeps raw obs


class FakeVectorEnv(gym.vector.VectorEnv):
    """the slice of ManiSkillVectorEnv that ResidualWrapper touches; returns fixed obs and records actions"""

    def __init__(self, obs):
        self.obs, self.sent, self.device = obs, [], "cpu"
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.single_action_space = ACTION_SPACE

    def reset(self, *, seed=None, options=None):
        return self.obs, {}

    def step(self, action):
        self.sent.append(action)
        done = torch.zeros(len(self.obs), dtype=torch.bool)
        return self.obs, torch.zeros(len(self.obs)), done, done, {}


def obs_stats():
    return dict(mean=torch.randn(OBS_DIM).tolist(), std=(torch.rand(OBS_DIM) * 0.5 + 0.01).tolist())


def test_residual_sees_clipped_normalized_obs_and_base_sees_raw(tmp_path):
    base = DiffusionBase(export(tmp_path, tiny_agent(tmp_path, False)), NUM_ENVS, "cpu")
    obs, norm = torch.randn(NUM_ENVS, OBS_DIM), ObsNormalizer(obs_stats(), "cpu")
    wrapped = ResidualWrapper(FakeVectorEnv(obs), base, 0.1, norm)
    residual_obs, _ = wrapped.reset()
    z = (obs - norm.mean) / norm.std
    assert z.abs().max() > OBS_CLIP  # the stats make some dims overflow, as joint velocities do
    assert torch.allclose(residual_obs, torch.cat([z.clamp(-OBS_CLIP, OBS_CLIP), wrapped.base_action], -1))
    assert torch.equal(base.history[:, -1], obs)


def state_dict_obs(flat):
    return {"agent": {"qpos": flat[:, :9], "qvel": flat[:, 9:18]}, "extra": {"rest": flat[:, 18:]}}


def test_adapters_execute_what_residual_training_executes(tmp_path):
    base_path = export(tmp_path, tiny_agent(tmp_path, False))
    obs, norm = torch.randn(NUM_ENVS, OBS_DIM), ObsNormalizer(obs_stats(), "cpu")
    env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu", single_action_space=ACTION_SPACE))
    residual = ResidualAgent(SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (OBS_DIM + ACT_DIM,), np.float32),
        single_action_space=ACTION_SPACE))
    torch.nn.init.normal_(residual.actor_mean[-1].weight, std=1.0)  # big enough that some actions clamp
    torch.save(dict(agent=residual.state_dict(), base_checkpoint=str(base_path), obs_norm=norm.stats), tmp_path / "ppo.pt")

    fake = FakeVectorEnv(obs)
    wrapped = ResidualWrapper(fake, DiffusionBase(base_path, NUM_ENVS, "cpu"), 0.1, norm)
    torch.manual_seed(0)
    residual_obs, _ = wrapped.reset()
    base_action = wrapped.base_action
    with torch.no_grad():
        wrapped.step(residual.get_action(residual_obs, deterministic=True))
    assert (fake.sent[0].abs() == 1).any()

    for factory, want in [(make_dp, base_action), (make_residual, fake.sent[0])]:
        policy = factory(NUM_ENVS, env, str(base_path if factory is make_dp else tmp_path / "ppo.pt"))
        torch.manual_seed(0)
        policy.reset()
        assert torch.allclose(policy.act(state_dict_obs(obs)), want, atol=1e-6)

def test_batched_hole_noise_matches_per_sample(monkeypatch):
    rng = np.random.default_rng(0)
    lengths = [20, 23, 31]
    obs, acts = [], []
    for L in lengths:
        o = rng.normal(size=(L + 1, OBS_DIM)).astype(np.float32)
        q = rng.normal(size=4)
        o[:, HOLE_POSE.start + 3:HOLE_POSE.stop] = (q / np.linalg.norm(q)).astype(np.float32)
        obs.append(o)
        acts.append(rng.uniform(-1, 1, size=(L, ACT_DIM)).astype(np.float32))
    monkeypatch.setattr("diffusion_policy.utils.load_demo_dataset",
                        lambda *a, **k: dict(observations=list(obs), actions=list(acts)))
    monkeypatch.setattr(train, "args", SimpleNamespace(control_mode="pd_ee_delta_pose", obs_horizon=2,
                                                       pred_horizon=16, hole_noise=True), raising=False)
    torch.manual_seed(0)
    dataset = train.SmallDemoDataset_DiffusionPolicy("demos.h5", "cpu", num_traj=None)
    assert any(s < 0 for _, s, _ in dataset.slices)
    assert any(e > lengths[t] for t, _, e in dataset.slices)

    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    rest = [i for i in range(OBS_DIM) if not HOLE_POSE.start <= i < HOLE_POSE.stop]
    for _ in range(2):
        batch = next(iter(loader))
        out = dataset.add_hole_noise(batch["observations"], batch["traj_idx"])
        for b in range(len(out)):
            expected = perturb_hole_pose(batch["observations"][b], dataset.hole_noise[batch["traj_idx"][b]])
            assert torch.allclose(out[b], expected, atol=1e-6)
        assert torch.equal(out[..., rest], batch["observations"][..., rest])
        assert not torch.allclose(out[..., HOLE_POSE], batch["observations"][..., HOLE_POSE])
        for i, (t, s, _) in enumerate(dataset.slices):
            item = dataset[i]
            assert item["traj_idx"] == t
            assert torch.equal(item["observations"][-1], torch.as_tensor(obs[t][s + 1]))
        before = dataset.hole_noise.clone()
        dataset.resample_hole_noise()
        assert not torch.equal(before, dataset.hole_noise)
