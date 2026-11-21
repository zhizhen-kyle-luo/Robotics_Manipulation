import gymnasium as gym
import pytest
import torch

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import noisy_env  # noqa: F401  registers PegInsertionSideNoisy-v1
from residual import ResidualWrapper


class StubBase:
    """residual.py's base interface with a constant zero action"""

    def __init__(self, act_dim):
        self.act_dim = act_dim

    def reset(self, mask):
        pass

    def act(self, obs):
        return torch.zeros((obs.shape[0], self.act_dim), device=obs.device)

    def normalize(self, obs):
        return obs


def make_env(hole_offset_range, hole_rot_range_deg=(0.0, 0.0)):
    env = gym.make("PegInsertionSideNoisy-v1", obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", hole_offset_range=hole_offset_range, hole_rot_range_deg=hole_rot_range_deg)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True)


def hole_frame_offset(env):
    base_env = env.unwrapped
    return (base_env.box_hole_pose.inv() * base_env.observed_hole_pose).p[0]


def hole_frame_rotation(env):
    base_env = env.unwrapped
    return (base_env.box_hole_pose.inv() * base_env.observed_hole_pose).q[0]


def rotation_deg(q):
    return torch.rad2deg(2 * torch.acos(torch.clamp(q[0].abs(), max=1.0))).item()


@pytest.mark.parametrize("r", [0.0, 0.002, 0.004, 0.006])
def test_sweep_offset_is_exact_in_hole_plane_and_fixed_per_episode(r):
    env = make_env((r, r))
    for seed in range(3):
        env.reset(seed=seed)
        first = hole_frame_offset(env)
        assert abs(first[0].item()) < 1e-6
        assert abs(torch.linalg.norm(first[1:]).item() - r) < 1e-6
        for _ in range(5):
            env.step(env.action_space.sample() * 0)
            assert torch.allclose(hole_frame_offset(env), first, atol=1e-6)
    env.close()


def test_training_offset_stays_within_range():
    env = make_env((0.0, 0.004))
    for seed in range(20):
        env.reset(seed=seed)
        assert torch.linalg.norm(hole_frame_offset(env)).item() <= 0.004 + 1e-6
    env.close()


def test_observed_hole_pose_reaches_policy_obs():
    env = make_env((0.004, 0.004))
    obs, _ = env.reset(seed=0)
    base_env = env.unwrapped
    observed = base_env.observed_hole_pose.raw_pose[0]
    true = base_env.box_hole_pose.raw_pose[0]
    assert any(torch.allclose(obs[0, i:i + 7], observed, atol=1e-6) for i in range(obs.shape[1] - 6))
    assert not any(torch.allclose(obs[0, i:i + 7], true, atol=1e-6) for i in range(obs.shape[1] - 6))
    env.close()


def test_zero_residual_reproduces_base():
    base_only = make_env((0.002, 0.002))
    residual_env = make_env((0.002, 0.002))
    act_dim = base_only.single_action_space.shape[0]
    base = StubBase(act_dim)
    wrapped = ResidualWrapper(residual_env, StubBase(act_dim), 0.1)
    obs, _ = base_only.reset(seed=0)
    wrapped_obs, _ = wrapped.reset(seed=0)
    for _ in range(20):
        assert torch.allclose(wrapped_obs[:, :obs.shape[1]], obs, atol=1e-5)
        assert torch.equal(wrapped_obs[:, obs.shape[1]:], base.act(obs))
        obs, *_ = base_only.step(base.act(obs))
        wrapped_obs, *_ = wrapped.step(torch.zeros((1, act_dim)))
    base_only.close()
    wrapped.close()


def test_residual_is_scaled_then_clamped():
    env = make_env((0.0, 0.0))
    act_dim = env.single_action_space.shape[0]
    wrapped = ResidualWrapper(env, StubBase(act_dim), 0.1)
    wrapped.reset(seed=0)
    sent = []
    inner_step = env.step
    env.step = lambda action: (sent.append(action.clone()), inner_step(action))[1]
    base_action = StubBase(act_dim).act(torch.zeros((1, 1)))
    wrapped.step(torch.full((1, act_dim), 0.5))
    assert torch.allclose(sent[0], torch.clamp(base_action + 0.05, -1, 1))
    wrapped.step(torch.full((1, act_dim), 100.0))
    assert torch.allclose(sent[1], torch.ones((1, act_dim)))
    wrapped.close()


class RecordingBase(StubBase):
    def __init__(self, act_dim):
        super().__init__(act_dim)
        self.events = []

    def reset(self, mask):
        self.events.append(("reset", bool(mask.any())))

    def act(self, obs):
        self.events.append(("act", obs.clone()))
        return super().act(obs)


def test_base_resets_before_acting_on_next_episode():
    env = make_env((0.0, 0.0))
    act_dim = env.single_action_space.shape[0]
    base = RecordingBase(act_dim)
    wrapped = ResidualWrapper(env, base, 0.1)
    wrapped.reset(seed=0)
    for _ in range(250):
        base.events.clear()
        obs, _, _, truncations, _ = wrapped.step(torch.zeros((1, act_dim)))
        if truncations.any():
            break
    assert truncations.any()
    # auto-reset already happened, so obs is the next episode's first obs
    assert env.unwrapped.elapsed_steps[0].item() == 0
    (reset_kind, reset_any), (act_kind, act_obs) = base.events
    assert (reset_kind, reset_any, act_kind) == ("reset", True, "act")
    assert torch.equal(act_obs, obs[:, :act_obs.shape[1]])
    wrapped.close()


def test_offset_is_redrawn_on_reset_to_saved_state():
    # start states are restored with reset_to_env_states, which skips _initialize_episode
    env = make_env((0.002, 0.004))
    env.reset(seed=0)
    saved = env.unwrapped.get_state_dict()
    offsets = []
    for seed in range(5):
        env.reset(seed=seed, options=dict(reset_to_env_states=dict(env_states=saved)))
        offset = hole_frame_offset(env)
        assert 0.002 - 1e-6 <= torch.linalg.norm(offset).item() <= 0.004 + 1e-6
        offsets.append(offset)
    assert len({tuple(o.tolist()) for o in offsets}) == 5
    env.close()


def test_rotation_error_stays_within_range_and_is_fixed_per_episode():
    env = make_env((0.0, 0.0), (0.5, 1.0))
    for seed in range(10):
        env.reset(seed=seed)
        first = hole_frame_rotation(env)
        assert 0.5 - 1e-3 <= rotation_deg(first) <= 1.0 + 1e-3
        for _ in range(3):
            env.step(env.action_space.sample() * 0)
            assert torch.allclose(hole_frame_rotation(env), first, atol=1e-6)
    env.close()


def test_rotation_error_is_redrawn_on_every_reset_path():
    env = make_env((0.0, 0.0), (1.0, 1.0))
    env.reset(seed=0)
    saved = env.unwrapped.get_state_dict()
    rotations = [hole_frame_rotation(env)]
    env.reset(seed=1)
    rotations.append(hole_frame_rotation(env))
    env.reset(seed=2, options=dict(reset_to_env_states=dict(env_states=saved)))
    rotations.append(hole_frame_rotation(env))
    # auto-reset at truncation
    for _ in range(250):
        *_, truncations, _ = env.step(env.action_space.sample() * 0)
        if truncations.any():
            break
    assert truncations.any()
    rotations.append(hole_frame_rotation(env))
    assert all(abs(rotation_deg(q) - 1.0) < 1e-3 for q in rotations)
    assert len({tuple(q.tolist()) for q in rotations}) == 4
    env.close()


def test_rotation_error_leaves_observed_position_unchanged():
    env = make_env((0.004, 0.004), (1.0, 1.0))
    env.reset(seed=0)
    offset = hole_frame_offset(env)
    assert abs(offset[0].item()) < 1e-6
    assert abs(torch.linalg.norm(offset[1:]).item() - 0.004) < 1e-6
    assert torch.allclose(offset, env.unwrapped.hole_offsets[0], atol=1e-6)
    assert abs(rotation_deg(hole_frame_rotation(env)) - 1.0) < 1e-3
    env.close()
