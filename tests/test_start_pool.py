import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import evaluate  # noqa: F401  macOS Vulkan setup before SAPIEN loads
import noisy_env  # noqa: F401  registers PegInsertionSideNoisy-v1
import start_states
from start_pool import StartPoolWrapper

STARTS = "data/interim_test.pt"


def make_env():
    # reconfiguration_freq defaults to 1 with one env, which the wrapper has to override
    env = gym.make("PegInsertionSideNoisy-v1", obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", hole_offset_range=(0.0, 0.004))
    env = StartPoolWrapper(env, STARTS, seed=0)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True)


def peg_pose(env):
    return env.unwrapped.peg.pose.raw_pose[0].clone()


def saved_peg_pose(starts, i):
    return starts["env_states"]["actors"]["peg"][i, :7]


def test_full_reset_lands_in_a_saved_start_state():
    env = make_env()
    starts = start_states.load(STARTS)
    env.reset(seed=0)
    i = env._env.slot_starts[0].item()
    assert torch.allclose(peg_pose(env), saved_peg_pose(starts, i), atol=1e-5)


def test_auto_reset_restores_the_same_slot_state_without_reconfiguring():
    env = make_env()
    starts = start_states.load(STARTS)
    env.reset(seed=0)
    i = env._env.slot_starts[0].item()
    half_sizes = env.unwrapped.peg_half_sizes.clone()
    act = torch.zeros(env.single_action_space.shape)[None]
    for _ in range(gym_utils.find_max_episode_steps_value(env._env)):
        _, _, _, truncated, info = env.step(act)
    assert truncated.all() and "final_observation" in info
    assert env._env.slot_starts[0].item() == i
    assert torch.equal(env.unwrapped.peg_half_sizes, half_sizes)
    assert torch.allclose(peg_pose(env), saved_peg_pose(starts, i), atol=1e-5)


def test_full_reset_draws_a_new_batch():
    env = make_env()
    drawn = set()
    for _ in range(5):
        env.reset()
        drawn.add(env._env.slot_starts[0].item())
    assert len(drawn) > 1


def test_pool_smaller_than_slots_is_dealt_evenly():
    # the real numbers: 1000 train states in 2048 slots, 500 eval states in 500 slots
    wrapper = StartPoolWrapper.__new__(StartPoolWrapper)
    wrapper.rng = np.random.default_rng(0)
    wrapper.pool_size, wrapper.num_envs = 1000, 2048
    assert not wrapper.needs_reshuffle
    counts = np.bincount(wrapper._deal_slots(), minlength=1000)
    assert counts.min() == 2 and counts.max() == 3
    wrapper.pool_size, wrapper.num_envs = 500, 500
    assert not wrapper.needs_reshuffle
    assert sorted(wrapper._deal_slots().tolist()) == list(range(500))


def test_full_reset_does_not_rebuild_when_slots_cover_the_pool():
    env = make_env()
    env._env.pool_size = 1  # pretend the pool fits in the single slot
    env.reset(seed=0)
    built = env._env.slot_starts.clone()
    half_sizes = env.unwrapped.peg_half_sizes.clone()
    reconfigures = []
    inner_reconfigure = env.unwrapped._reconfigure
    env.unwrapped._reconfigure = lambda options: (reconfigures.append(1), inner_reconfigure(options))[1]
    env.step(torch.zeros(env.single_action_space.shape)[None])
    env.reset()
    assert not reconfigures
    assert torch.equal(env._env.slot_starts, built)
    assert torch.equal(env.unwrapped.peg_half_sizes, half_sizes)


def test_sut_generated_states_restore_on_the_noisy_env():
    # generate_start_states.py saves a list of records from PegInsertionSide-v1; skipped until fetched
    path = "start_states/eval/states.pt"
    if not os.path.exists(path):
        pytest.skip("run scripts/fetch_start_states.sh first")
    env = gym.make("PegInsertionSideNoisy-v1", obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", hole_offset_range=(0.0, 0.004))
    starts = start_states.load(path)
    assert len(starts["seeds"]) == 500
    for k in [0, 123, 499]:
        start_states.reset_to(env, starts, [k])
        assert torch.allclose(env.unwrapped.peg.pose.raw_pose[0], starts["env_states"]["actors"]["peg"][k, :7], atol=1e-5)
        hold = torch.zeros((1, 7))
        hold[:, -1] = -1
        env.step(hold)
        assert env.unwrapped.agent.is_grasping(env.unwrapped.peg)[0]
    env.close()
