import gymnasium as gym
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
