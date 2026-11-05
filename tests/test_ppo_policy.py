import torch

from mani_skill.utils import common

import evaluate
import start_states
from ppo_policy import make_policy

STARTS = "data/interim_test.pt"


def save_checkpoint(path, gripper_closed):
    from types import SimpleNamespace

    import gymnasium as gym
    import numpy as np

    from ppo import Agent

    spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (43,), np.float32),
                             single_action_space=gym.spaces.Box(-1, 1, (7,), np.float32))
    agent = Agent(spaces, logstd_init=-0.5)
    torch.save(dict(agent=agent.state_dict(), gripper_closed=gripper_closed), path)
    return agent


def test_policy_returns_the_checkpoints_mean_action_on_the_flat_obs(tmp_path):
    path = tmp_path / "ckpt.pt"
    agent = save_checkpoint(path, gripper_closed=False)
    env = evaluate.make_env(num_envs=1, sim_backend="physx_cpu")
    obs, _ = start_states.reset_to(env, start_states.load(STARTS), [0])
    policy = make_policy(num_envs=1, env=env, checkpoint=str(path))
    want = torch.clamp(agent.get_action(common.flatten_state_dict(obs, use_torch=True), deterministic=True), -1, 1)
    assert torch.allclose(policy.act(obs), want)
    env.close()


def test_gripper_closed_checkpoint_keeps_the_gripper_closed(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, gripper_closed=True)
    env = evaluate.make_env(num_envs=1, sim_backend="physx_cpu")
    obs, _ = start_states.reset_to(env, start_states.load(STARTS), [0])
    policy = make_policy(num_envs=1, env=env, checkpoint=str(path))
    assert policy.act(obs)[0, -1] == -1
    env.close()


def test_evaluate_runs_a_ppo_checkpoint_end_to_end(tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, gripper_closed=True)
    results = evaluate.evaluate("ppo_policy:make_policy", STARTS, [0.0, 4.0], num_envs=1, sim_backend="physx_cpu",
                                max_steps=5, seed=0, limit=2, policy_kwargs=dict(checkpoint=str(path)))
    assert [row["noise_mm"] for row in results["sweep"]] == [0.0, 4.0]
    assert results["sweep"][0]["success_once"] == 0.0  # an untrained policy cannot insert in 5 steps
