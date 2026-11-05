"""A checkpoint from ppo.py (PPO from scratch) as an evaluate.py policy:

    python evaluate.py --policy ppo_policy:make_policy --policy-kwargs '{"checkpoint": "runs/<run>/final_ckpt.pt"}' \
        --starts start_states/eval/states.pt --num-envs 100 --out results/<run>.json
"""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils import common

from ppo import Agent


class PPOPolicy:
    def __init__(self, env, checkpoint):
        self.device = env.unwrapped.device
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        # the state observation's size is the checkpoint's; the policy is what produces the 7 action dims
        obs_dim = ckpt["agent"]["actor_mean.0.weight"].shape[1]
        act_dim = ckpt["agent"]["actor_logstd"].shape[1]
        spaces = SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
                                 single_action_space=gym.spaces.Box(-1, 1, (act_dim,), np.float32))
        self.agent = Agent(spaces, logstd_init=0.0).to(self.device)
        self.agent.load_state_dict(ckpt["agent"])
        self.agent.eval()
        self.gripper_closed = ckpt["gripper_closed"]

    def reset(self):
        pass  # no memory between steps

    def act(self, obs):
        # flattening the state_dict gives exactly the obs_mode="state" vector the agent trained on
        flat = common.flatten_state_dict(obs, use_torch=True)
        with torch.no_grad():
            # the mean action, as in ppo.py's evaluation
            action = torch.clamp(self.agent.get_action(flat, deterministic=True), -1, 1)
        if self.gripper_closed:
            action[:, -1] = -1
        return action


def make_policy(num_envs, env, checkpoint):
    return PPOPolicy(env, checkpoint)
