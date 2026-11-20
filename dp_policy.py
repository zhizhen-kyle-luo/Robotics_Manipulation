"""evaluate.py adapters for the frozen Diffusion Policy, alone or with a trained residual

    uv run python evaluate.py --policy dp_policy:make_dp --policy-kwargs '{"checkpoint": "base_ckpts/dp_clean_s0.pt"}' ...
    uv run python evaluate.py --policy dp_policy:make_residual --policy-kwargs '{"checkpoint": "runs/<run>/final_ckpt.pt"}' ...

make_residual takes a ppo.py checkpoint and loads the base it was trained on; base_checkpoint overrides the saved
path when the run moved machines
"""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import common

from ppo import Args, ResidualAgent
from residual import DiffusionBase, residual_input


class DPPolicy:
    """the base (plus the residual's mean action, if given) behind evaluate.py's reset() / act(state_dict obs)

    composes the action as ResidualWrapper does during training, so evaluation runs the trained policy exactly
    """

    def __init__(self, base, residual=None, scale=None, action_space=None):
        self.base, self.residual, self.scale = base, residual, scale
        if residual is not None:
            self.action_low = torch.as_tensor(action_space.low, device=base.device)
            self.action_high = torch.as_tensor(action_space.high, device=base.device)

    def reset(self):
        # evaluate.py starts every env on a new episode together
        self.base.reset(torch.ones(self.base.n, dtype=torch.bool))

    @torch.no_grad()
    def act(self, obs):
        obs = common.flatten_state_dict(obs, use_torch=True).to(self.base.device)
        base_action = self.base.act(obs)
        if self.residual is None:
            return base_action
        residual = self.residual.get_action(residual_input(self.base, obs, base_action), deterministic=True)
        return torch.clamp(base_action + self.scale * residual, self.action_low, self.action_high)


def make_dp(num_envs, env, checkpoint):
    return DPPolicy(DiffusionBase(checkpoint, num_envs, env.unwrapped.device))


def make_residual(num_envs, env, checkpoint, base_checkpoint=None):
    device = env.unwrapped.device
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    base = DiffusionBase(base_checkpoint or ckpt["base_checkpoint"], num_envs, device)
    action_space = env.unwrapped.single_action_space
    # ResidualAgent only reads the residual's input and output sizes
    spaces = SimpleNamespace(
        single_observation_space=gym.spaces.Box(-np.inf, np.inf, (base.obs_dim + base.act_dim,), np.float32),
        single_action_space=action_space)
    residual = ResidualAgent(spaces).to(device)
    residual.load_state_dict(ckpt["agent"])
    residual.eval()
    # ppo.py does not save --residual-scale; runs use the default
    return DPPolicy(base, residual, ckpt.get("residual_scale", Args.residual_scale), action_space)
