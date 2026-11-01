"""Train from saved grasped start states (the proposal: every method, RL included, trains from them).

Peg and box geometry is built per env slot at reconfigure time, and ManiSkill cannot reconfigure
a subset of slots. So a full reset reconfigures every slot from a fresh random batch of the pool,
and a partial reset puts each finished slot back into the start state its geometry was built for.
ppo.py calls a full reset every --starts-reshuffle-freq iterations so training covers the pool.
"""

import gymnasium as gym
import numpy as np
import torch

import start_states


class StartPoolWrapper(gym.Wrapper):
    """Wraps the raw ManiSkill env (below ManiSkillVectorEnv) so every reset lands in a saved start state."""

    def __init__(self, env, starts_path, seed=0):
        super().__init__(env)
        self.starts = start_states.load(starts_path)
        self.pool_size = len(self.starts["seeds"])
        self.num_envs = env.unwrapped.num_envs
        assert self.pool_size >= self.num_envs, (
            f"{starts_path} has {self.pool_size} start states but there are {self.num_envs} env slots"
        )
        self.rng = np.random.default_rng(seed)
        # only our full reset may reconfigure; an automatic one would draw new geometry under a slot's saved state
        env.unwrapped.reconfiguration_freq = 0
        # pool index each slot was built for; set on the first full reset
        self.slot_starts = None

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if "env_idx" in options and self.slot_starts is not None:
            # partial reset (also the all-slot auto reset at truncation): no reconfigure, same geometry
            env_idx = torch.as_tensor(options["env_idx"]).cpu()
            states = start_states.index_states(self.starts["env_states"], self.slot_starts[env_idx])
            states = start_states._to_device(states, self.env.unwrapped.device)
            options["reset_to_env_states"] = dict(env_states=states)
            return self.env.reset(options=options)
        # full reset: rebuild every slot from a new batch drawn without replacement
        self.slot_starts = torch.from_numpy(self.rng.choice(self.pool_size, self.num_envs, replace=False))
        return start_states.reset_to(self.env, self.starts, self.slot_starts)
