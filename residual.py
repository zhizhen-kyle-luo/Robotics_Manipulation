import gymnasium as gym
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from conditional_unet1d import ConditionalUnet1D


class DiffusionBase:
    """frozen Diffusion Policy (diffusion_policy/train.py on the data_setup branch) as the residual's base

    every base follows this interface: reset(mask) clears the memory of the masked envs, act(obs) returns this
    control step's action for all envs and is called exactly once per env step. it reproduces the policy's own
    evaluation: the network sees the last obs_horizon raw observations (an episode's first observation repeated,
    as ManiSkill's FrameStack does), predicts pred_horizon actions with 100 DDPM steps, and executes act_horizon
    of them before planning again. envs plan independently, since partial resets put them at different points
    of their chunks; the ones whose chunk ran out this step are denoised together as one batch

    the checkpoint is a scripts/export_dp_base.py export: the EMA weights and the horizons they were trained with
    """

    def __init__(self, checkpoint, num_envs, device):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        weights, cfg = ckpt["ema_agent"], ckpt["config"]
        if any(k in weights for k in ("obs_mean", "obs_std")):
            raise NotImplementedError("this Diffusion Policy was trained with observation normalization; "
                                      "DiffusionBase feeds raw observations")
        self.obs_horizon, self.act_horizon, self.pred_horizon = cfg["obs_horizon"], cfg["act_horizon"], cfg["pred_horizon"]
        self.act_dim = weights["noise_pred_net.final_conv.1.weight"].shape[0]
        cond_dim = weights["noise_pred_net.mid_modules.0.cond_encoder.1.weight"].shape[1]
        self.obs_dim = (cond_dim - cfg["diffusion_step_embed_dim"]) // self.obs_horizon
        self.net = ConditionalUnet1D(input_dim=self.act_dim, global_cond_dim=self.obs_horizon * self.obs_dim,
                                     diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
                                     down_dims=cfg["unet_dims"], n_groups=cfg["n_groups"]).to(device)
        self.net.load_state_dict({k.removeprefix("noise_pred_net."): v for k, v in weights.items()})
        self.net.eval().requires_grad_(False)
        # same scheduler as the training script; its default timesteps are all 100, as in training
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                       clip_sample=True, prediction_type="epsilon")

        self.n, self.device = num_envs, device
        self.history = torch.zeros((num_envs, self.obs_horizon, self.obs_dim), device=device)
        self.chunk = torch.zeros((num_envs, self.act_horizon, self.act_dim), device=device)
        # next action's index in chunk; act_horizon means plan a new chunk
        self.step = torch.full((num_envs,), self.act_horizon, dtype=torch.long, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)  # next obs starts a new episode

    def reset(self, mask):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        self.fresh |= mask
        self.step[mask] = self.act_horizon

    @torch.no_grad()
    def act(self, obs):
        if obs.shape != (self.n, self.obs_dim):
            raise ValueError(f"expected obs of shape {(self.n, self.obs_dim)}, got {tuple(obs.shape)}")
        obs = obs.float()
        self.history = torch.cat([self.history[:, 1:], obs[:, None]], 1)
        self.history[self.fresh] = obs[self.fresh, None]
        self.fresh[:] = False

        plan = self.step >= self.act_horizon
        if plan.any():
            self.chunk[plan] = self.plan(self.history[plan])
            self.step[plan] = 0
        action = self.chunk[torch.arange(self.n, device=self.device), self.step]
        self.step += 1
        return action

    def plan(self, obs_seq):
        """the training script's Agent.get_action: act_horizon actions for each (obs_horizon, obs_dim) history"""
        cond = obs_seq.flatten(start_dim=1)
        actions = torch.randn((obs_seq.shape[0], self.pred_horizon, self.act_dim), device=obs_seq.device)
        for k in self.scheduler.timesteps:
            noise = self.net(sample=actions, timestep=k, global_cond=cond)
            actions = self.scheduler.step(model_output=noise, timestep=k, sample=actions).prev_sample
        start = self.obs_horizon - 1
        return actions[:, start:start + self.act_horizon]


class ResidualWrapper(gym.vector.VectorWrapper):
    """ResiP residual: executes base action + scale * residual at every control step

    the agent observes [env obs, base action for this step]; the frozen base sees the same env obs,
    so a hole offset injected by the env reaches both. wraps a ManiSkillVectorEnv, whose auto-reset
    returns the next episode's first obs for done envs, so the base is reset before it acts on them
    """

    def __init__(self, env, base, scale):
        super().__init__(env)
        self.base, self.scale = base, scale
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim + act_dim,), np.float32)
        self.action_low = torch.as_tensor(env.single_action_space.low, device=env.device)
        self.action_high = torch.as_tensor(env.single_action_space.high, device=env.device)
        self.base_action = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if options is not None and "env_idx" in options:
            # partial reset: only the listed envs start a new episode
            mask = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
            mask[options["env_idx"]] = True
        else:
            mask = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        self.base.reset(mask)
        self.base_action = self.base.act(obs)
        return torch.cat([obs, self.base_action], -1), info

    def step(self, actions):
        action = torch.clamp(self.base_action + self.scale * actions, self.action_low, self.action_high)
        obs, rew, terminations, truncations, infos = self.env.step(action)
        self.base.reset(terminations | truncations)
        self.base_action = self.base.act(obs)
        return torch.cat([obs, self.base_action], -1), rew, terminations, truncations, infos
