from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

import ppo


def spaces(obs_dim=50, act_dim=7):
    return SimpleNamespace(single_observation_space=gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
                           single_action_space=gym.spaces.Box(-1, 1, (act_dim,), np.float32))


def test_residual_agent_starts_at_the_base_with_fixed_std():
    agent = ppo.ResidualAgent(spaces())
    obs = torch.randn(32, 50)
    assert torch.equal(agent.get_action(obs, deterministic=True), torch.zeros(32, 7))
    assert not agent.actor_logstd.requires_grad
    assert torch.allclose(agent.actor_logstd, torch.full((1, 7), -1.0))
    linears = [m for m in agent.actor_mean if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linears] == [256, 256, 7] and linears[-1].bias is None
    assert all(isinstance(m, torch.nn.ReLU) for m in agent.critic if not isinstance(m, torch.nn.Linear))
    assert torch.equal(agent.critic[-1].bias, torch.tensor([0.25]))


def test_lr_schedule_matches_resip_cosine_with_warmup():
    total, warmup = 40, 5
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total)
    for step in range(total):
        assert ppo.cosine_with_warmup(step, warmup, total) == pytest.approx(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()


def test_reward_normalizer_divides_by_running_std_and_clips():
    norm = ppo.RewardNormalizer(clip=5.0)
    batches = [torch.bernoulli(torch.full((64,), 0.05)) for _ in range(20)]
    for b in batches:
        out = norm(b)
    seen = torch.cat(batches)
    # running variance of everything seen, against a 1e-4 count prior of mean 0 and variance 1
    want = torch.clamp(batches[-1] / torch.sqrt(seen.var(unbiased=False) + 1e-8), -5, 5)
    assert torch.allclose(out, want, rtol=1e-3)
    assert norm(torch.full((4,), 1e6)).max() == 5.0


def test_defaults_follow_resip():
    args = ppo.Args()
    assert (args.gamma, args.num_envs, args.num_minibatches, args.update_epochs) == (0.999, 1024, 1, 50)
    assert (args.vf_coef, args.max_grad_norm) == (1.0, 1.0)
