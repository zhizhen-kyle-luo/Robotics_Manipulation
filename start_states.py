"""Saved start states (peg already grasped) shared by every method.

load() returns a dict:
    seeds          LongTensor (N,)   episode seed; fixes peg size and hole offset on reconfigure
    env_states     nested dict of tensors with leading dim N, from env.get_state_dict()
    meta           dict (how the states were made)
It reads that dict as saved by save(), and the list of per-state records (seed, state_dict, ...)
saved by generate_start_states.py.
"""

import torch


def stack_states(state_dicts):
    first = state_dicts[0]
    if isinstance(first, dict):
        return {k: stack_states([s[k] for s in state_dicts]) for k in first}
    return torch.cat([torch.as_tensor(s) for s in state_dicts], dim=0)


def index_states(states, idx):
    if isinstance(states, dict):
        return {k: index_states(v, idx) for k, v in states.items()}
    return states[idx]


def load(path):
    # generate_start_states.py's records also hold numpy poses, which weights_only=True refuses
    raw = torch.load(path, weights_only=False)
    if isinstance(raw, list):
        return dict(seeds=torch.tensor([r["seed"] for r in raw]),
                    env_states=stack_states([r["state_dict"] for r in raw]),
                    meta=dict(source="generate_start_states.py", path=str(path)))
    return raw


def save(path, seeds, env_states, meta):
    torch.save(dict(seeds=torch.as_tensor(seeds), env_states=env_states, meta=meta), path)


def reset_to(env, starts, idx):
    """Reset env (num_envs == len(idx)) to start states starts[idx]. Returns obs, info."""
    idx = torch.as_tensor(idx)
    u = env.unwrapped
    states = {k: v for k, v in index_states(starts["env_states"], idx).items()}
    states = _to_device(states, u.device)
    obs, info = env.reset(
        seed=starts["seeds"][idx].tolist(),
        options=dict(reconfigure=True, reset_to_env_states=dict(env_states=states)),
    )
    # the task puts the box center at height = peg half length, so a saved state records its geometry
    saved_half_length = states["actors"]["box_with_hole"][:, 2]
    assert torch.allclose(u.peg_half_sizes[:, 0], saved_half_length, atol=1e-5), (
        "geometry mismatch: seeds do not reproduce the saved scene")
    return obs, info


def _to_device(states, device):
    if isinstance(states, dict):
        return {k: _to_device(v, device) for k, v in states.items()}
    return states.to(device)
