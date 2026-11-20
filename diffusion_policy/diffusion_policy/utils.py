import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from h5py import Dataset, File, Group
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply, quaternion_multiply
from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """Wraps a BatchSampler.
    Resampling from it until a specified number of iterations have been sampled
    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def worker_init_fn(worker_id, base_seed=None):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    if base_seed is None:
        base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)


TARGET_KEY_TO_SOURCE_KEY = {
    "states": "env_states",
    "observations": "obs",
    "success": "success",
    "next_observations": "obs",
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    "actions": "actions",
}


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_hdf5(
    path,
):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    ret = load_content_from_h5_file(file)
    file.close()
    print("Loaded")
    return ret


def clean_traj_keys(json_path):
    """h5 keys of the successful demos, one per seed

    the planner's demo file repeats seeds (byte-identical actions) and keeps failed attempts, so training on it
    as is double counts some scenes and imitates failures. episodes are kept in episode_id order, so the first
    successful demo of each seed survives
    """
    with open(json_path) as f:
        episodes = sorted(json.load(f)["episodes"], key=lambda e: e["episode_id"])
    keys, seeds = [], set()
    for e in episodes:
        if e["success"] and e["episode_seed"] not in seeds:
            seeds.add(e["episode_seed"])
            keys.append(f"traj_{e['episode_id']}")
    num_failed = sum(not e["success"] for e in episodes)
    print(f"Kept {len(keys)} of {len(episodes)} demos: dropped {num_failed} failed and "
          f"{len(episodes) - num_failed - len(keys)} repeated seeds")
    return keys


def load_traj_hdf5(path, num_traj=None):
    print("Loading HDF5 file", path)
    file = File(path, "r")
    keys = clean_traj_keys(path[:-len(".h5")] + ".json")
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = keys[:num_traj]
    ret = {key: load_content_from_h5_file(file[key]) for key in keys}
    file.close()
    print("Loaded")
    return ret


def load_demo_dataset(
    path, keys=["observations", "actions"], num_traj=None, concat=True
):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    # raw_data has keys like: ['traj_0', 'traj_1', ...]
    # raw_data['traj_0'] has keys like: ['actions', 'dones', 'env_states', 'infos', ...]
    _traj = next(iter(raw_data.values()))  # traj_0 may have been filtered out
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in the first traj: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [raw_data[idx][source_key] for idx in raw_data]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ["observations", "states"] and len(
                dataset[target_key][0]
            ) > len(_traj["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[:-1] for t in dataset[target_key]], axis=0
                )
            elif target_key in ["next_observations", "next_states"] and len(
                dataset[target_key][0]
            ) > len(_traj["actions"]):
                dataset[target_key] = np.concatenate(
                    [t[1:] for t in dataset[target_key]], axis=0
                )
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print("Load", target_key, dataset[target_key].shape)
        else:
            print(
                "Load",
                target_key,
                len(dataset[target_key]),
                type(dataset[target_key][0]),
            )
    return dataset


# box_hole_pose (p, then wxyz q) in PegInsertionSide's 43-dim state obs, after qpos 9, qvel 9, tcp_pose 7,
# peg_pose 7 and peg_half_size 3
HOLE_POSE = slice(35, 42)


def sample_hole_noise(n, device):
    """n hole-frame pose errors, (n, 7) as p then wxyz q

    the error model of the noisy evaluation: a shift uniform in [0, 6 mm] along a random direction in the hole's
    y-z plane (x is the insertion axis) and a rotation uniform in [0, 1 deg] about a uniformly random 3D axis
    """
    shift = torch.rand(n, device=device) * 0.006
    direction = torch.rand(n, device=device) * 2 * torch.pi
    p = torch.stack([torch.zeros_like(shift), shift * torch.cos(direction), shift * torch.sin(direction)], -1)
    angle = torch.rand(n, device=device) * np.deg2rad(1.0)
    axis = F.normalize(torch.randn(n, 3, device=device), dim=-1)
    q = torch.cat([torch.cos(angle / 2)[:, None], torch.sin(angle / 2)[:, None] * axis], -1)
    return torch.cat([p, q], -1)


def perturb_hole_pose(obs, noise):
    """obs (..., 43) whose observed hole pose is hole_pose * noise, the hole-frame composition noisy_env uses"""
    hole_p, hole_q = obs[..., HOLE_POSE][..., :3], obs[..., HOLE_POSE][..., 3:]
    p = hole_p + quaternion_apply(hole_q, noise[..., :3].expand_as(hole_p))
    q = quaternion_multiply(hole_q, noise[..., 3:].expand_as(hole_q))
    return torch.cat([obs[..., :HOLE_POSE.start], p, q, obs[..., HOLE_POSE.stop:]], -1)


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor, depth = True):
    img_dict = obs["sensor_data"]
    ls = ["rgb"]
    if depth:
        ls = ["rgb", "depth"]

    new_img_dict = {
        key: transpose_fn(
            concat_fn([v[key] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
        for key in ls
    }
    if "depth" in new_img_dict and isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
        new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
    }

    if "depth" in new_img_dict:
        out_dict["depth"] = new_img_dict["depth"]


    return out_dict


def build_obs_space(env, depth_dtype, state_obs_extractor):
    # NOTE: We have to use float32 for gym AsyncVecEnv since it does not support float16, but we can use float16 for MS2 vec env
    obs_space = env.observation_space

    # Unified version
    state_dim = sum([v.shape[0] for v in state_obs_extractor(obs_space)])

    single_img_space = next(iter(env.observation_space["image"].values()))
    h, w, _ = single_img_space["rgb"].shape
    n_images = len(env.observation_space["image"])

    return spaces.Dict(
        {
            "state": spaces.Box(
                -float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32
            ),
            "rgb": spaces.Box(0, 255, shape=(n_images * 3, h, w), dtype=np.uint8),
            "depth": spaces.Box(
                -float("inf"), float("inf"), shape=(n_images, h, w), dtype=depth_dtype
            ),
        }
    )


def build_state_obs_extractor(env_id):
    # NOTE: You can tune/modify state observations specific to each environment here as you wish. By default we include all data
    # but in some use cases you might want to exclude e.g. obs["agent"]["qvel"] as qvel is not always something you query in the real world.
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
