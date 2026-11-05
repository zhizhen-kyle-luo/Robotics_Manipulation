"""Cluster smoke test: do saved start states survive GPU physics, and how fast is the simulator?

    python scripts/gpu_smoke.py --starts start_states/eval/states.pt --num-envs 1024
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import start_states
from evaluate import make_env


def hold_test(path, steps=20):
    """The states were generated on CPU physics. Restore them all on the GPU, hold still with the
    gripper closed, and report how many grasps survive. Training is meaningless if they do not."""
    starts = start_states.load(path)
    n = len(starts["seeds"])
    env = make_env(num_envs=n, sim_backend="physx_cuda")
    u = env.unwrapped
    start_states.reset_to(env, starts, torch.arange(n))
    p0 = u.peg.pose.p.clone()
    hold = torch.zeros((n, 7), device=u.device)
    hold[:, -1] = -1
    for _ in range(steps):
        env.step(hold)
    grasped = u.agent.is_grasping(u.peg, max_angle=20)
    drift = (u.peg.pose.p - p0).norm(dim=1)
    print(f"hold test ({n} states, {steps} steps on GPU physics): grasped {grasped.float().mean().item():.3f}, "
          f"peg drift mean {drift.mean().item() * 1000:.1f} mm, max {drift.max().item() * 1000:.1f} mm")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", default=None, help="saved start states for the hold test")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        # Sut's note: the locked torch wheels are built for CUDA 13.2 and need NVIDIA driver >= 580
        raise SystemExit("torch cannot see a GPU; compare the driver version above with what the torch wheel needs")
    print("cuda", torch.cuda.get_device_name(0))
    if args.starts is not None:
        hold_test(args.starts)

    env = make_env(num_envs=args.num_envs, sim_backend="physx_cuda", obs_mode="state", hole_offset_range=(0.0, 0.004))
    u = env.unwrapped
    obs, _ = env.reset(seed=0)
    print("obs", tuple(obs.shape), "action", env.action_space.shape, "device", obs.device)
    off = u.hole_offsets[:, 1:].norm(dim=1)
    print(f"training hole noise: max {off.max().item() * 1000:.2f} mm, mean {off.mean().item() * 1000:.2f} mm")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        obs, rew, term, trunc, info = env.step(2 * torch.rand(env.action_space.shape, device=u.device) - 1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = args.num_envs * args.steps / dt
    print(f"{sps:,.0f} env steps/s -> 75M steps in {75e6 / sps / 3600:.2f} h of pure simulation")
    env.close()


if __name__ == "__main__":
    main()
