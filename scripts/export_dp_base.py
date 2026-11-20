"""Export a diffusion_policy/train.py checkpoint as the frozen base residual.DiffusionBase loads.

    uv run python scripts/export_dp_base.py diffusion_policy/runs/dp_clean_s0/checkpoints/final.pt base_ckpts/dp_clean_s0.pt

Keeps the EMA weights (with the obs_mean/obs_std buffers of a --norm-stats-path run) and adds the config
DiffusionBase needs. train.py checkpoints do not store their Args, so pass any architecture flag the run changed;
the defaults are train.py's. The export is loaded once at the end, so a config that does not match the weights
fails here instead of in the residual run.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residual import DiffusionBase


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--obs-horizon", type=int, default=2)
    ap.add_argument("--act-horizon", type=int, default=8)
    ap.add_argument("--pred-horizon", type=int, default=16)
    ap.add_argument("--diffusion-step-embed-dim", type=int, default=64)
    ap.add_argument("--unet-dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--n-groups", type=int, default=8)
    args = ap.parse_args()

    # train.py checkpoints also hold optimizer and RNG state, which need the full unpickler
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(obs_horizon=args.obs_horizon, act_horizon=args.act_horizon, pred_horizon=args.pred_horizon,
                  diffusion_step_embed_dim=args.diffusion_step_embed_dim, unet_dims=args.unet_dims,
                  n_groups=args.n_groups)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(ema_agent=ckpt["ema_agent"], config=config), args.out)
    base = DiffusionBase(args.out, 1, "cpu")
    print(f"wrote {args.out}: obs_dim {base.obs_dim}, act_dim {base.act_dim}, "
          f"normalized {'obs_mean' in ckpt['ema_agent']}")


if __name__ == "__main__":
    main()
