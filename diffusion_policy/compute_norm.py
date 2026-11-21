"""Write norm_stats.json for train.py --norm-stats-path, from the same demos train.py loads

the original script was never committed and its stats covered all 1000 planner episodes; loading through
load_demo_dataset applies clean_traj_keys, so the stats match the 649 demos the policy is trained on
"""

import argparse
import json
import os

import numpy as np

from diffusion_policy.utils import load_demo_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-std", type=float, default=0.01, help="std floor for near-constant dims")
    ap.add_argument("--include-actions", action="store_true")
    args = ap.parse_args()

    keys = ["observations", "actions"] if args.include_actions else ["observations"]
    data = load_demo_dataset(args.demo_path, keys=keys)
    stats = {}
    for name, key in [("state", "observations"), ("action", "actions")]:
        if key in data:
            x = data[key].astype(np.float64)
            stats[name] = {"mean": x.mean(0).tolist(), "std": np.maximum(x.std(0), args.min_std).tolist()}
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "norm_stats.json")
    with open(out, "w") as f:
        json.dump({"norm_stats": stats}, f, indent=2)
    print(f"wrote {out} from {len(data['observations'])} steps")


if __name__ == "__main__":
    main()
