"""Print ppo.py runs side by side, and optionally save learning curves for the comparison plot.

    python scripts/curves.py runs/ppo_tune_*
    python scripts/curves.py runs/ppo_scratch_* --csv results/ppo_curves.csv
"""

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["eval/success_once", "eval/success_at_end", "eval/return", "eval/episode_len", "train/return",
        "train/success_once", "losses/entropy", "charts/SPS"]


def read_run(run_dir):
    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    return {tag: [(e.step, e.value) for e in events.Scalars(tag)] for tag in TAGS if tag in tags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", default=None, help="write run, step, tag, value rows here")
    ap.add_argument("--rows", type=int, default=6, help="evaluations to print per run")
    args = ap.parse_args()

    rows = []
    for run_dir in args.runs:
        run = read_run(run_dir)
        name = Path(run_dir).name
        for tag, points in run.items():
            rows += [(name, step, tag, value) for step, value in points]
        print(f"\n{name}")
        last = {tag: points[-1][1] for tag, points in run.items()}
        print("  last: " + "  ".join(f"{tag.split('/')[-1]}={value:.3f}" for tag, value in last.items() if tag.startswith("train") or tag.startswith("losses") or tag.startswith("charts")))
        success = dict(run.get("eval/success_once", []))
        returns = dict(run.get("eval/return", []))
        steps = sorted(returns)
        shown = steps[:: max(1, len(steps) // args.rows)][-args.rows:]
        for step in shown:
            print(f"  step {step:>11,}  eval success_once {success.get(step, float('nan')):.3f}  eval return {returns[step]:.2f}")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "step", "tag", "value"])
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
