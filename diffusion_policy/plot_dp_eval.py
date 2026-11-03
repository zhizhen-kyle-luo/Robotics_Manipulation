"""Plots for eval_start_states.py results (reads <out-dir>/*.csv, writes PNGs into <out-dir>).

  python plot_dp_eval.py --out-dir ../results/dp_eval --runs dp_full136_s0 dp_full136_s1 dp_full136_s2

  success_rates.png       grasped-start success per seed (95% Wilson CI) + mean across seeds (+-1 std)
  grasped_vs_fulltask.png grasped-start vs full-task success per seed (95% Wilson CI)
  steps_to_success.png    histogram of steps to first success from grasped starts
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from eval_start_states import wilson

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
MEAN = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
})


@dataclass
class PlotArgs:
    out_dir: str = "../results/dp_eval"
    runs: List[str] = field(default_factory=lambda: ["dp_full136_s0", "dp_full136_s1", "dp_full136_s2"])
    metric: str = "success_once"


def load(path):
    with open(path) as f:
        return [{k: (float(v) if k != "hole_offset_dir" and v != "" else v) for k, v in r.items()} for r in csv.DictReader(f)]


def rate_ci(rows, metric):
    k, n = int(sum(r[metric] for r in rows)), len(rows)
    lo, hi = wilson(k, n)
    return k / n, lo, hi, k, n


def label(run):
    return run.replace("dp_full136_", "seed ").replace("seed s", "seed ")


def bar_with_ci(ax, x, p, lo, hi, color, width):
    ax.bar(x, p, width, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=3)


def main(a: PlotArgs):
    metric = a.metric
    names = [label(r) for r in a.runs]
    grasp = {r: load(os.path.join(a.out_dir, f"{r}.csv")) for r in a.runs}
    full = {r: load(os.path.join(a.out_dir, f"{r}__fulltask.csv")) for r in a.runs
            if os.path.exists(os.path.join(a.out_dir, f"{r}__fulltask.csv"))}

    # 1. grasped-start success per seed + mean across seeds
    fig, ax = plt.subplots(figsize=(6.4, 4))
    rates = []
    for i, r in enumerate(a.runs):
        p, lo, hi, k, n = rate_ci(grasp[r], metric)
        rates.append(p)
        bar_with_ci(ax, i, p, lo, hi, SERIES[i], 0.6)
        ax.text(i, hi + 0.015, f"{100 * p:.1f}%\n({k}/{n})", ha="center", va="bottom", color=INK2, fontsize=9)
    m, sd = float(np.mean(rates)), float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    i = len(a.runs)
    ax.bar(i, m, 0.6, color=MEAN, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(i, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=3)
    ax.text(i, m + sd + 0.015, f"{100 * m:.1f}%\n±{100 * sd:.1f}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(range(len(a.runs) + 1), names + ["mean of seeds"])
    ax.set_ylim(0, min(1.0, max(0.25, max(rates) * 1.6 + 0.1)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel(f"{metric} rate")
    ax.set_title(f"DP from grasped start states ({len(grasp[a.runs[0]])} held-out states)", loc="left", fontsize=11)
    ax.text(0, -0.16, "Error bars: 95% Wilson interval per seed; ±1 std across seeds for the mean.",
            transform=ax.transAxes, color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "success_rates.png"), dpi=150)
    plt.close(fig)

    # 2. grasped-start vs full-task per seed
    if full:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        w = 0.36
        tops = []
        for i, r in enumerate(a.runs):
            for j, (rows, name) in enumerate([(grasp[r], "grasped start"), (full.get(r), "full task")]):
                if rows is None:
                    continue
                p, lo, hi, k, n = rate_ci(rows, metric)
                x = i + (j - 0.5) * w
                bar_with_ci(ax, x, p, lo, hi, SERIES[j], w)
                ax.text(x, hi + 0.015, f"{100 * p:.0f}%", ha="center", va="bottom", color=INK2, fontsize=9)
                tops.append(hi)
        ax.set_xticks(range(len(a.runs)), names)
        ax.set_ylim(0, min(1.0, max(tops) + 0.12))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylabel(f"{metric} rate")
        n_g, n_f = len(grasp[a.runs[0]]), len(next(iter(full.values())))
        handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
        ax.legend(handles, [f"grasped start (n={n_g}, 200 steps)", f"full task (n={n_f}, 300 steps)"],
                  frameon=False, loc="upper left", fontsize=9)
        ax.set_title("Grasped-start vs full-task success", loc="left", fontsize=11)
        ax.text(0, -0.16, "Error bars: 95% Wilson interval.", transform=ax.transAxes, color=INK2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "grasped_vs_fulltask.png"), dpi=150)
        plt.close(fig)

    # 3. steps to first success from grasped starts
    fig, ax = plt.subplots(figsize=(6.4, 4))
    max_len = int(max(r["episode_len"] for rows in grasp.values() for r in rows))
    bins = np.arange(0, max_len + 10, 10)
    steps = [[r["steps_to_first_success"] for r in grasp[run] if r["steps_to_first_success"] >= 0] for run in a.runs]
    ax.hist(steps, bins=bins, color=SERIES[:len(a.runs)], edgecolor=SURFACE, linewidth=1, stacked=True,
            label=[f"{n} (n={len(s)})" for n, s in zip(names, steps)], zorder=2)
    allsteps = np.concatenate([s for s in steps if len(s)]) if any(len(s) for s in steps) else np.array([])
    if len(allsteps):
        med = float(np.median(allsteps))
        ax.axvline(med, color=INK, linewidth=1, linestyle="--", zorder=3)
        ax.text(med + 2, ax.get_ylim()[1] * 0.95, f"median {med:.0f}", color=INK2, fontsize=9, va="top")
    ax.set_xlabel("steps to first success (from grasped start)")
    ax.set_ylabel("episodes")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("Steps to first success, successful episodes only (stacked by seed)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "steps_to_success.png"), dpi=150)
    plt.close(fig)
    print("wrote", [f for f in os.listdir(a.out_dir) if f.endswith(".png")])


if __name__ == "__main__":
    main(tyro.cli(PlotArgs))
