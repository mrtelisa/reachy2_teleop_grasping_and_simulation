#!/usr/bin/env python3
"""
Plot the path driven in a session from its odometry log
(<subject>_run<N>_odometry.csv written by session_metrics.py, in results_robot/)
to a PNG.

Top-down view in the odometry frame: x = forward, y = left. The path is coloured by driving mode (max speed / reduced speed /
repositioning / transport), with the start, the end and the heading every few seconds.

Usage:
    python3 plot_odometry.py results_robot/S000_run1_odometry.csv [-o path.png] [--arrows-every 2]
The PNG is saved next to the csv (<name>_path.png) unless -o is given.
"""

import argparse
import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Driving modes in the order of the test, one categorical colour each
MODE_LABELS = (("max_speed", "max speed"), ("reduced_speed", "reduced speed"), ("repositioning", "repositioning"),
               ("transport", "transport"))
MODE_COLORS = {"max_speed": "#2a78d6", "reduced_speed": "#eb6834", "repositioning": "#1baf7a", "transport": "#8a4fd8"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load_odometry(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no samples")
    return {
        "t": np.array([float(r["t_test"]) for r in rows]),
        "x": np.array([float(r["x"]) for r in rows]),
        "y": np.array([float(r["y"]) for r in rows]),
        "theta": np.array([float(r["theta"]) if r["theta"] else np.nan for r in rows]),
        "mode": [r["mode"] for r in rows],
    }


def session_summary(csv_path: str) -> dict:
    """The <subject>_run<N>_session.json next to the csv, {} if missing."""
    json_path = csv_path[:-len("_odometry.csv")] + "_session.json" if csv_path.endswith("_odometry.csv") else None
    if json_path and os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def plot_path(odom: dict, summary: dict, out_png: str, arrows_every_s: float) -> None:
    # Page axes: horizontal = -y (y is "left" in the base frame, so left stays left), vertical = x (forward)
    px, py = -odom["y"], odom["x"]
    modes = odom["mode"]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # One polyline per run of the same mode (kept contiguous by sharing the boundary sample)
    present = []
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            mode = modes[start]
            end = min(i + 1, len(modes))
            ax.plot(px[start:end], py[start:end], color=MODE_COLORS.get(mode, INK), linewidth=2,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=dict(MODE_LABELS).get(mode, mode) if mode not in present else None)
            if mode not in present:
                present.append(mode)
            start = i

    # Heading every arrows_every_s seconds (theta = 0 is "up", positive = counter-clockwise)
    if arrows_every_s > 0 and not np.all(np.isnan(odom["theta"])):
        next_t = odom["t"][0]
        for t, x, y, th in zip(odom["t"], px, py, odom["theta"]):
            if t >= next_t and not math.isnan(th):
                ax.annotate("", xy=(x - 0.12 * math.sin(th), y + 0.12 * math.cos(th)), xytext=(x, y),
                            arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.0, mutation_scale=9))
                next_t += arrows_every_s

    ax.plot(px[0], py[0], "o", color=INK, markersize=8, markerfacecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.plot(px[-1], py[-1], "s", color=INK, markersize=8, zorder=5)
    ax.annotate("start", (px[0], py[0]), textcoords="offset points", xytext=(8, -12), color=INK, fontsize=9)
    ax.annotate("end", (px[-1], py[-1]), textcoords="offset points", xytext=(8, 6), color=INK, fontsize=9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.15)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel("left  <-  -y [m]  ->  right", color=INK_2)
    ax.set_ylabel("x [m]  (forward)", color=INK_2)

    subject = summary.get("subject", os.path.basename(out_png).split("_")[0])
    parts = [f"{subject}"]
    if summary.get("timestamp"):
        parts.append(summary["timestamp"])
    title = " - ".join(parts)
    details = f"{odom['t'][-1] - odom['t'][0]:.0f} s, {len(modes)} samples"
    if "path_length_total" in summary:
        details = f"path {summary['path_length_total']:.2f} m, " + details
    ax.set_title(f"{title}\n{details}", color=INK, fontsize=11, loc="left")
    if len(present) >= 2:
        ax.legend(loc="best", frameon=False, labelcolor=INK_2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="<subject>_run<N>_odometry.csv of the session")
    parser.add_argument("-o", "--out", default=None, help="Output PNG (default: <csv name>_path.png next to it)")
    parser.add_argument("--arrows-every", type=float, default=2.0,
                        help="Seconds between heading arrows (default: 2, 0 = none)")
    args = parser.parse_args()

    out_png = args.out or (args.csv[:-4] if args.csv.endswith(".csv") else args.csv) + "_path.png"
    odom = load_odometry(args.csv)
    plot_path(odom, session_summary(args.csv), out_png, args.arrows_every)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
