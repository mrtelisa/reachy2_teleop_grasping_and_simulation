#!/usr/bin/env python3
"""
Fullscreen cursor reaching test, RANDOM sequence -- no robot, no socket.

Same test as reaching_center_out.py (same hand -> cursor chain, canvas,
target size, dwell, 10 s trial timeout, session timer, score, metrics and
result files), but without the returns to the centre: after the first goal
at the centre (which starts the session timer), the 248 targets are shown one
after the other, each somewhere on the screen.

The target positions are the 248 of config/cursor_targets.csv (the 8 circle
targets of the center-out test in seeded random order, every target before
any repeat; the center-out test only uses the first 64), in the same order:
fixed, identical for every participant and every launch.

Results (a subject with previous sessions gets _1, _2, ... appended):
  results_random/<subject>_random_trials.csv
  results_random/<subject>_random_summary.json

Usage:
    python3 reaching_random.py --calib <name> --subject S001 [--cam 0] [--max-minutes 4]
Keys: Q / ESC = abort (results so far are still saved).
"""

import os

import reaching_center_out as base

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_random")


def build_trials() -> list:
    """One goal at the centre, then all the N_TARGETS (248) targets of the
    frozen sequence in the same order, with no return to the centre in between."""
    center_out = base.build_trials(base.N_TARGETS)
    home = next(t for t in center_out if t["kind"] == "home")
    targets = [t for t in center_out if t["kind"] == "target"]
    return [home] + targets


if __name__ == "__main__":
    base.main(build=build_trials, results_dir=RESULTS_DIR, sequence="random", description=__doc__)
