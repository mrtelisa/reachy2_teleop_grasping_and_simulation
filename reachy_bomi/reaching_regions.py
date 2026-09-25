#!/usr/bin/env python3
"""
Fullscreen 9-region blind reaching test, in two parts separated by a pause.

The hand drives an on-screen cursor through the BoMI chain of bomi.py
(webcam -> MediaPipe -> autoencoder map -> Butterworth filter); the map's
BASE_WIDTH x BASE_HEIGHT space is scaled onto a 1200 x 650 canvas, itself
scaled to the whole screen. The canvas is divided in the 9 regions of the
teleoperation interface (3 x 3 grid):
    1 | 2 | 3   (top row)
    4 | 5 | 6   (middle row)
    7 | 8 | 9   (bottom row)
Region 5 is the HOME: a circle at the screen centre; the target of every
outer region is a circle at the centre of that region. All of them have the
same radius (TARGET_RADIUS).

Sequence: config/cursor_regions.csv, one region per row, e.g.
    5, 1, 5, 2, 5, 3 ...   = home, region 1, home, region 2, ...
(any other file with --sequence). It is generated once if missing: every
outer region N_REPETITIONS (12) times -> 96 targets, in seeded random order
(all 8 once before any repeat, never the same twice in a row), a home before
each.

Session: N_PARTS (2) parts, each the whole sequence (96 targets):
  part 1 -> back to the home -> PAUSE_S (3 min) pause -> part 2
During the pause no goal is shown and nothing is recorded: the cursor is
visible, with the time left. After it the home is shown again and part 2
starts the first time the cursor enters it. The session ends when the last
target of part 2 is reached (or on Q/ESC).

Trial flow:
  home goal   -> the home circle is shown (no target).
                 reach_time = circle shown -> entering it.
  region goal -> starts the moment the home is reached: the home disappears
                 and the target circle at the centre of the region is shown
                 (yellow). For the first HIDDEN_S (1 s) the cursor is NOT
                 drawn (no visual feedback: can the participant reach the
                 target from the map alone?), then it reappears.
                 reach_time = target shown -> entering the circle.
Every goal (home or region) is reached when the cursor stays inside its
circle DWELL_S (0.5 s), hidden or not; leaving it restarts the dwell, and
reach_time counts to the entry that completed the dwell. A circle turns blue
while the cursor is inside it (a target only once the cursor is visible).
The timer of a part starts when the cursor enters its home for the first time
and stops at its last goal. The session timer on screen is part 1 + part 2:
it stops when the pause starts and stays still until the home is entered
again. The summary has time_total, time_part_1 and time_part_2.

Metrics of every goal: the reaching_metrics.py kinematics
(reaction_time, normalized_path_length, dimensionless_jerk, n_speed_peaks,
...; the ideal target point of a region is its centre). Region goals also get:
  reached_hidden            100 if the circle was entered while the cursor was
                            still hidden (entry of the completed dwell), else 0
  region_at_reveal          region of the cursor when it reappears (the target
                            if it was reached while hidden)
  region_at_reveal_correct  100 if that is the target, else 0
  first_region              first region (outside the centre) visited, i.e.
                            stayed in for MIN_VISIT_S (the target counts at once)
  first_region_correct      100 if it was the target, else 0
  n_wrong_regions           distinct other regions visited before the target
Averages (mean_* in the summary) over all region goals, per part (before vs
after the pause, and the difference part 2 - part 1), per region, and per
block of BLOCK_SIZE (8) consecutive targets = one repetition of every region
(12 blocks per part, 24 in all: learning curve).

Results (a subject with previous sessions gets _1, _2, ... appended):
  results_regions/<subject>_regions_trials.csv       one row per goal (with its part)
  results_regions/<subject>_regions_blocks.csv       one row per block of 8 targets
  results_regions/<subject>_regions_trajectory.csv   every cursor sample (part, trial, t, x, y, cursor_visible)
  results_regions/<subject>_regions_summary.json     means (all, per part, part 2 vs 1, per region, per block), config

Usage:
    python3 reaching_regions.py --subject S001 [--calib <name>] [--cam 0] [--sequence file.csv]
Keys: Q / ESC = abort (results so far are still saved).
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np

import bomi
from reaching_metrics import METRIC_KEYS, block_summaries, compute_trial_metrics, summarize

# --- Geometry (canvas px, markerlessBoMI reaching.py) ---
CANVAS_W, CANVAS_H = 1200, 650
CRS_RADIUS = 15
REGION_X = (0, CANVAS_W / 3.0, 2 * CANVAS_W / 3.0, CANVAS_W)   # column boundaries (400 px wide)
REGION_Y = (0, CANVAS_H / 3.0, 2 * CANVAS_H / 3.0, CANVAS_H)   # row boundaries (216.7 px high)
HOME_REGION = 5
TARGET_RADIUS = 70       # target circle at the centre of each outer region
HOME_RADIUS = TARGET_RADIUS   # the home circle is as large as the targets
REGION_NAMES = {1: "top-left", 2: "top", 3: "top-right", 4: "left", 5: "centre (home)",
                6: "right", 7: "bottom-left", 8: "bottom", 9: "bottom-right"}

# --- Sequence ---
SEQUENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "cursor_regions.csv")
N_REPETITIONS = 12       # default sequence: every outer region this many times (8 x 12 = 96 targets)
SEQUENCE_SEED = 20
BLOCK_SIZE = 8           # statistics per block of 8 consecutive targets (= one repetition of every region)
N_PARTS = 2              # the whole sequence this many times, with a pause in between

# --- Timing ---
DWELL_S = 0.5            # the cursor must stay in a goal circle (home or target) this long
PAUSE_S = 180.0          # pause between two parts
HIDDEN_S = 1.0           # the cursor is not drawn for this long after the target is shown
MIN_VISIT_S = 0.25       # a region counts as visited if the cursor stays in it this long

# --- Metrics (pixels of the canvas) ---
MOTION_ONSET_SPEED = 40.0     # [px/s]
SPEED_PEAK_THRESHOLD = 80.0   # [px/s]
RESAMPLE_HZ = 50.0
BLIND_KEYS = ("reached_hidden", "region_at_reveal", "region_at_reveal_correct",
              "first_region", "first_region_correct", "n_wrong_regions")
# Averaged in the summary (region numbers are not)
REGION_KEYS = METRIC_KEYS + tuple(k for k in BLIND_KEYS if k not in ("region_at_reveal", "first_region"))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_regions")
WINDOW = "BoMI - Reaching (regions)"

# Colours (BGR)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 80, 0)
GRID = (90, 90, 90)
TARGET = (0, 255, 255)   # target circle
CURSOR = (int(0.4 * 255), int(0.65 * 255), int(0.19 * 255))   # markerlessBoMI CURSOR, RGB -> BGR


def region_of(x: float, y: float) -> int:
    """Region 1..9 of a canvas point (3 x 3 grid, row-major from the top-left)."""
    col = 0 if x < REGION_X[1] else 1 if x < REGION_X[2] else 2
    row = 0 if y < REGION_Y[1] else 1 if y < REGION_Y[2] else 2
    return row * 3 + col + 1


def region_bounds(region: int) -> tuple:
    """(x0, y0, x1, y1) of a region in canvas px."""
    row, col = divmod(region - 1, 3)
    return REGION_X[col], REGION_Y[row], REGION_X[col + 1], REGION_Y[row + 1]


def inside_goal(trial: dict, x: float, y: float) -> bool:
    """True if (x, y) is inside the circle of the goal (home or target)."""
    radius = HOME_RADIUS if trial["kind"] == "home" else TARGET_RADIUS
    return math.hypot(x - trial["x"], y - trial["y"]) < radius


def visited_regions(samples: list, target_region: int) -> list:
    """Regions (outside the centre) visited in order by samples (t, x, y): the
    cursor must stay in a region MIN_VISIT_S for it to count, except the
    target, which counts at once (crossing the corner of a side region on the
    diagonal to a corner is not an error)."""
    regions = [region_of(x, y) for _, x, y in samples]
    times = [t for t, _, _ in samples]
    visited, i = [], 0
    while i < len(regions):
        j = i
        while j + 1 < len(regions) and regions[j + 1] == regions[i]:
            j += 1
        r = regions[i]
        if r != HOME_REGION and (r == target_region or times[j] - times[i] >= MIN_VISIT_S):
            if not visited or visited[-1] != r:
                visited.append(r)
        i = j + 1
    return visited


def blind_metrics(samples: list, target_region: int, t_shown: float, t_enter: float) -> dict:
    """Region-level metrics of a region goal (see the module docstring)."""
    m = {k: None for k in BLIND_KEYS}
    if not samples:
        return m
    t_reveal = t_shown + HIDDEN_S
    if t_enter is not None:
        m["reached_hidden"] = 100.0 if t_enter < t_reveal else 0.0
    if t_enter is not None and t_enter < t_reveal:
        m["region_at_reveal"] = target_region
    else:
        at_reveal = [(t, x, y) for t, x, y in samples if t <= t_reveal]
        if at_reveal and samples[-1][0] >= t_reveal:
            m["region_at_reveal"] = region_of(at_reveal[-1][1], at_reveal[-1][2])
    if m["region_at_reveal"] is not None:
        m["region_at_reveal_correct"] = 100.0 if m["region_at_reveal"] == target_region else 0.0
    visited = visited_regions(samples, target_region)
    if visited:
        m["first_region"] = visited[0]
        m["first_region_correct"] = 100.0 if visited[0] == target_region else 0.0
    if target_region in visited:
        m["n_wrong_regions"] = len(set(visited[:visited.index(target_region)]))
    return m


def generate_sequence(repetitions: int = N_REPETITIONS, seed: int = SEQUENCE_SEED) -> list:
    """[5, r1, 5, r2, 5, ...]: the 8 outer regions `repetitions` times in
    seeded random order (all of them once before any repeat, never the same
    twice in a row), a home before each. Only used to create SEQUENCE_FILE once."""
    rng = np.random.default_rng(seed)
    outer = [r for r in range(1, 10) if r != HOME_REGION]
    order, last = [], None
    for _ in range(repetitions):
        perm = [outer[i] for i in rng.permutation(len(outer))]
        if perm[0] == last:
            perm.append(perm.pop(0))
        order += perm
        last = order[-1]
    seq = []
    for r in order:
        seq += [HOME_REGION, r]
    return seq


def load_sequence(path: str = SEQUENCE_FILE) -> list:
    """The regions to reach, one per row of `path` (a header line and blank
    lines are ignored). The default file is generated once if missing."""
    if path == SEQUENCE_FILE and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("region\n" + "".join(f"{r}\n" for r in generate_sequence()))
        print(f"[sequence] generated the default sequence -> {path} (commit this file)")
    seq = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            s = line.strip().strip(",")
            if not s or not s.lstrip("-").isdigit():
                continue   # header / blank
            r = int(s)
            if not 1 <= r <= 9:
                raise ValueError(f"{path} line {n}: region must be 1..9, got {r}")
            seq.append(r)
    if not seq:
        raise ValueError(f"{path}: no regions found")
    return seq


def build_trials(sequence: list, parts: int = N_PARTS) -> list:
    """One goal per region of the sequence, the whole sequence `parts` times:
    kind ("home" for region 5, "region" otherwise), region, part (1..parts),
    target_number (the region goals counted 1..n over the whole session; a
    home takes the number of the region goal that follows it),
    part_target_number (the same, counted within its part), x, y (goal
    centre: the home, or the centre of the region). Every part but the last
    ends with a home (the return from its last target) marked pause_after."""
    trials, n_targets = [], 0
    for part in range(1, parts + 1):
        first = n_targets
        for r in sequence:
            if r == HOME_REGION:
                trials.append({"kind": "home", "region": r, "part": part, "target_number": n_targets + 1,
                               "part_target_number": n_targets - first + 1,
                               "x": CANVAS_W / 2.0, "y": CANVAS_H / 2.0})
            else:
                n_targets += 1
                x0, y0, x1, y1 = region_bounds(r)
                trials.append({"kind": "region", "region": r, "part": part, "target_number": n_targets,
                               "part_target_number": n_targets - first,
                               "x": (x0 + x1) / 2.0, "y": (y0 + y1) / 2.0})
        if part < parts:
            trials.append({"kind": "home", "region": HOME_REGION, "part": part, "target_number": n_targets,
                           "part_target_number": n_targets - first,
                           "x": CANVAS_W / 2.0, "y": CANVAS_H / 2.0, "pause_after": True})
    return trials


def compare_parts(first: dict, second: dict, keys=REGION_KEYS) -> dict:
    """{metric: {part_1, part_2, delta}} of two summarize() results (delta = part 2 - part 1)."""
    out = {}
    for k in ("success_rate",) + tuple(f"mean_{k}" for k in keys):
        x, y = first.get(k), second.get(k)
        out[k] = {"part_1": x, "part_2": y, "delta": (y - x) if x is not None and y is not None else None}
    return out


def session_name(subject: str, sequence: str, results_dir: str) -> str:
    """<subject>_<sequence> for the first session, then <subject>_<sequence>_1,
    _2, ... (any existing <results_dir>/<subject>_<sequence>* file counts as a
    previous session)."""
    prefix = f"{subject}_{sequence}"
    existing = [f for f in os.listdir(results_dir) if f.startswith(prefix)]
    if not existing:
        return prefix
    used = {int(m.group(1)) for f in existing
            for m in [re.match(rf"{re.escape(prefix)}_(\d+)_(trials\.csv|summary\.json)$", f)] if m}
    return f"{prefix}_{max(used, default=0) + 1}"


class RegionsTest:
    def __init__(self, subject: str, trials: list, results_dir: str = RESULTS_DIR,
                 sequence: str = "regions") -> None:
        self.subject = subject
        self.trials = trials
        self.sequence = sequence
        self.results = []
        self.trajectory = []   # every cursor sample: (part, trial, kind, region, t, x, y, cursor_visible)
        self.n_parts = max(tr["part"] for tr in trials)

        os.makedirs(results_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.base = os.path.join(results_dir, session_name(subject, sequence, results_dir))

        # Session/trial state (times are time.time())
        self.t_start = None
        self.t_session0 = None
        self.t_part0 = {}       # {part: first entry into the home of that part}
        self.t_part_end = {}    # {part: its last goal reached}
        self.parts_done = set() # parts whose timer has stopped (pause started / session completed)
        self.t_pause = None     # (start, end) of the pause
        self.pause_end = None   # while paused: when the pause ends
        self.trial_i = -1
        self.trial = None
        self.t_shown = None   # goal shown: disc (home) / target circle (region)
        self.t_enter = None
        self.samples = []
        self.end_reason = None

    # --- trial flow ---
    def start(self, t: float) -> None:
        self.t_start = t
        self._next_trial(t)

    def _next_trial(self, t: float) -> None:
        self.trial_i += 1
        if self.trial_i >= len(self.trials):
            self.parts_done.add(self.trials[-1]["part"])
            self.end_reason = "completed"
            return
        self.trial = self.trials[self.trial_i]
        self.t_shown = t
        self.t_enter = None
        self.samples = []

    def active(self) -> bool:
        return self.trial is not None and not self.end_reason

    def paused(self) -> bool:
        return self.pause_end is not None

    def cursor_visible(self, t: float) -> bool:
        """The cursor is hidden for the first HIDDEN_S of a region goal."""
        return not (self.active() and self.trial["kind"] == "region" and t - self.t_shown < HIDDEN_S)

    def update(self, t: float, x: float, y: float) -> None:
        """One frame with the current (filtered) cursor position in canvas px."""
        if self.paused():
            if t < self.pause_end:
                return   # nothing is recorded during the pause
            self.t_pause = (self.pause_end - PAUSE_S, self.pause_end)
            self.pause_end = None
            print(f"  pause over: part {self.trials[self.trial_i + 1]['part']} starts at the first entry into the home")
            self._next_trial(t)
        if not self.active():
            return
        tr = self.trial
        self.trajectory.append((tr["part"], self.trial_i + 1, tr["kind"], tr["region"], t, x, y,
                                int(self.cursor_visible(t))))
        self.samples.append((t, x, y))

        if not inside_goal(tr, x, y):
            self.t_enter = None   # left the circle: the dwell restarts
            return
        if self.t_enter is None:
            self.t_enter = t
            if tr["kind"] == "home" and tr["part"] not in self.t_part0:
                self.t_part0[tr["part"]] = t   # first entry into the home: the part (and session) timer starts
                if self.t_session0 is None:
                    self.t_session0 = t
                print(f"  home entered: part {tr['part']} timer started")
        if t - self.t_enter >= DWELL_S:
            self._end_trial(t)

    def part_time(self, part: int, t: float) -> float:
        """Time of a part: first entry into its home -> its last goal reached
        (t while it is still running; 0 before it starts)."""
        t0 = self.t_part0.get(part)
        if t0 is None:
            return 0.0
        return (self.t_part_end[part] if part in self.parts_done else t) - t0

    def elapsed(self, t: float) -> float:
        """The session timer: the parts only, so it stops when the pause starts
        and resumes at the first entry into the home after it."""
        return sum(self.part_time(p, t) for p in self.t_part0)

    def t0(self) -> float:
        """Time origin for the logs: session start if it has begun, else the launch."""
        return self.t_session0 if self.t_session0 is not None else self.t_start

    def _end_trial(self, t: float) -> None:
        self._record_trial(t, success=True, reason="reached")
        r = self.results[-1]
        if self.trial["kind"] == "region":
            hidden = "  (cursor hidden)" if r["reached_hidden"] else ""
            print(f"  part {r['part']} target {r['part_target_number']}: region {r['region']} "
                  f"({REGION_NAMES[r['region']]}) reached in {r['reach_time']:.2f}s{hidden}")
        self.t_part_end[self.trial["part"]] = t
        if self.trial.get("pause_after"):
            self.parts_done.add(self.trial["part"])   # its timer stops here, the next one starts in its home
            self.trial = None
            self.pause_end = t + PAUSE_S
            print(f"\n  === PAUSE {PAUSE_S / 60:.0f} min === (nothing recorded)")
            return
        self._next_trial(t)

    def _record_trial(self, t: float, success: bool, reason: str) -> None:
        tr = self.trial
        t_enter = self.t_enter if success else None
        metrics = compute_trial_metrics(self.samples, (tr["x"], tr["y"]), self.t_shown, t_enter,
                                        MOTION_ONSET_SPEED, SPEED_PEAK_THRESHOLD, RESAMPLE_HZ)
        blind = blind_metrics(self.samples, tr["region"], self.t_shown, t_enter) \
            if tr["kind"] == "region" else {k: None for k in BLIND_KEYS}
        rel = lambda ts: (ts - self.t0()) if ts is not None else None
        self.results.append({
            "part": tr["part"], "trial": self.trial_i + 1, "kind": tr["kind"], "region": tr["region"],
            "target_number": tr["target_number"], "part_target_number": tr["part_target_number"],
            "goal_x": tr["x"], "goal_y": tr["y"],
            "success": success, "end_reason": reason,
            # Session times (s from the session start): goal shown, entered, trial end
            "t_shown": rel(self.t_shown),
            "t_enter": rel(t_enter),
            "t_end": rel(t),
            "trial_duration": t - self.t_shown,
            **metrics,   # reach_time = t_enter - t_shown
            **blind,     # region goals only
        })

    def finish(self, t: float, reason: str = None) -> dict:
        """Close the current (unfinished) goal as missed (only on an abort) and write the results."""
        reason = reason or self.end_reason or "aborted"
        if self.active() and reason != "completed":
            self._record_trial(t, success=False, reason=reason)
        targets = [r for r in self.results if r["kind"] == "region"]
        homes = [r for r in self.results if r["kind"] == "home"]
        blocks = block_summaries(targets, BLOCK_SIZE, keys=REGION_KEYS)
        part_of_target = {tr["target_number"]: tr["part"] for tr in self.trials if tr["kind"] == "region"}
        for blk in blocks:
            blk["part"] = part_of_target.get(blk["first_target"])

        def per_region(rows):
            return {str(reg): summarize([r for r in rows if r["region"] == reg], keys=REGION_KEYS)
                    for reg in sorted({r["region"] for r in rows})}

        parts = {}
        for p in range(1, self.n_parts + 1):
            p_targets = [r for r in targets if r["part"] == p]
            parts[str(p)] = {
                "n_targets_total": sum(1 for tr in self.trials if tr["kind"] == "region" and tr["part"] == p),
                "regions": summarize(p_targets, keys=REGION_KEYS),
                "homes": summarize([r for r in homes if r["part"] == p]),
                "per_region": per_region(p_targets),
            }
        summary = {
            "subject": self.subject, "sequence": self.sequence, "end_reason": reason,
            "n_trials_total": len(self.trials),
            "n_targets_total": sum(1 for tr in self.trials if tr["kind"] == "region"),
            # Timer of the session (what is on screen): part 1 + part 2, without the pause
            # and the wait for the first entry into the home after it
            "time_total": self.elapsed(t),
            **{f"time_part_{p}": self.part_time(p, t) for p in range(1, self.n_parts + 1)},
            # first entry into the home -> end, pause included
            "wall_clock_duration": (t - self.t_session0) if self.t_session0 is not None else 0.0,
            "pause": ({"start": self.t_pause[0] - self.t0(), "end": self.t_pause[1] - self.t0(),
                       "duration": self.t_pause[1] - self.t_pause[0]} if self.t_pause else None),
            "timestamp": self.timestamp,
            # centre -> target region (what the test measures), both parts
            "regions": summarize(targets, keys=REGION_KEYS),
            "homes": summarize(homes),      # region -> back to the centre
            # Same statistics per part (before / after the pause) ...
            "parts": parts,
            # ... and their difference (part 2 - part 1)
            "part_2_vs_part_1": (compare_parts(parts["1"]["regions"], parts["2"]["regions"])
                                 if "1" in parts and "2" in parts else None),
            # ... per region, over its repetitions in both parts
            "per_region": per_region(targets),
            # ... and per block of BLOCK_SIZE consecutive targets (learning curve, numbered over the session)
            "block_size": BLOCK_SIZE,
            "n_blocks": len(blocks),
            "blocks": blocks,
            "missed": [{"part": r["part"], "trial": r["trial"], "kind": r["kind"], "region": r["region"],
                        "reason": r["end_reason"]}
                       for r in self.results if not r["success"]],
            "config": {
                "canvas": [CANVAS_W, CANVAS_H], "crs_radius": CRS_RADIUS, "home_radius": HOME_RADIUS,
                "target_radius": TARGET_RADIUS,
                "region_sequence": [tr["region"] for tr in self.trials if not tr.get("pause_after")
                                    and tr["part"] == 1],
                "n_parts": self.n_parts, "pause_s": PAUSE_S,
                "dwell_s": DWELL_S, "hidden_s": HIDDEN_S, "min_visit_s": MIN_VISIT_S,
                "motion_onset_speed": MOTION_ONSET_SPEED, "speed_peak_threshold": SPEED_PEAK_THRESHOLD,
            },
        }
        if self.results:
            with open(self.base + "_trials.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
                w.writeheader()
                w.writerows(self.results)
        if blocks:
            with open(self.base + "_blocks.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for b in blocks for k in b)))
                w.writeheader()
                w.writerows(blocks)
        if self.trajectory:
            with open(self.base + "_trajectory.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["part", "trial", "kind", "region", "t", "x", "y", "cursor_visible"])
                w.writerows((p, i, k, reg, f"{ts - self.t0():.4f}", f"{x:.2f}", f"{y:.2f}", vis)
                            for p, i, k, reg, ts, x, y, vis in self.trajectory)
        with open(self.base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


# --- Drawing ---
class Screen:
    """Fullscreen window; the canvas is scaled uniformly and centred."""

    def __init__(self, title: str = WINDOW) -> None:
        self.window = title
        self.w, self.h = bomi.open_fullscreen_window(self.window)
        self.scale = min(self.w / CANVAS_W, self.h / CANVAS_H)
        self.ox = (self.w - CANVAS_W * self.scale) / 2.0
        self.oy = (self.h - CANVAS_H * self.scale) / 2.0

    def to_screen(self, x: float, y: float) -> tuple:
        return int(self.ox + x * self.scale), int(self.oy + y * self.scale)

    def px(self, r: float) -> int:
        return max(1, int(round(r * self.scale)))

    def draw(self, test: RegionsTest, crs_x: float, crs_y: float, t: float, hand_detected: bool) -> None:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        x0, y0 = self.to_screen(0, 0)
        x1, y1 = self.to_screen(CANVAS_W, CANVAS_H)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), GRID, 1)
        for x in REGION_X[1:3]:
            cv2.line(img, self.to_screen(x, 0), self.to_screen(x, CANVAS_H), GRID, self.px(2))
        for y in REGION_Y[1:3]:
            cv2.line(img, self.to_screen(0, y), self.to_screen(CANVAS_W, y), GRID, self.px(2))
        if test.active():
            # Blue while the cursor is inside the circle (a target only once the cursor is visible)
            inside = test.t_enter is not None and test.cursor_visible(t)
            if test.trial["kind"] == "home":
                cv2.circle(img, self.to_screen(test.trial["x"], test.trial["y"]), self.px(HOME_RADIUS),
                           BLUE if inside else GREEN, self.px(3))
            else:
                cv2.circle(img, self.to_screen(test.trial["x"], test.trial["y"]), self.px(TARGET_RADIUS),
                           BLUE if inside else TARGET, self.px(3))
        if test.cursor_visible(t):
            cv2.circle(img, self.to_screen(crs_x, crs_y), self.px(CRS_RADIUS), CURSOR if hand_detected else GRID, -1)
        if test.paused():
            left = max(0.0, test.pause_end - t)
            text = f"Pause  {int(left // 60)}:{int(left % 60):02d}"
            size = 1.6 * self.scale
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 3)
            cv2.putText(img, text, ((self.w - tw) // 2, int(self.oy + 60 * self.scale) + th),
                        cv2.FONT_HERSHEY_SIMPLEX, size, WHITE, 3)
        elapsed = test.elapsed(t)   # stopped during the pause, until the home is entered again
        tr = test.trial or test.trials[min(max(test.trial_i, 0), len(test.trials) - 1)]
        n_part = sum(1 for g in test.trials if g["kind"] == "region" and g["part"] == tr["part"])
        number = min(tr["part_target_number"], n_part)
        info = (f"part {tr['part']}/{test.n_parts}   target {number}/{n_part}   "
                f"{int(elapsed // 60)}:{int(elapsed % 60):02d}")
        cv2.putText(img, info, (int(20 * self.scale), self.h - int(20 * self.scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.scale, WHITE, 2)
        cv2.imshow(self.window, img)


# --- Entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="S000",
                        help="Participant id: loads calibrations/<subject>.npz if it exists, else the latest "
                             "calibrations/<subject>_<date>_<time>.npz from customize_bomi.py; also names the "
                             "result files (default: S000)")
    parser.add_argument("--calib", default=None,
                        help="Map to load (calibrations/<NAME>.npz) instead of the participant's own")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH, help="MediaPipe hand_landmarker.task model")
    parser.add_argument("--sequence", default=SEQUENCE_FILE,
                        help="Region sequence file, one region (1-9, 5 = home) per row (default: config/cursor_regions.csv)")
    args = parser.parse_args()

    sequence = load_sequence(args.sequence)
    print(f"Sequence ({len(sequence)} goals, {sum(1 for r in sequence if r != HOME_REGION)} targets): "
          + " ".join(map(str, sequence)))

    subject = bomi._strip_npz(args.subject)
    calib_path = bomi._resolve_calib_path(args.calib) if args.calib else bomi._resolve_subject_map_path(args.subject)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No calibration map for '{args.subject}' found: run customize_bomi.py {subject} first.")
        available = bomi._list_saved_maps()
        if available:
            print("        Available: " + ", ".join(available))
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[ERROR] MediaPipe model not found: '{args.model}'")
        sys.exit(1)

    bomi_map = bomi.BoMIMap()
    bomi_map.load(calib_path)
    print(f"Loaded calibration from {calib_path}")
    bomi_map.print_metrics()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        sys.exit(1)
    landmarker = bomi.create_hand_landmarker(args.model)

    test = RegionsTest(subject, build_trials(sequence))
    bomi.open_camera_window(cap)   # before the test window, which keeps the keyboard focus
    screen = Screen()
    cursor_filter = bomi.CursorFilter()
    # Map space (BASE_WIDTH x BASE_HEIGHT) -> canvas
    sx, sy = CANVAS_W / bomi.BASE_WIDTH, CANVAS_H / bomi.BASE_HEIGHT
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0

    print(f"\n=== 9-REGION BLIND REACHING === {test.n_parts} parts x "
          f"{sum(1 for tr in test.trials if tr['kind'] == 'region' and tr['part'] == 1)} targets, "
          f"{PAUSE_S / 60:.0f} min pause in between. Q = abort")
    test.start(time.time())
    try:
        while not test.end_reason:
            frame, crs_x, crs_y, hand_detected = bomi.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
            bomi.show_camera(frame, hand_detected)
            t = time.time()
            cx = min(max(crs_x * sx, 0.0), CANVAS_W)
            cy = min(max(crs_y * sy, 0.0), CANVAS_H)
            test.update(t, cx, cy)
            screen.draw(test, cx, cy, t, hand_detected)
            key = cv2.waitKey(1) & 0xFF
            if bomi._quit_requested(key, screen.window):
                test.end_reason = "aborted"
    finally:
        summary = test.finish(time.time())
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    fmt = lambda v, u="s": f"{v:.2f}{u}" if v is not None else "-"
    print(f"\nSession over ({summary['end_reason']}): {summary['regions']['n_success']}/{summary['n_targets_total']} "
          f"targets reached, time {fmt(summary['time_total'])} (pause excluded; "
          f"{fmt(summary['wall_clock_duration'])} with it)")
    for p, part in summary["parts"].items():
        a, h = part["regions"], part["homes"]
        print(f"  part {p}: {a['n_success']}/{part['n_targets_total']} targets, time {fmt(summary[f'time_part_{p}'])}")
        print(f"    centre -> region: mean time {fmt(a['mean_reach_time'])}, norm. path {fmt(a['mean_normalized_path_length'], '')}, "
              f"reached hidden {fmt(a['mean_reached_hidden'], '%')}, correct at reveal {fmt(a['mean_region_at_reveal_correct'], '%')}, "
              f"first region correct {fmt(a['mean_first_region_correct'], '%')}, wrong regions {fmt(a['mean_n_wrong_regions'], '')}")
        print(f"    region -> centre: mean time {fmt(h['mean_reach_time'])}, norm. path {fmt(h['mean_normalized_path_length'], '')}")
    cmp = summary["part_2_vs_part_1"]
    if cmp:
        print("  part 2 - part 1:")
        for k in ("mean_reach_time", "mean_normalized_path_length", "mean_reached_hidden",
                  "mean_region_at_reveal_correct", "mean_first_region_correct", "mean_n_wrong_regions"):
            c = cmp[k]
            print(f"    {k[5:]:26s} {fmt(c['part_1'], ''):>8s} -> {fmt(c['part_2'], ''):>8s}   delta {fmt(c['delta'], ''):>8s}")
    print(f"  results: {test.base}_trials.csv / _blocks.csv / _trajectory.csv / _summary.json")


if __name__ == "__main__":
    main()
