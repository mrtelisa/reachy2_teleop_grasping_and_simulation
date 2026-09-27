#!/usr/bin/env python3
"""
Fullscreen 4-target blind reaching test, repeated N_TESTS (5) times per participant.

The hand drives a cursor through the BoMI chain of bomi.py (webcam ->
MediaPipe -> autoencoder map -> Butterworth filter) on the same 1200 x 650
canvas as reaching_regions.py, scaled to the whole screen. During the test
the cursor is NEVER drawn: the participant only sees the current target, an
empty yellow circle shown on its own, exactly as the target of the same region
in reaching_regions.py (same position and radius; the other targets are not
on screen). Every TRIAL_S (4 s)
it is replaced by the next target, whatever the cursor did. The cursor is
tracked the whole time, so the results say whether (and how well) the
participant got there from the learned map alone.

Targets: the training targets of TARGET_REGIONS (2, 4, 5, 9: top, left,
centre/home, bottom-right), each numbered as its region. Sequence:
config/blind_targets.csv, one trial per row (trial, target, x, y), any other
file with --sequence. It is generated once if missing: N_REPETITIONS (3)
visits of each target -> 12 trials, in seeded random order (all 4 once before
any repeat, never the same twice in a row). The file is frozen, so the 5 tests (and every
participant) see exactly the same targets.

Session flow: before the start only the cursor is shown (the only time it is
visible, no target), so the participant can see where the hand is.
After START_CURSOR_S (2 s) ENTER (experimenter) is accepted: it starts the
session, the cursor disappears and the first target appears at once;
after 12 x 4 s the session ends (or on Q/ESC). A small dot in the top-right
corner is green while the hand is tracked, grey when it is lost (no position
information). The webcam with the tracked hand is shown in a separate window
on the experimenter's monitor (never to the participant).

Metrics of every trial (window = the TRIAL_S the target is yellow; distances
in canvas px from the target centre):
  hit                 100 if the cursor stayed inside the target circle for DWELL_S
                      (0.5 s, as in reaching_regions.py) without leaving it, else 0:
                      crossing the circle by chance is not a hit
  reach_time          target shown -> the entry that completed that stay (hits only)
  time_in_target      % of the window the cursor was inside the circle
  on_target_at_end    100 if the cursor is inside the circle when the target changes
  initial_error       distance when the target is shown (start of the reach)
  final_error         distance when the target changes (endpoint error)
  end_error           mean distance over the last END_WINDOW_S (1 s)
  min_error           closest approach
  relative_final_error  final_error / initial_error (0 = perfect, 1 = did not get closer)
  chosen_target, chosen_correct   target of the 4 closest to the cursor when the
                      target changes (all 4 count, even if only the current
                      one is on screen), and 100 if it is the current one
  hand_lost           % of the window without a tracked hand
  region_at_end, region_at_end_correct   region (1..9 of the 3 x 3 grid) of the
                      cursor when the target changes, and 100 if it is the
                      target's region: the command the robot would have got
  end_time_in_region  % of the last END_WINDOW_S spent in the target's region
  time_in_region      % of the window spent in the target's region
  region_hit, region_reach_time   100 if the cursor stayed in the target's
                      region for DWELL_S, and the time to the entry of that stay
and the reaching_metrics.py kinematics (reaction_time, normalized_path_length,
initial_direction_error, dimensionless_jerk, n_speed_peaks, ...) from the start of the window to the
entry of the hit (hits) or to its end (misses). "success" in the files means the
trial ran its full TRIAL_S (False only for the trial cut by an abort): the
means are over the completed trials.

Results (--test 1..5; a subject who repeats the same test number gets _1,
_2, ... appended):
  results_blind/<subject>_blind_test<N>_trials.csv       one row per trial
  results_blind/<subject>_blind_test<N>_blocks.csv       one row per block of 4 trials (each target once)
  results_blind/<subject>_blind_test<N>_trajectory.csv   every cursor sample (trial, target, t, x, y, hand_detected)
  results_blind/<subject>_blind_test<N>_summary.json     means (all, per target, per block), config
Test N is compared with the subject's latest session of every earlier test
(1..N-1): the main metrics of each and the differences (test N - test k) are
printed and stored in the summary (comparison_with_previous).

Usage:
    python3 reaching_blind.py --subject S001 --test 1..5 [--calib <name>] [--cam 0] [--sequence file.csv]
Keys: ENTER = start (after the first 2 s), Q / ESC = abort (results so far are still saved).
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time

import cv2
import numpy as np

import bomi
from reaching_metrics import METRIC_KEYS, block_summaries, compute_trial_metrics, summarize
import reaching_regions
from reaching_regions import CANVAS_H, CANVAS_W, DWELL_S, latest_summary, region_of, session_name

# --- Targets: the reaching_regions.py targets of these regions (same canvas, position and radius) ---
TARGET_REGIONS = (2, 4, 5, 9)
TARGET_RADIUS = reaching_regions.TARGET_RADIUS

# --- Sequence ---
SEQUENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "blind_targets.csv")
N_TARGETS = len(TARGET_REGIONS)
N_REPETITIONS = 3        # default sequence: every target this many times (4 x 3 = 12 trials)
SEQUENCE_SEED = 20
BLOCK_SIZE = N_TARGETS   # statistics per block of 4 consecutive trials (= every target once)
N_TESTS = 5              # blind tests per participant (--test 1..N_TESTS)

# --- Timing ---
START_CURSOR_S = 2.0     # the cursor is shown at least this long before ENTER can start the session
TRIAL_S = 4.0            # the current target changes every TRIAL_S
END_WINDOW_S = 1.0       # end_error = mean distance over the last END_WINDOW_S of the trial

# --- Metrics (pixels of the canvas) ---
MOTION_ONSET_SPEED = 40.0     # [px/s]
SPEED_PEAK_THRESHOLD = 80.0   # [px/s]
RESAMPLE_HZ = 50.0
BLIND_KEYS = ("hit", "time_in_target", "on_target_at_end", "initial_error", "final_error", "end_error",
              "min_error", "relative_final_error", "chosen_target", "chosen_correct", "hand_lost",
              "region_at_end", "region_at_end_correct", "end_time_in_region", "time_in_region",
              "region_hit", "region_reach_time")
# Averaged in the summary (target and region numbers are not)
TRIAL_KEYS = METRIC_KEYS + tuple(k for k in BLIND_KEYS if k not in ("chosen_target", "region_at_end"))
# Printed at the end and compared with the earlier tests: the main metrics of the analysis first
MAIN_KEYS = ("hit", "end_error", "reach_time", "initial_direction_error", "region_at_end_correct",
             "chosen_correct", "final_error", "relative_final_error", "time_in_target", "normalized_path_length")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_blind")
WINDOW = "BoMI - Blind reaching"

# Colours (BGR)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
GRID = (90, 90, 90)
TARGET = (0, 255, 255)   # current target (empty circle, as in reaching_regions.py)


def target_position(region: int) -> tuple:
    """(x, y) of the reaching_regions.py target of a region (the home for region 5)."""
    x0, y0, x1, y1 = reaching_regions.region_bounds(region)
    return round((x0 + x1) / 2.0), round((y0 + y1) / 2.0)


def generate_sequence(repetitions: int = N_REPETITIONS, seed: int = SEQUENCE_SEED) -> list:
    """TARGET_REGIONS `repetitions` times in seeded random order (all of them
    once before any repeat, never the same twice in a row)."""
    rng = np.random.default_rng(seed + 1)
    order, last = [], None
    for _ in range(repetitions):
        perm = [TARGET_REGIONS[i] for i in rng.permutation(N_TARGETS)]
        if perm[0] == last:
            perm.append(perm.pop(0))
        order += perm
        last = order[-1]
    return order


def load_trials(path: str = SEQUENCE_FILE) -> list:
    """Trials of `path` (header trial,target,x,y; one row per trial; target =
    its region) as dicts target, x, y. The default file is generated once if missing."""
    if path == SEQUENCE_FILE and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["trial", "target", "x", "y"])
            for i, tgt in enumerate(generate_sequence(), start=1):
                w.writerow([i, tgt, *target_position(tgt)])
        print(f"[sequence] generated the default targets -> {path} (commit this file)")
    trials, positions = [], {}
    with open(path, encoding="utf-8") as f:
        for n, row in enumerate(csv.DictReader(f), start=2):
            tgt, x, y = int(row["target"]), float(row["x"]), float(row["y"])
            if positions.setdefault(tgt, (x, y)) != (x, y):
                raise ValueError(f"{path} line {n}: target {tgt} moved from {positions[tgt]} to {(x, y)}")
            if trials and trials[-1]["target"] == tgt:
                print(f"[WARNING] {path} line {n}: target {tgt} twice in a row")
            trials.append({"target": tgt, "x": x, "y": y})
    if not trials:
        raise ValueError(f"{path}: no trials found")
    return trials


def target_positions(trials: list) -> dict:
    """{target: (x, y)} of every target of the session."""
    return {tr["target"]: (tr["x"], tr["y"]) for tr in trials}


def blind_metrics(samples: list, hand: list, goal: tuple, positions: dict, target: int, t_shown: float,
                  t_enter: float) -> dict:
    """End-point / accuracy metrics of one trial (see the module docstring).
    samples: (t, x, y) of the whole window, hand: hand_detected per sample,
    t_enter: entry of the first DWELL_S stay in the circle (None = no hit)."""
    m = {k: None for k in BLIND_KEYS}
    if not samples:
        return m
    arr = np.asarray(samples, dtype=float)
    t = arr[:, 0]
    d = np.hypot(arr[:, 1] - goal[0], arr[:, 2] - goal[1])
    inside = d < TARGET_RADIUS
    m["hit"] = 100.0 if t_enter is not None else 0.0
    # Time-weighted fractions: each sample holds until the next one
    dt = np.diff(t, append=t[-1])
    total = float(np.sum(dt))
    if total > 0:
        m["time_in_target"] = 100.0 * float(np.sum(dt[inside])) / total
        m["hand_lost"] = 100.0 * float(np.sum(dt[~np.asarray(hand, dtype=bool)])) / total
    m["on_target_at_end"] = 100.0 if inside[-1] else 0.0
    m["initial_error"] = float(d[0])
    m["final_error"] = float(d[-1])
    m["end_error"] = float(np.mean(d[t >= t[-1] - END_WINDOW_S]))
    m["min_error"] = float(np.min(d))
    if d[0] > 1e-6:
        m["relative_final_error"] = float(d[-1] / d[0])
    x, y = arr[-1, 1], arr[-1, 2]
    m["chosen_target"] = min(positions, key=lambda k: math.hypot(x - positions[k][0], y - positions[k][1]))
    m["chosen_correct"] = 100.0 if m["chosen_target"] == target else 0.0
    # Region-level accuracy: the 3 x 3 region is the command the robot would get
    in_region = np.array([region_of(px, py) == target for px, py in arr[:, 1:3]])
    m["region_at_end"] = region_of(x, y)
    m["region_at_end_correct"] = 100.0 if m["region_at_end"] == target else 0.0
    if total > 0:
        m["time_in_region"] = 100.0 * float(np.sum(dt[in_region])) / total
    end = t >= t[-1] - END_WINDOW_S
    m["end_time_in_region"] = 100.0 * float(np.sum(dt[end & in_region])) / max(float(np.sum(dt[end])), 1e-9)
    t_stay = None
    for ti, ok in zip(t, in_region):
        if not ok:
            t_stay = None
        elif t_stay is None:
            t_stay = ti
        if t_stay is not None and ti - t_stay >= DWELL_S:
            m["region_reach_time"] = float(t_stay - t_shown)
            break
    m["region_hit"] = 100.0 if m["region_reach_time"] is not None else 0.0
    return m


def compare(earlier: dict, current: dict) -> dict:
    """{metric: {earlier, current, delta}} of the MAIN_KEYS means (delta = current - earlier)."""
    out = {}
    for k in MAIN_KEYS:
        a, b = earlier["all"].get(f"mean_{k}"), current["all"].get(f"mean_{k}")
        out[k] = {"earlier": a, "current": b, "delta": (b - a) if a is not None and b is not None else None}
    return out


class BlindTest:
    def __init__(self, subject: str, test_number: int, trials: list, results_dir: str = RESULTS_DIR) -> None:
        self.subject = subject
        self.test_number = test_number
        self.trials = trials
        self.positions = target_positions(trials)
        self.results = []
        self.trajectory = []   # every cursor sample: (trial, target, t, x, y, hand_detected)

        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.base = os.path.join(results_dir, session_name(subject, f"blind_test{test_number}", results_dir))

        # Session/trial state (times are time.time())
        self.t_start = None
        self.trial_i = -1
        self.trial = None
        self.t_shown = None
        self.t_inside = None   # entry of the current stay in the target circle
        self.t_enter = None    # entry of the first stay of DWELL_S (the hit)
        self.samples = []
        self.hand = []
        self.end_reason = None

    # --- trial flow ---
    def start(self, t: float) -> None:
        self.t_start = t
        self._next_trial(t)

    def _next_trial(self, t: float) -> None:
        self.trial_i += 1
        if self.trial_i >= len(self.trials):
            self.trial = None
            self.end_reason = "completed"
            return
        self.trial = self.trials[self.trial_i]
        self.t_shown = t
        self.t_inside = None
        self.t_enter = None
        self.samples = []
        self.hand = []

    def active(self) -> bool:
        return self.trial is not None and not self.end_reason

    def update(self, t: float, x: float, y: float, hand_detected: bool) -> None:
        """One frame with the current (filtered) cursor position in canvas px."""
        if not self.active():
            return
        if t - self.t_shown >= TRIAL_S:
            t_change = self.t_shown + TRIAL_S
            self._end_trial(t_change)
            self._next_trial(t_change)
            if not self.active():
                return
        self.trajectory.append((self.trial_i + 1, self.trial["target"], t, x, y, int(hand_detected)))
        self.samples.append((t, x, y))
        self.hand.append(hand_detected)
        if math.hypot(x - self.trial["x"], y - self.trial["y"]) >= TARGET_RADIUS:
            self.t_inside = None   # left the circle: the stay restarts
        elif self.t_inside is None:
            self.t_inside = t
        if self.t_enter is None and self.t_inside is not None and t - self.t_inside >= DWELL_S:
            self.t_enter = self.t_inside

    def _end_trial(self, t: float) -> None:
        self._record_trial(t, success=True, reason="completed")
        r = self.results[-1]
        hit = f"hit in {r['reach_time']:.2f}s" if r["hit"] else "missed"
        print(f"  trial {r['trial']}: target {r['target']} {hit}, final error {r['final_error']:.0f}px, "
              f"closest target {r['chosen_target']}")

    def _record_trial(self, t: float, success: bool, reason: str) -> None:
        tr = self.trial
        goal = (tr["x"], tr["y"])
        metrics = compute_trial_metrics(self.samples, goal, self.t_shown, self.t_enter,
                                        MOTION_ONSET_SPEED, SPEED_PEAK_THRESHOLD, RESAMPLE_HZ)
        blind = blind_metrics(self.samples, self.hand, goal, self.positions, tr["target"], self.t_shown,
                              self.t_enter)
        rel = lambda ts: (ts - self.t_start) if ts is not None else None
        self.results.append({
            "trial": self.trial_i + 1, "target": tr["target"], "target_number": self.trial_i + 1,
            "goal_x": tr["x"], "goal_y": tr["y"],
            "success": success, "end_reason": reason,
            # Session times (s from ENTER): target shown, entry of the hit, target changed
            "t_shown": rel(self.t_shown),
            "t_enter": rel(self.t_enter),
            "t_end": rel(t),
            "trial_duration": t - self.t_shown,
            **metrics,   # reach_time = entry of the hit - t_shown
            **blind,
        })

    def finish(self, t: float, reason: str = None) -> dict:
        """Close the current (unfinished) trial (only on an abort) and write the results."""
        reason = reason or self.end_reason or "aborted"
        if self.active():
            self._record_trial(t, success=False, reason=reason)
        blocks = block_summaries(self.results, BLOCK_SIZE, keys=TRIAL_KEYS)
        summary = {
            "subject": self.subject, "test": self.test_number, "end_reason": reason,
            "n_trials_total": len(self.trials),
            "session_duration": (t - self.t_start) if self.t_start is not None else 0.0,
            "timestamp": self.timestamp,
            "all": summarize(self.results, keys=TRIAL_KEYS),
            # Same statistics per target, over its repetitions
            "per_target": {str(k): summarize([r for r in self.results if r["target"] == k], keys=TRIAL_KEYS)
                           for k in sorted(self.positions)},
            # ...and per block of BLOCK_SIZE consecutive trials (every target once)
            "block_size": BLOCK_SIZE,
            "n_blocks": len(blocks),
            "blocks": blocks,
            "config": {
                "canvas": [CANVAS_W, CANVAS_H], "target_radius": TARGET_RADIUS,
                "targets": {str(k): list(v) for k, v in sorted(self.positions.items())},
                "target_sequence": [tr["target"] for tr in self.trials],
                "trial_s": TRIAL_S, "end_window_s": END_WINDOW_S, "dwell_s": DWELL_S,
                "motion_onset_speed": MOTION_ONSET_SPEED, "speed_peak_threshold": SPEED_PEAK_THRESHOLD,
            },
        }
        # Test N vs the latest session of every earlier test 1..N-1
        previous = {}
        for k in range(1, self.test_number):
            path = latest_summary(f"{self.subject}_blind_test{k}", self.results_dir)
            if path:
                with open(path, encoding="utf-8") as f:
                    previous[str(k)] = {"summary": os.path.basename(path), **compare(json.load(f), summary)}
        summary["comparison_with_previous"] = previous
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
                w.writerow(["trial", "target", "t", "x", "y", "hand_detected"])
                w.writerows((i, k, f"{ts - self.t_start:.4f}", f"{x:.2f}", f"{y:.2f}", h)
                            for i, k, ts, x, y, h in self.trajectory)
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

    def draw(self, test: BlindTest, hand_detected: bool, message: str = "", cursor: tuple = None) -> None:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        x0, y0 = self.to_screen(0, 0)
        x1, y1 = self.to_screen(CANVAS_W, CANVAS_H)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), GRID, 1)
        if test.active():   # only the current target, empty (as in reaching_regions.py)
            cv2.circle(img, self.to_screen(test.trial["x"], test.trial["y"]), self.px(TARGET_RADIUS),
                       TARGET, self.px(3))
        if cursor is not None:   # before the start, or always with --show-cursor (debug)
            cv2.circle(img, self.to_screen(*cursor), self.px(15), WHITE, -1)
        # Hand-tracking indicator (no position information)
        cv2.circle(img, (self.w - self.px(25), self.px(25)), self.px(8), GREEN if hand_detected else GRID, -1)
        n = len(test.trials)
        info = message or f"target {min(test.trial_i + 1, n)}/{n}"
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
    parser.add_argument("--test", required=True, type=int, choices=range(1, N_TESTS + 1), metavar=f"1..{N_TESTS}",
                        help="Number of the blind test (compared with the subject's earlier tests)")
    parser.add_argument("--calib", default=None,
                        help="Map to load (calibrations/<NAME>.npz) instead of the participant's own")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH, help="MediaPipe hand_landmarker.task model")
    parser.add_argument("--sequence", default=SEQUENCE_FILE,
                        help="Target file, one trial per row: trial,target,x,y (default: config/blind_targets.csv)")
    parser.add_argument("--show-cursor", action="store_true",
                        help="Draw the cursor (to test the setup only, never with a participant)")
    args = parser.parse_args()

    trials = load_trials(args.sequence)
    print("Targets: " + ", ".join(f"{k}=({x:.0f},{y:.0f})" for k, (x, y) in sorted(target_positions(trials).items())))
    print(f"Sequence ({len(trials)} trials x {TRIAL_S:.0f}s): " + " ".join(str(tr["target"]) for tr in trials))

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

    test = BlindTest(subject, args.test, trials)
    bomi.open_camera_window(cap)   # before the test window, which keeps the keyboard focus
    screen = Screen()
    cursor_filter = bomi.CursorFilter()
    # Map space (BASE_WIDTH x BASE_HEIGHT) -> canvas
    sx, sy = CANVAS_W / bomi.BASE_WIDTH, CANVAS_H / bomi.BASE_HEIGHT
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0

    print(f"\n=== BLIND REACHING (test {args.test}/{N_TESTS}) === {len(trials)} targets. ENTER = start, Q = abort")
    t_launch = time.time()
    try:
        while not test.end_reason:
            frame, crs_x, crs_y, hand_detected = bomi.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
            bomi.show_camera(frame, hand_detected)
            t = time.time()
            cx = min(max(crs_x * sx, 0.0), CANVAS_W)
            cy = min(max(crs_y * sy, 0.0), CANVAS_H)
            test.update(t, cx, cy, hand_detected)
            waiting = test.t_start is None
            can_start = waiting and t - t_launch >= START_CURSOR_S
            message = "ENTER to start" if can_start else ""
            # The cursor is visible only before the start (or with --show-cursor)
            screen.draw(test, hand_detected, message, (cx, cy) if waiting or args.show_cursor else None)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10) and can_start:
                test.start(time.time())
                print("  started")
            elif bomi._quit_requested(key, screen.window):
                test.end_reason = "aborted"
    finally:
        summary = test.finish(time.time())
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    a = summary["all"]
    fmt = lambda v, u="": f"{v:.2f}{u}" if v is not None else "-"
    print(f"\nSession over ({summary['end_reason']}): {a['n_success']}/{summary['n_trials_total']} trials, "
          f"hit {fmt(a['mean_hit'], '%')}, closest target correct {fmt(a['mean_chosen_correct'], '%')}, "
          f"final error {fmt(a['mean_final_error'], 'px')}, relative final error {fmt(a['mean_relative_final_error'])}, "
          f"time in target {fmt(a['mean_time_in_target'], '%')}")
    previous = summary["comparison_with_previous"]
    if previous:
        print(f"\n  test {args.test} vs earlier tests (" + ", ".join(f"{k}: {c['summary']}" for k, c in previous.items())
              + "); delta = this test - that test:")
        print(f"    {'':24s}" + "".join(f"{'test ' + k:>9s}" for k in previous) + f"{'test ' + str(args.test):>9s}"
              + "".join(f"{'d vs ' + k:>9s}" for k in previous))
        for m in MAIN_KEYS:
            print(f"    {m:24s}" + "".join(f"{fmt(c[m]['earlier']):>9s}" for c in previous.values())
                  + f"{fmt(a[f'mean_{m}']):>9s}" + "".join(f"{fmt(c[m]['delta']):>9s}" for c in previous.values()))
    elif args.test > 1:
        print(f"\n  no earlier test of '{subject}' in {RESULTS_DIR}: nothing to compare")
    print(f"  results: {test.base}_trials.csv / _blocks.csv / _trajectory.csv / _summary.json")


if __name__ == "__main__":
    main()
