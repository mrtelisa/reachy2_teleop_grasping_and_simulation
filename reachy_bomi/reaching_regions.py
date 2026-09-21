#!/usr/bin/env python3
"""
Fullscreen 9-region cursor test -- no robot, no socket, no velocities.

Same hand -> cursor chain, canvas (1200 x 650, scaled to the whole screen),
cursor and metrics as reaching_center_out.py, but the screen is
divided in the 9 regions of the teleoperation interface (3 x 3 grid) and the
targets are INVISIBLE: one square AREA per region (its coordinates are saved),
which the participant has to enter following the experimenter's verbal
instruction ("go to the top-left corner").

Regions / areas:
    1 | 2 | 3   (top row)
    4 | 5 | 6   (middle row)
    7 | 8 | 9   (bottom row)
Areas are squares of AREA_SIDE canvas px: the ones in the outer regions are
pushed towards the screen edge (AREA_EDGE_MARGIN px from it). Region 5 is the
HOME: a visible disc at the screen centre (radius HOME_RADIUS), the only
thing the participant sees besides the grid and the cursor.

Sequence: config/cursor_regions.csv, one region per row, e.g.
    5, 1, 5, 2, 5, 3 ...   = home, area 1, home, area 2, ...
(any other file with --sequence). It is generated once if missing: every
outer region N_REPETITIONS times in seeded random order, a home between them.

Trial flow (the participant is told "go to the top-left corner", then
"back to the centre"; nothing has to be pressed):
  home goal  -> the disc appears the moment the previous area is entered;
                reached when the cursor stays inside it HOME_DWELL_S (0.5 s).
                Its reach_time = disc shown -> entering the disc (return).
  area goal  -> the disc stays visible until the cursor LEAVES it: that is
                the trial start (the disc disappears). The trial ends the
                moment the cursor enters the area (no dwell), and the disc
                reappears for the return.
                Its reach_time = leaving the disc -> entering the area.
The trajectory of every goal is recorded from its start (leaving the disc /
disc shown) and the reaching_metrics.py kinematics (normalized_path_length,
dimensionless_jerk, n_speed_peaks, ...) are computed on it; the whole cursor
trajectory is saved too.

The session timer starts when the home is reached for the first time. The
session ends when every goal has been reached (or on Q/ESC); results
(a subject with previous sessions gets _1, _2, ... appended):
  results_regions/<subject>_regions_trials.csv       one row per goal (times, metrics)
  results_regions/<subject>_regions_trajectory.csv   every cursor sample (trial, t, x, y)
  results_regions/<subject>_regions_summary.json     means (areas, returns, per region), config

Usage:
    python3 reaching_regions.py --subject S001 [--calib <name>] [--cam 0] [--sequence file.csv]
        --show-areas      draw the areas (to check them: never with a participant)
        --preview out.png write the layout to an image and exit (no webcam)
Keys: Q / ESC = abort (results so far are still saved).
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

import reaching_center_out as base
import socket_client as bomi
from reaching_center_out import CANVAS_W, CANVAS_H, CRS_RADIUS, CURSOR, WHITE, GREEN, BLUE
from reaching_metrics import compute_trial_metrics, summarize

# --- Geometry (canvas px) ---
REGION_X = (0, CANVAS_W / 3.0, 2 * CANVAS_W / 3.0, CANVAS_W)   # column boundaries (400 px wide)
REGION_Y = (0, CANVAS_H / 3.0, 2 * CANVAS_H / 3.0, CANVAS_H)   # row boundaries (216.7 px high)
AREA_SIDE = 140          # square side: the participant cannot see it, so keep it large
AREA_EDGE_MARGIN = 50    # distance of the outer areas from the screen edge
HOME_REGION = 5
HOME_RADIUS = base.TGT_RADIUS   # visible disc at the centre (40 px, as the center-out home)
REGION_NAMES = {1: "top-left", 2: "top", 3: "top-right", 4: "left", 5: "centre (home)",
                6: "right", 7: "bottom-left", 8: "bottom", 9: "bottom-right"}

# --- Sequence ---
SEQUENCE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "cursor_regions.csv")
N_REPETITIONS = 3        # default sequence: every outer region this many times
SEQUENCE_SEED = base.TARGET_SEED

# --- Timing / metrics (as reaching_center_out) ---
HOME_DWELL_S = base.DWELL_S   # the cursor must stay on the home disc this long (0.5 s)
AREA_DWELL_S = 0.0            # an area counts as reached the moment the cursor enters it
MOTION_ONSET_SPEED = base.MOTION_ONSET_SPEED
SPEED_PEAK_THRESHOLD = base.SPEED_PEAK_THRESHOLD
RESAMPLE_HZ = base.RESAMPLE_HZ

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_regions")
WINDOW = "BoMI - Reaching (regions)"
GRID = (90, 90, 90)
AREA_TARGET = (0, 0, 255)   # --show-areas only: the area currently to reach


def region_of(x: float, y: float) -> int:
    """Region 1..9 of a canvas point (3 x 3 grid, row-major from the top-left)."""
    col = 0 if x < REGION_X[1] else 1 if x < REGION_X[2] else 2
    row = 0 if y < REGION_Y[1] else 1 if y < REGION_Y[2] else 2
    return row * 3 + col + 1


def build_areas(side: float = AREA_SIDE, margin: float = AREA_EDGE_MARGIN) -> dict:
    """The square area of every region: {region: dict(region, cx, cy, side,
    x0, y0, x1, y1)} in canvas px. Outer columns/rows are pushed `margin` px
    from the screen edge, the middle ones are centred on their region."""
    areas = {}
    for region in range(1, 10):
        row, col = divmod(region - 1, 3)
        x0, y0, x1, y1 = REGION_X[col], REGION_Y[row], REGION_X[col + 1], REGION_Y[row + 1]
        cx = margin + side / 2.0 if col == 0 else CANVAS_W - margin - side / 2.0 if col == 2 else (x0 + x1) / 2.0
        cy = margin + side / 2.0 if row == 0 else CANVAS_H - margin - side / 2.0 if row == 2 else (y0 + y1) / 2.0
        if not (x0 <= cx - side / 2 and cx + side / 2 <= x1 and y0 <= cy - side / 2 and cy + side / 2 <= y1):
            raise ValueError(f"area {region} does not fit in its region (side {side}, margin {margin})")
        areas[region] = {"region": region, "cx": cx, "cy": cy, "side": side,
                         "x0": cx - side / 2.0, "y0": cy - side / 2.0, "x1": cx + side / 2.0, "y1": cy + side / 2.0}
    return areas


def inside_area(area: dict, x: float, y: float) -> bool:
    return area["x0"] <= x <= area["x1"] and area["y0"] <= y <= area["y1"]


def inside_home(x: float, y: float) -> bool:
    return math.hypot(x - CANVAS_W / 2.0, y - CANVAS_H / 2.0) < HOME_RADIUS


def print_areas(areas: dict, margin: float) -> None:
    print(f"Areas: side {areas[1]['side']:.0f} px, edge margin {margin:.0f} px (canvas {CANVAS_W}x{CANVAS_H}), "
          f"home disc radius {HOME_RADIUS} px at the centre")
    for a in areas.values():
        print(f"  region {a['region']} ({REGION_NAMES[a['region']]:12s}): centre ({a['cx']:.0f}, {a['cy']:.0f})  "
              f"x {a['x0']:.0f}-{a['x1']:.0f}  y {a['y0']:.0f}-{a['y1']:.0f}")


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


def build_trials(sequence: list, areas: dict) -> list:
    """One goal per region of the sequence: kind ("home" for region 5, "area"
    otherwise), region, target_number (the area goals counted 1..n; a home
    takes the number of the area that follows it), x, y (goal centre) and,
    for areas, x0/y0/x1/y1."""
    trials, n_areas = [], 0
    for r in sequence:
        if r == HOME_REGION:
            trials.append({"kind": "home", "region": r, "target_number": n_areas + 1,
                           "x": CANVAS_W / 2.0, "y": CANVAS_H / 2.0})
        else:
            n_areas += 1
            a = areas[r]
            trials.append({"kind": "area", "region": r, "target_number": n_areas, "x": a["cx"], "y": a["cy"],
                           "x0": a["x0"], "y0": a["y0"], "x1": a["x1"], "y1": a["y1"]})
    return trials


class RegionsTest:
    def __init__(self, subject: str, trials: list, areas: dict, results_dir: str = RESULTS_DIR,
                 sequence: str = "regions") -> None:
        self.subject = subject
        self.trials = trials
        self.areas = areas
        self.sequence = sequence
        self.results = []
        self.trajectory = []   # every cursor sample: (trial, kind, region, t, x, y, phase)

        os.makedirs(results_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.base = os.path.join(results_dir, base.session_name(subject, sequence, results_dir))

        # Session/trial state (times are time.time())
        self.t_start = None
        self.t_session0 = None
        self.trial_i = -1
        self.trial = None
        self.t_shown = None   # trial start: disc shown (home) / cursor left the disc (area); None = area waiting
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
            self.end_reason = "completed"
            return
        self.trial = self.trials[self.trial_i]
        self.t_shown = t if self.trial["kind"] == "home" else None
        self.t_enter = None
        self.samples = []
        n = f"{self.trial_i + 1}/{len(self.trials)}"
        if self.trial["kind"] == "home":
            print(f"  goal {n}: HOME (disc shown)")
        else:
            print(f"  goal {n}: region {self.trial['region']} ({REGION_NAMES[self.trial['region']]})"
                  f"  -> give the instruction; the trial starts when the cursor leaves the disc")

    def waiting(self) -> bool:
        """An area goal is current but the cursor has not left the home disc yet."""
        return self.trial is not None and not self.end_reason and self.t_shown is None

    def home_visible(self) -> bool:
        """The centre disc is shown while it is the goal and while an area goal waits for the cursor to leave it."""
        return self.trial is not None and not self.end_reason and (self.trial["kind"] == "home" or self.waiting())

    def update(self, t: float, x: float, y: float) -> None:
        """One frame with the current (filtered) cursor position in canvas px."""
        if self.trial is None or self.end_reason:
            return
        if self.waiting():
            if inside_home(x, y):
                self.trajectory.append((self.trial_i + 1, self.trial["kind"], self.trial["region"], t, x, y, "wait"))
                return
            self.t_shown = t   # the cursor left the disc: the area trial starts (and the disc disappears)
        self.trajectory.append((self.trial_i + 1, self.trial["kind"], self.trial["region"], t, x, y, "go"))
        self.samples.append((t, x, y))

        inside = inside_home(x, y) if self.trial["kind"] == "home" else inside_area(self.trial, x, y)
        dwell = HOME_DWELL_S if self.trial["kind"] == "home" else AREA_DWELL_S
        if inside:
            if self.t_enter is None:
                self.t_enter = t
            if t - self.t_enter >= dwell:
                self._end_trial(t)
                self._next_trial(t)
        else:
            self.t_enter = None

    def t0(self) -> float:
        """Time origin for the logs: session start if it has begun, else the launch."""
        return self.t_session0 if self.t_session0 is not None else self.t_start

    def _end_trial(self, t: float) -> None:
        if self.t_session0 is None:
            self.t_session0 = t   # first reach of the home: the session (and its timer) starts now
            print("  home reached: session timer started")
        self._record_trial(t, success=True, reason="reached")
        r = self.results[-1]
        what = "area entered" if self.trial["kind"] == "area" else "back at the centre"
        npl = f", norm. path {r['normalized_path_length']:.2f}" if r["normalized_path_length"] is not None else ""
        print(f"    {what} in {r['reach_time']:.2f}s{npl}")

    def _record_trial(self, t: float, success: bool, reason: str) -> None:
        tr = self.trial
        started = self.t_shown is not None   # an area goal aborted while waiting has nothing to measure
        metrics = compute_trial_metrics(self.samples if started else [], (tr["x"], tr["y"]),
                                        self.t_shown if started else t, self.t_enter if success else None,
                                        MOTION_ONSET_SPEED, SPEED_PEAK_THRESHOLD, RESAMPLE_HZ)
        t_enter = self.t_enter if success else None
        rel = lambda ts: (ts - self.t0()) if ts is not None else None
        self.results.append({
            "trial": self.trial_i + 1, "kind": tr["kind"], "region": tr["region"],
            "target_number": tr["target_number"],
            "goal_x": tr["x"], "goal_y": tr["y"],
            "area_x0": tr.get("x0"), "area_y0": tr.get("y0"), "area_x1": tr.get("x1"), "area_y1": tr.get("y1"),
            "success": success, "end_reason": reason,
            # Session times (s from the session start): trial start (area: the
            # cursor left the disc; home: the disc appeared), goal entered, trial end
            "t_start": rel(self.t_shown),
            "t_enter": rel(t_enter),
            "t_end": rel(t),
            "trial_duration": (t - self.t_shown) if started else None,
            **metrics,   # reach_time = t_enter - t_start, path metrics from t_start
        })

    def finish(self, t: float, reason: str = None) -> dict:
        """Close the current (unfinished) goal as missed (only on an abort) and write the results."""
        reason = reason or self.end_reason or "aborted"
        if self.trial is not None and self.trial_i < len(self.trials) and reason != "completed":
            self._record_trial(t, success=False, reason=reason)
        areas = [r for r in self.results if r["kind"] == "area"]
        homes = [r for r in self.results if r["kind"] == "home"]
        summary = {
            "subject": self.subject, "sequence": self.sequence, "end_reason": reason,
            "n_trials_total": len(self.trials),
            "n_areas_total": sum(1 for tr in self.trials if tr["kind"] == "area"),
            "session_duration": (t - self.t_session0) if self.t_session0 is not None else 0.0,
            "timestamp": self.timestamp,
            "areas": summarize(areas),      # centre -> invisible area (what the test measures)
            "homes": summarize(homes),      # area -> back to the centre disc
            # Same statistics per region, over its repetitions
            "per_region": {str(reg): summarize([r for r in areas if r["region"] == reg])
                           for reg in sorted({r["region"] for r in areas})},
            "missed": [{"trial": r["trial"], "kind": r["kind"], "region": r["region"], "reason": r["end_reason"]}
                       for r in self.results if not r["success"]],
            "config": {
                "canvas": [CANVAS_W, CANVAS_H], "crs_radius": CRS_RADIUS,
                "area_side": self.areas[1]["side"], "home_radius": HOME_RADIUS,
                "areas": {str(k): {"cx": a["cx"], "cy": a["cy"], "x0": a["x0"], "y0": a["y0"], "x1": a["x1"], "y1": a["y1"]}
                          for k, a in self.areas.items()},
                "region_sequence": [tr["region"] for tr in self.trials],
                "home_dwell_s": HOME_DWELL_S, "area_dwell_s": AREA_DWELL_S,
                "motion_onset_speed": MOTION_ONSET_SPEED, "speed_peak_threshold": SPEED_PEAK_THRESHOLD,
            },
        }
        if self.results:
            with open(self.base + "_trials.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
                w.writeheader()
                w.writerows(self.results)
        if self.trajectory:
            with open(self.base + "_trajectory.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["trial", "kind", "region", "t", "x", "y", "phase"])
                w.writerows((i, k, reg, f"{ts - self.t0():.4f}", f"{x:.2f}", f"{y:.2f}", ph)
                            for i, k, reg, ts, x, y, ph in self.trajectory)
        with open(self.base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


# --- Drawing ---
class Screen(base.Screen):
    """reaching_center_out's fullscreen window (canvas scaled and centred);
    `size` renders offscreen instead (--preview)."""

    def __init__(self, title: str = WINDOW, size: tuple = None) -> None:
        if size is None:
            super().__init__(title)
            return
        self.window = title
        self.w, self.h = size
        self.scale = min(self.w / CANVAS_W, self.h / CANVAS_H)
        self.ox = (self.w - CANVAS_W * self.scale) / 2.0
        self.oy = (self.h - CANVAS_H * self.scale) / 2.0

    def render(self, crs_x: float, crs_y: float, hand_detected: bool, test: RegionsTest = None,
               areas: dict = None, info: str = "") -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        x0, y0 = self.to_screen(0, 0)
        x1, y1 = self.to_screen(CANVAS_W, CANVAS_H)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), GRID, 1)
        for x in REGION_X[1:3]:
            cv2.line(img, self.to_screen(x, 0), self.to_screen(x, CANVAS_H), GRID, self.px(2))
        for y in REGION_Y[1:3]:
            cv2.line(img, self.to_screen(0, y), self.to_screen(CANVAS_W, y), GRID, self.px(2))
        tr = test.trial if test is not None and not test.end_reason else None
        for a in (areas or {}).values():   # --show-areas only
            colour = AREA_TARGET if tr is not None and tr["kind"] == "area" and tr["region"] == a["region"] \
                else BLUE if inside_area(a, crs_x, crs_y) else GREEN
            cv2.rectangle(img, self.to_screen(a["x0"], a["y0"]), self.to_screen(a["x1"], a["y1"]), colour, self.px(2))
            cv2.putText(img, str(a["region"]), self.to_screen(a["x0"] + 8, a["y0"] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.scale, colour, 2)
        if test is not None and test.home_visible():
            colour = BLUE if tr["kind"] == "home" and test.t_enter is not None else GREEN
            cv2.circle(img, self.to_screen(CANVAS_W / 2.0, CANVAS_H / 2.0), self.px(HOME_RADIUS), colour, self.px(2))
        cv2.circle(img, self.to_screen(crs_x, crs_y), self.px(CRS_RADIUS), CURSOR if hand_detected else (90, 90, 90), -1)
        if info:
            cv2.putText(img, info, (int(20 * self.scale), self.h - int(20 * self.scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.scale, WHITE, 2)
        return img

    def draw(self, *args, **kwargs) -> None:
        cv2.imshow(self.window, self.render(*args, **kwargs))


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
    parser.add_argument("--show-areas", action="store_true",
                        help="Draw the areas (red = the one to reach): to check them, not for a participant")
    parser.add_argument("--side", type=float, default=AREA_SIDE, help=f"Area side in canvas px (default: {AREA_SIDE})")
    parser.add_argument("--margin", type=float, default=AREA_EDGE_MARGIN,
                        help=f"Distance of the outer areas from the screen edge (default: {AREA_EDGE_MARGIN})")
    parser.add_argument("--preview", default=None, metavar="PNG",
                        help="Write the layout (areas shown, cursor at the centre) to this image and exit: no webcam")
    args = parser.parse_args()

    areas = build_areas(args.side, args.margin)
    print_areas(areas, args.margin)
    sequence = load_sequence(args.sequence)
    print(f"Sequence ({len(sequence)} goals, {sum(1 for r in sequence if r != HOME_REGION)} areas): "
          + " ".join(map(str, sequence)))

    if args.preview:
        screen = Screen(size=(1920, 1080))
        cv2.imwrite(args.preview, screen.render(CANVAS_W / 2, CANVAS_H / 2, True, None, areas,
                                                f"side {args.side:.0f}  margin {args.margin:.0f}"))
        print(f"Layout written to {args.preview}")
        return

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
    landmarker = bomi.hand_landmarker.HandLandmarker.create_from_options(
        bomi.hand_landmarker.HandLandmarkerOptions(
            base_options=bomi.base_options.BaseOptions(model_asset_path=args.model),
            running_mode=bomi.vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    test = RegionsTest(subject, build_trials(sequence, areas), areas)
    screen = Screen()
    cursor_filter = bomi.CursorFilter()
    # Map space (BASE_WIDTH x BASE_HEIGHT) -> canvas, same as reaching_center_out
    sx, sy = CANVAS_W / bomi.BASE_WIDTH, CANVAS_H / bomi.BASE_HEIGHT
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0

    print(f"\n=== 9-REGION REACHING === {len(test.trials)} goals. Q = abort")
    test.start(time.time())
    try:
        while not test.end_reason:
            frame, crs_x, crs_y, hand_detected = bomi.update_bomi_cursor(cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y)
            t = time.time()
            cx = min(max(crs_x * sx, 0.0), CANVAS_W)
            cy = min(max(crs_y * sy, 0.0), CANVAS_H)
            test.update(t, cx, cy)
            elapsed = (t - test.t_session0) if test.t_session0 is not None else 0.0
            info = f"{min(test.trial_i + 1, len(test.trials))}/{len(test.trials)}   {int(elapsed // 60)}:{int(elapsed % 60):02d}"
            if test.trial is not None and not test.end_reason:   # the goal to reach
                goal = "CENTRE" if test.trial["kind"] == "home" else f"region {test.trial['region']} ({REGION_NAMES[test.trial['region']]})"
                info += f"   ->  {goal}"
            if args.show_areas:
                info += f"   region {region_of(cx, cy)}  cursor ({cx:.0f}, {cy:.0f})"
            screen.draw(cx, cy, hand_detected, test, areas if args.show_areas else None, info)

            key = cv2.waitKey(1) & 0xFF
            if bomi._quit_requested(key, screen.window):
                test.end_reason = "aborted"
    finally:
        summary = test.finish(time.time())
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    a, h = summary["areas"], summary["homes"]
    fmt = lambda v, u="s": f"{v:.2f}{u}" if v is not None else "-"
    print(f"\nSession over ({summary['end_reason']}): {a['n_success']}/{summary['n_areas_total']} areas reached, "
          f"{summary['session_duration']:.0f}s")
    print(f"  centre -> area: mean time {fmt(a['mean_reach_time'])}, norm. path {fmt(a['mean_normalized_path_length'], '')}")
    print(f"  area -> centre: mean time {fmt(h['mean_reach_time'])}, norm. path {fmt(h['mean_normalized_path_length'], '')}")
    print(f"  results: {test.base}_trials.csv / _trajectory.csv / _summary.json")


if __name__ == "__main__":
    main()
