"""
Session metrics of a reachy_control.py run, written to
results_robot/<subject>_run<N>_session.json (N = --run, 1 or 2; a repeated run
number gets _1, _2, ... appended).
Test = from Control start
(after the cursor preview) to the placement of the carried object. Positions
come from the mobile base odometry.

  test_duration, navigation_duration   navigation = up to the first object selection
  n_repositioning                      repositioning navigations used
  objects_moved                        objects picked and placed
  path_length_max_speed / _reduced_speed / _repositioning / _transport / _navigation / _total  [m]
  optimal_path_length, normalized_path_length   navigation path / DEFAULT_OPTIMAL_PATH_LENGTH
  log_dimensionless_jerk               navigation smoothness (Hogan & Sternad 2009)
  region_time_percent                  driving time per region, dwells removed from region 5
  n_dwell, n_dwell_declined            dwells while driving, and those answered "No"

Every odometry sample taken while driving (control-loop rate, PUBLISH_HZ =
20 Hz) is also written as is to <subject>_run<N>_odometry.csv next to the json:
t (unix), t_test (s since test start), x, y [m], theta [rad], vx, vy [m/s],
vtheta [rad/s], mode (max_speed / reduced_speed / repositioning / transport).
"""

import csv

import datetime
import json
import math
import os
import re
import time

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_robot")

# Optimal navigation path [m] for normalized_path_length: set before a session (None = skip)
DEFAULT_OPTIMAL_PATH_LENGTH = None

MODE_MAX = "max_speed"
MODE_REDUCED = "reduced_speed"
MODE_REPOSITIONING = "repositioning"
MODE_TRANSPORT = "transport"   # driving with the grasped object in hand, towards the placement table
MODES = (MODE_MAX, MODE_REDUCED, MODE_REPOSITIONING, MODE_TRANSPORT)

JERK_RESAMPLE_HZ = 20.0   # odometry is sampled at the control loop rate (PUBLISH_HZ)
REGIONS = tuple(range(1, 10))
# Longer gaps between region ticks = control loop not running (preview, dialog): not counted
REGION_TICK_MAX_GAP_S = 0.5


def log_dimensionless_jerk(samples) -> float:
    """samples: list of (t, x, y). Returns -ln(dimensionless jerk) of the
    trajectory, or None if it is too short / the base did not move."""
    if len(samples) < 5:
        return None
    arr = np.asarray(samples, dtype=float)
    t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(np.diff(t) <= 0):
        keep = np.concatenate([[True], np.diff(t) > 0])
        t, x, y = t[keep], x[keep], y[keep]
        if t.size < 5:
            return None
    dt = 1.0 / JERK_RESAMPLE_HZ
    tu = np.arange(t[0], t[-1], dt)
    if tu.size < 5:
        return None
    xu, yu = np.interp(tu, t, x), np.interp(tu, t, y)
    vx, vy = np.gradient(xu, dt), np.gradient(yu, dt)
    ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
    jx, jy = np.gradient(ax, dt), np.gradient(ay, dt)
    duration = tu[-1] - tu[0]
    amplitude = math.hypot(xu[-1] - xu[0], yu[-1] - yu[0])
    if duration <= 0 or amplitude < 1e-3:
        return None
    jerk_int = float(np.sum(jx ** 2 + jy ** 2) * dt)
    dj = math.sqrt(0.5 * jerk_int * duration ** 5 / amplitude ** 2)
    return -math.log(dj) if dj > 0 else None


def _session_name(subject: str, run: int, results_dir: str) -> str:
    """<subject>_run<run> for the first session of that run, then
    <subject>_run<run>_1, _2, ... if it is repeated."""
    prefix = f"{subject}_run{run}"
    if not os.path.exists(os.path.join(results_dir, prefix + "_session.json")):
        return prefix
    used = {int(m.group(1)) for f in os.listdir(results_dir)
            for m in [re.match(rf"{re.escape(prefix)}_(\d+)_session\.json$", f)] if m}
    return f"{prefix}_{max(used, default=0) + 1}"


class SessionMetrics:
    def __init__(self, subject: str, run: int, optimal_path_m: float = DEFAULT_OPTIMAL_PATH_LENGTH,
                 dwell_seconds: float = 0.0, results_dir: str = RESULTS_DIR) -> None:
        self.subject = subject
        self.run = run
        self.optimal_path_m = optimal_path_m
        self.dwell_seconds = dwell_seconds   # default duration of a dwell (dwell() can override it)
        os.makedirs(results_dir, exist_ok=True)
        base = os.path.join(results_dir, _session_name(subject, run, results_dir))
        self.path_json = base + "_session.json"
        self.path_odometry_csv = base + "_odometry.csv"
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.t_test_start = None
        self.t_test_end = None
        self.end_reason = None
        self.t_first_selection = None
        self.n_repositioning = 0
        self.objects_moved = []
        self.path = {m: 0.0 for m in MODES}
        self.n_samples = {m: 0 for m in MODES}
        self._last_xy = None
        self._nav_samples = []   # (t, x, y) from test start to the first object selection
        self._odometry_log = []  # every sample, raw, for the csv
        self.region_time = {r: 0.0 for r in REGIONS}
        self._last_region_tick = None
        self.n_dwell = 0
        self.n_dwell_declined = 0
        self.dwell_time = 0.0   # [s], sum of the durations of the completed dwells (removed from region 5)
        self.saved = False

    # --- events ---
    def start_test(self) -> None:
        if self.t_test_start is None:
            self.t_test_start = time.time()
            print(f"[metrics] test started ({self.subject})")

    def enter_object_selection(self) -> None:
        if self.t_first_selection is None and self.t_test_start is not None:
            self.t_first_selection = time.time()
            print(f"[metrics] navigation over after {self.t_first_selection - self.t_test_start:.1f}s, "
                  f"{self.path_length_navigation:.2f} m")

    def repositioning(self) -> None:
        self.n_repositioning += 1

    def object_moved(self, name: str) -> None:
        self.objects_moved.append(name)

    def dwell(self, accepted, seconds: float = None) -> None:
        """A dwell in region 5 completed while driving. accepted: True = it
        changed state (pre-grasp pose / object selection / back to selection),
        False = the user answered No and kept driving, None = quit.
        seconds: how long that dwell was (default: dwell_seconds)."""
        self.n_dwell += 1
        self.dwell_time += self.dwell_seconds if seconds is None else seconds
        if accepted is False:
            self.n_dwell_declined += 1

    def end_test(self, reason: str) -> None:
        if self.t_test_end is None:
            self.t_test_end = time.time()
            self.end_reason = reason

    # --- odometry ---
    def sample(self, odom: dict, mode: str) -> None:
        """odom: mobile_base.get_current_odometry() dict (x, y in metres);
        mode: MODE_MAX / MODE_REDUCED / MODE_REPOSITIONING / MODE_TRANSPORT, the driving mode the
        base is in right now. Call it at the control-loop rate while driving."""
        if self.t_test_start is None or self.t_test_end is not None:
            return
        x, y = float(odom["x"]), float(odom["y"])
        t = time.time()
        if self._last_xy is not None:
            self.path[mode] += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        self.n_samples[mode] += 1
        if self.t_first_selection is None:
            self._nav_samples.append((t, x, y))
        self._odometry_log.append((t, t - self.t_test_start, x, y, odom.get("theta"),
                                   odom.get("vx"), odom.get("vy"), odom.get("vtheta"), mode))

    def region_tick(self, region: int, now: float = None) -> None:
        """Called every control-loop iteration with the cursor's current
        region: accumulates the time since the previous tick on that region.
        Gaps longer than REGION_TICK_MAX_GAP_S (previews, dialogs) are skipped."""
        if self.t_test_start is None or self.t_test_end is not None:
            return
        now = time.time() if now is None else now
        if self._last_region_tick is not None:
            gap = now - self._last_region_tick
            if 0.0 < gap <= REGION_TICK_MAX_GAP_S and region in self.region_time:
                self.region_time[region] += gap
        self._last_region_tick = now

    # --- results ---
    @property
    def path_length_navigation(self) -> float:
        return self.path[MODE_MAX] + self.path[MODE_REDUCED]

    def region_shares(self) -> tuple:
        """(seconds per region after removing the dwells from region 5, percent per region)."""
        secs = dict(self.region_time)
        removed = min(secs[5], self.dwell_time)
        secs[5] -= removed
        total = sum(secs.values())
        pct = {f"region{r}": (100.0 * secs[r] / total if total > 0 else 0.0) for r in REGIONS}
        return secs, pct, removed

    def summary(self) -> dict:
        t_end = self.t_test_end or time.time()
        nav_end = self.t_first_selection or t_end
        nav_path = self.path_length_navigation
        region_secs, region_pct, dwell_removed = self.region_shares()
        return {
            "subject": self.subject,
            "run": self.run,
            "timestamp": self.timestamp,
            "end_reason": self.end_reason,
            "test_duration": (t_end - self.t_test_start) if self.t_test_start else None,
            "navigation_duration": (nav_end - self.t_test_start) if self.t_test_start else None,
            "reached_object_selection": self.t_first_selection is not None,
            "n_repositioning": self.n_repositioning,
            "n_objects_moved": len(self.objects_moved),
            "objects_moved": list(self.objects_moved),
            "path_length_max_speed": self.path[MODE_MAX],
            "path_length_reduced_speed": self.path[MODE_REDUCED],
            "path_length_repositioning": self.path[MODE_REPOSITIONING],
            "path_length_transport": self.path[MODE_TRANSPORT],
            "path_length_navigation": nav_path,
            "path_length_total": sum(self.path.values()),
            "optimal_path_length": self.optimal_path_m,
            "normalized_path_length": (nav_path / self.optimal_path_m) if self.optimal_path_m else None,
            "log_dimensionless_jerk": log_dimensionless_jerk(self._nav_samples),
            "region_time_percent": {k: round(v, 1) for k, v in region_pct.items()},
            "region_time_seconds": {f"region{r}": round(region_secs[r], 2) for r in REGIONS},
            "region5_dwell_time_removed": round(dwell_removed, 2),
            "n_dwell": self.n_dwell,
            "n_dwell_declined": self.n_dwell_declined,
            "n_odometry_samples": dict(self.n_samples),
            "odometry_file": os.path.basename(self.path_odometry_csv),
        }

    def save(self) -> dict:
        s = self.summary()
        with open(self.path_json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        with open(self.path_odometry_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "t_test", "x", "y", "theta", "vx", "vy", "vtheta", "mode"])
            w.writerows(self._odometry_log)
        self.saved = True
        print(f"[metrics] saved {self.path_json} and {self.path_odometry_csv} ({len(self._odometry_log)} odometry samples)")
        return s
