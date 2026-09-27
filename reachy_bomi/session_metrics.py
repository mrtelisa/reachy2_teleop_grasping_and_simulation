"""
Session metrics of a reachy_control.py run, written to
results_robot/<subject>_run<N>_session.json (N = --run, 1 or 2; a repeated run
number gets _1, _2, ... appended).
Test = from Control start
(after the cursor preview) to the placement of the carried object. Positions
come from the mobile base odometry.

Phases of the task, delimited by the events reachy_control.py marks:
  outbound    test start -> pre-grasp dwell accepted       driving at full speed (MODE_MAX)
  approach    -> object selection opened                   pre-grasp pose, driving at reduced speed (MODE_REDUCED)
  grasp       -> object in hand, in the carry pose         object selection (+ repositioning), grasp by the robot
  return      -> placement grid opened                     driving with the object (MODE_TRANSPORT)
  placement   -> object placed                             cell selection (+ repositioning), place by the robot
  success                              the object has been placed
  phase_durations, phase_completed     every phase reached (the last one up to the end of the test)
  test_duration, navigation_duration   navigation = up to the first object selection
  n_repositioning (+ _per_phase)       repositioning navigations used, in the grasp / placement phase
  n_dwell, n_dwell_declined (+ _per_phase)   dwells while driving, and those answered "No"
  objects_moved                        objects picked and placed

Driving metrics ("driving" in the json), per driving phase (outbound, approach,
return, repositioning) and over all the driving ("all"):
  driving_time                         time on the cursor map while driving [s], completed dwells removed
  path_length [m], cumulative_rotation [rad]   sum of |d theta|: the turning on the spot, which
                                       the path length does not see (doors, slalom)
  log_dimensionless_jerk               smoothness of the trajectory (Hogan & Sternad 2009)
  stop_time, stop_percent              time with the cursor in region 5 (robot still), completed dwells removed
  region_time_seconds, region_time_percent   per region, completed dwells removed from region 5
Command sequence: the regions the cursor stayed in at least MIN_COMMAND_S
(shorter visits are jitter on a border and are merged into the surrounding
command), within each uninterrupted driving stretch:
  n_commands, n_command_changes, command_changes_per_min
  mean_command_duration, median_command_duration   of the motion commands (regions != 5) [s]
  non_adjacent_percent                 changes between regions that do not touch (e.g. 3 -> 7):
                                       the cursor swept across the grid without stopping
  n_reversals, reversal_percent        sign changes of the linear (forward/back) or angular
                                       (left/right) command between consecutive motion
                                       commands, a stop in between included (2-8, 2-5-8, 1-3...):
                                       overcorrections, as in the steering reversal rate; percent
                                       of the motion -> motion changes
  stop_passage_percent                 motion -> motion changes with a stop (5) in between:
                                       stop-and-go (5 2 5 2) vs fluid (2 3 2 1) driving
  sequence_entropy                     conditional entropy H(next | current) of the command
                                       changes [bits] (as the steering entropy): 0 = predictable
Manual counts, typed in by the experimenter at the end of the run ("manual"):
  n_collisions, n_drops_grasp, n_drops_transport, notes
Kept from the previous version: path_length_<mode>, _navigation, _total,
log_dimensionless_jerk (navigation), region_time_percent/_seconds (all the driving).

Files next to the json:
  <subject>_run<N>_odometry.csv   every odometry sample taken while driving (control-loop rate,
                                  PUBLISH_HZ = 20 Hz): t (unix), t_test (s since test start), x, y [m],
                                  theta [rad], vx, vy [m/s], vtheta [rad/s], mode
  <subject>_run<N>_regions.csv    every cursor region tick while driving: t_test, stretch, mode, region
                                  (to recompute the command metrics with another threshold)
"""

import collections
import csv
import datetime
import json
import math
import os
import re
import time

import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_robot")

MODE_MAX = "max_speed"
MODE_REDUCED = "reduced_speed"
MODE_REPOSITIONING = "repositioning"
MODE_TRANSPORT = "transport"   # driving with the grasped object in hand, towards the placement table
MODES = (MODE_MAX, MODE_REDUCED, MODE_REPOSITIONING, MODE_TRANSPORT)
# Driving phase of each mode (repositioning happens in both the grasp and the placement phase)
MODE_PHASE = {MODE_MAX: "outbound", MODE_REDUCED: "approach", MODE_TRANSPORT: "return",
              MODE_REPOSITIONING: "repositioning"}

# Phases of the task, each ending at the next event
PHASES = ("outbound", "approach", "grasp", "return", "placement")
EVENTS = ("start", "pre_grasp", "object_selection", "object_in_hand", "placement_grid", "placed")

JERK_RESAMPLE_HZ = 20.0   # odometry is sampled at the control loop rate (PUBLISH_HZ)
REGIONS = tuple(range(1, 10))
# Longer gaps between region ticks = control loop not running (preview, dialog): a new driving stretch
REGION_TICK_MAX_GAP_S = 0.5
# A region visit shorter than this is jitter on a border, not a command (reaching_regions.MIN_VISIT_S)
MIN_COMMAND_S = 0.25
STOP_REGION = 5
# (linear, angular) sign of the command of every region: up = forward, left = positive rotation
REGION_SIGNS = {1: (1, 1), 2: (1, 0), 3: (1, -1),
                4: (0, 1), 5: (0, 0), 6: (0, -1),
                7: (-1, 1), 8: (-1, 0), 9: (-1, -1)}

# Counted by the experimenter during the run, typed in at the end
MANUAL_FIELDS = (
    ("n_collisions", "collisions of the robot with obstacles"),
    ("n_drops_grasp", "objects dropped by the robot during the grasp"),
    ("n_drops_transport", "objects dropped by the robot during the transport or the placement"),
)


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


def adjacent(a: int, b: int) -> bool:
    """True if regions a and b touch (side or corner) on the 3 x 3 grid."""
    ra, ca = divmod(a - 1, 3)
    rb, cb = divmod(b - 1, 3)
    return max(abs(ra - rb), abs(ca - cb)) <= 1


def region_visits(ticks: list, trim_end: float = 0.0) -> list:
    """ticks: (t, region) of one driving stretch -> [[region, seconds], ...],
    one per run of equal regions (a tick holds until the next one); the last
    visit, the region-5 hold of a completed dwell, is shortened by trim_end."""
    visits = []
    for i, (t, region) in enumerate(ticks):
        held = ticks[i + 1][0] - t if i + 1 < len(ticks) else 0.0
        if visits and visits[-1][0] == region:
            visits[-1][1] += held
        else:
            visits.append([region, held])
    if trim_end > 0 and visits and visits[-1][0] == STOP_REGION:
        visits[-1][1] = max(0.0, visits[-1][1] - trim_end)
    return visits


def commands(visits: list) -> list:
    """The visits of at least MIN_COMMAND_S, consecutive equal regions merged
    (a short visit in between is jitter on a border)."""
    out = []
    for region, seconds in visits:
        if seconds < MIN_COMMAND_S:
            continue
        if out and out[-1][0] == region:
            out[-1][1] += seconds
        else:
            out.append([region, seconds])
    return out


def command_metrics(stretches: list, driving_time: float) -> dict:
    """Command-sequence metrics (see the module docstring) of a list of
    stretches, each a commands() list; changes never span two stretches."""
    n_commands = n_changes = n_non_adjacent = 0
    n_motion_changes = n_via_stop = n_reversals = 0
    durations = []
    pairs = collections.Counter()
    for cmds in stretches:
        n_commands += len(cmds)
        durations += [s for r, s in cmds if r != STOP_REGION]
        for (a, _), (b, _) in zip(cmds, cmds[1:]):
            n_changes += 1
            pairs[(a, b)] += 1
            if not adjacent(a, b):
                n_non_adjacent += 1
        # Motion commands only, remembering whether a stop came before each one
        last_lin = last_ang = 0
        first, stop_before = True, False
        for region, _ in cmds:
            if region == STOP_REGION:
                stop_before = True
                continue
            lin, ang = REGION_SIGNS[region]
            if not first:
                n_motion_changes += 1
                n_via_stop += stop_before
                if (lin and last_lin and lin != last_lin) or (ang and last_ang and ang != last_ang):
                    n_reversals += 1
            last_lin, last_ang = lin or last_lin, ang or last_ang
            first, stop_before = False, False
    # H(next | current) = -sum p(a, b) log2 p(b | a)
    entropy = None
    if n_changes:
        from_count = collections.Counter()
        for (a, _), c in pairs.items():
            from_count[a] += c
        entropy = 0.0 - sum((c / n_changes) * math.log2(c / from_count[a]) for (a, _), c in pairs.items()) + 0.0
    pct = lambda n, d: (100.0 * n / d) if d else None
    return {
        "n_commands": n_commands,
        "n_command_changes": n_changes,
        "command_changes_per_min": (60.0 * n_changes / driving_time) if driving_time > 0 else None,
        "mean_command_duration": float(np.mean(durations)) if durations else None,
        "median_command_duration": float(np.median(durations)) if durations else None,
        "non_adjacent_percent": pct(n_non_adjacent, n_changes),
        "n_non_adjacent": n_non_adjacent,
        "n_motion_changes": n_motion_changes,
        "n_reversals": n_reversals,
        "reversal_percent": pct(n_reversals, n_motion_changes),
        "stop_passage_percent": pct(n_via_stop, n_motion_changes),
        "sequence_entropy": entropy,
        "command_sequences": [[r for r, _ in cmds] for cmds in stretches],
    }


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
    def __init__(self, subject: str, run: int, dwell_seconds: float = 0.0, results_dir: str = RESULTS_DIR) -> None:
        self.subject = subject
        self.run = run
        self.dwell_seconds = dwell_seconds   # default duration of a dwell (dwell() can override it)
        os.makedirs(results_dir, exist_ok=True)
        base = os.path.join(results_dir, _session_name(subject, run, results_dir))
        self.path_json = base + "_session.json"
        self.path_odometry_csv = base + "_odometry.csv"
        self.path_regions_csv = base + "_regions.csv"
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.t_test_start = None
        self.t_test_end = None
        self.end_reason = None
        self.t_first_selection = None
        self.events = {}   # event name -> time of its first occurrence
        self.n_repositioning = 0
        self.n_repositioning_per_phase = {"grasp": 0, "placement": 0}
        self.objects_moved = []
        self.path = {m: 0.0 for m in MODES}
        self.rotation = {m: 0.0 for m in MODES}
        self.n_samples = {m: 0 for m in MODES}
        self._last_xy = None
        self._last_theta = None
        self._nav_samples = []   # (t, x, y) from test start to the first object selection
        self._mode_samples = {m: [] for m in MODES}   # (t, x, y) per mode, for its jerk
        self._odometry_log = []  # every sample, raw, for the csv
        self._stretches = []     # uninterrupted driving stretches: {mode, ticks [(t, region)], dwell [s]}
        self.n_dwell = 0
        self.n_dwell_declined = 0
        self.n_dwell_per_phase = collections.Counter()
        self.n_dwell_declined_per_phase = collections.Counter()
        self.dwell_time = 0.0   # [s], sum of the durations of the completed dwells (removed from region 5)
        self.manual = {key: None for key, _ in MANUAL_FIELDS}
        self.manual["notes"] = None
        self.saved = False

    # --- events ---
    def event(self, name: str) -> None:
        """Marks the first occurrence of a task event (EVENTS), once the test has started."""
        if self.t_test_start is not None and self.t_test_end is None and name not in self.events:
            self.events[name] = time.time()

    def start_test(self) -> None:
        if self.t_test_start is None:
            self.t_test_start = time.time()
            self.events["start"] = self.t_test_start
            print(f"[metrics] test started ({self.subject})")

    def pre_grasp(self) -> None:
        self.event("pre_grasp")

    def enter_object_selection(self) -> None:
        self.event("object_selection")
        if self.t_first_selection is None and self.t_test_start is not None:
            self.t_first_selection = time.time()
            print(f"[metrics] navigation over after {self.t_first_selection - self.t_test_start:.1f}s, "
                  f"{self.path_length_navigation:.2f} m")

    def object_in_hand(self) -> None:
        self.event("object_in_hand")

    def placement_grid(self) -> None:
        self.event("placement_grid")

    def repositioning(self) -> None:
        self.n_repositioning += 1
        self.n_repositioning_per_phase["placement" if "object_in_hand" in self.events else "grasp"] += 1

    def object_moved(self, name: str) -> None:
        self.objects_moved.append(name)
        self.event("placed")

    def dwell(self, accepted, seconds: float = None) -> None:
        """A dwell in region 5 completed while driving. accepted: True = it
        changed state (pre-grasp pose / object selection / back to selection),
        False = the user answered No and kept driving, None = quit.
        seconds: how long that dwell was (default: dwell_seconds). The hold
        is removed from the end of the current driving stretch."""
        seconds = self.dwell_seconds if seconds is None else seconds
        self.n_dwell += 1
        self.dwell_time += seconds
        phase = MODE_PHASE[self._stretches[-1]["mode"]] if self._stretches else "unknown"
        self.n_dwell_per_phase[phase] += 1
        if accepted is False:
            self.n_dwell_declined += 1
            self.n_dwell_declined_per_phase[phase] += 1
        if self._stretches:
            self._stretches[-1]["dwell"] += seconds

    def end_test(self, reason: str) -> None:
        if self.t_test_end is None:
            self.t_test_end = time.time()
            self.end_reason = reason

    # --- odometry ---
    def sample(self, odom: dict, mode: str) -> None:
        """odom: mobile_base.get_current_odometry() dict (x, y in metres, theta
        in radians); mode: MODE_MAX / MODE_REDUCED / MODE_REPOSITIONING /
        MODE_TRANSPORT, the driving mode the base is in right now. Call it at
        the control-loop rate while driving."""
        if self.t_test_start is None or self.t_test_end is not None:
            return
        x, y = float(odom["x"]), float(odom["y"])
        theta = odom.get("theta")
        t = time.time()
        if self._last_xy is not None:
            self.path[mode] += math.hypot(x - self._last_xy[0], y - self._last_xy[1])
        self._last_xy = (x, y)
        if theta is not None:
            if self._last_theta is not None:
                d = (float(theta) - self._last_theta + math.pi) % (2.0 * math.pi) - math.pi
                self.rotation[mode] += abs(d)
            self._last_theta = float(theta)
        self.n_samples[mode] += 1
        if self.t_first_selection is None:
            self._nav_samples.append((t, x, y))
        self._mode_samples[mode].append((t, x, y))
        self._odometry_log.append((t, t - self.t_test_start, x, y, theta,
                                   odom.get("vx"), odom.get("vy"), odom.get("vtheta"), mode))

    def region_tick(self, region: int, now: float = None, mode: str = MODE_MAX) -> None:
        """Called every control-loop iteration while driving, with the cursor's
        current region and the driving mode. A gap longer than
        REGION_TICK_MAX_GAP_S (preview, dialog) or a new mode starts a new
        driving stretch."""
        if self.t_test_start is None or self.t_test_end is not None or region not in REGIONS:
            return
        now = time.time() if now is None else now
        last = self._stretches[-1] if self._stretches else None
        if last is None or last["mode"] != mode or not last["ticks"] \
                or not 0.0 <= now - last["ticks"][-1][0] <= REGION_TICK_MAX_GAP_S:
            last = {"mode": mode, "ticks": [], "dwell": 0.0}
            self._stretches.append(last)
        last["ticks"].append((now, region))

    # --- results ---
    @property
    def path_length_navigation(self) -> float:
        return self.path[MODE_MAX] + self.path[MODE_REDUCED]

    def _phases(self, t_end: float) -> tuple:
        """({phase: duration}, {phase: completed}) of every phase reached; the
        last one reached runs to t_end and is not completed."""
        durations, completed = {}, {}
        for i, phase in enumerate(PHASES):
            t0, t1 = self.events.get(EVENTS[i]), self.events.get(EVENTS[i + 1])
            if t0 is None:
                durations[phase], completed[phase] = None, False
            elif t1 is None:
                durations[phase], completed[phase] = t_end - t0, False
            else:
                durations[phase], completed[phase] = t1 - t0, True
        return durations, completed

    def _driving(self, stretches: list, modes: tuple) -> dict:
        """Driving metrics (see the module docstring) of the given stretches and modes."""
        region_secs = {r: 0.0 for r in REGIONS}
        cmds = []
        for s in stretches:
            visits = region_visits(s["ticks"], s["dwell"])
            for r, sec in visits:
                region_secs[r] += sec
            cmds.append(commands(visits))
        driving_time = sum(region_secs.values())
        out = {
            "driving_time": driving_time,
            "path_length": sum(self.path[m] for m in modes),
            "cumulative_rotation": sum(self.rotation[m] for m in modes),
            "log_dimensionless_jerk": log_dimensionless_jerk(self._mode_samples[modes[0]]) if len(modes) == 1 else None,
            "stop_time": region_secs[STOP_REGION],
            "stop_percent": (100.0 * region_secs[STOP_REGION] / driving_time) if driving_time > 0 else None,
            "region_time_seconds": {f"region{r}": round(region_secs[r], 2) for r in REGIONS},
            "region_time_percent": {f"region{r}": round(100.0 * region_secs[r] / driving_time, 1)
                                    if driving_time > 0 else 0.0 for r in REGIONS},
        }
        out.update(command_metrics(cmds, driving_time))
        return out

    def summary(self) -> dict:
        t_end = self.t_test_end or time.time()
        nav_end = self.t_first_selection or t_end
        phase_durations, phase_completed = self._phases(t_end)
        driving = {MODE_PHASE[m]: self._driving([s for s in self._stretches if s["mode"] == m], (m,))
                   for m in MODES}
        driving["all"] = self._driving(self._stretches, MODES)
        dwell_removed = sum(sum(sec for _, sec in region_visits(s["ticks"]))
                            - sum(sec for _, sec in region_visits(s["ticks"], s["dwell"]))
                            for s in self._stretches)
        return {
            "subject": self.subject,
            "run": self.run,
            "timestamp": self.timestamp,
            "end_reason": self.end_reason,
            "success": len(self.objects_moved) > 0,
            "test_duration": (t_end - self.t_test_start) if self.t_test_start else None,
            "navigation_duration": (nav_end - self.t_test_start) if self.t_test_start else None,
            "phase_durations": phase_durations,
            "phase_completed": phase_completed,
            "reached_object_selection": self.t_first_selection is not None,
            "n_repositioning": self.n_repositioning,
            "n_repositioning_per_phase": dict(self.n_repositioning_per_phase),
            "n_dwell": self.n_dwell,
            "n_dwell_declined": self.n_dwell_declined,
            "n_dwell_per_phase": dict(self.n_dwell_per_phase),
            "n_dwell_declined_per_phase": dict(self.n_dwell_declined_per_phase),
            "n_objects_moved": len(self.objects_moved),
            "objects_moved": list(self.objects_moved),
            "manual": dict(self.manual),
            # Outbound vs return, the two legs of the same route (compared run 1 vs run 2)
            "path_length_outbound": self.path[MODE_MAX],
            "path_length_return": self.path[MODE_TRANSPORT],
            "cumulative_rotation_outbound": self.rotation[MODE_MAX],
            "cumulative_rotation_return": self.rotation[MODE_TRANSPORT],
            "driving": driving,
            "path_length_max_speed": self.path[MODE_MAX],
            "path_length_reduced_speed": self.path[MODE_REDUCED],
            "path_length_repositioning": self.path[MODE_REPOSITIONING],
            "path_length_transport": self.path[MODE_TRANSPORT],
            "path_length_navigation": self.path_length_navigation,
            "path_length_total": sum(self.path.values()),
            "log_dimensionless_jerk": log_dimensionless_jerk(self._nav_samples),
            "region_time_percent": driving["all"]["region_time_percent"],
            "region_time_seconds": driving["all"]["region_time_seconds"],
            "region5_dwell_time_removed": round(dwell_removed, 2),
            "n_odometry_samples": dict(self.n_samples),
            "odometry_file": os.path.basename(self.path_odometry_csv),
            "regions_file": os.path.basename(self.path_regions_csv),
            "config": {"min_command_s": MIN_COMMAND_S, "dwell_s": self.dwell_seconds,
                       "region_tick_max_gap_s": REGION_TICK_MAX_GAP_S},
        }

    def ask_manual(self) -> None:
        """Asks the experimenter, on the terminal, for what the software cannot
        see: collisions and objects dropped by the robot (ENTER = 0)."""
        print("\n[metrics] experimenter's counts for this run (ENTER = 0)")
        for key, label in MANUAL_FIELDS:
            while True:
                try:
                    answer = input(f"  {label}: ").strip()
                except EOFError:
                    answer = ""
                if not answer:
                    self.manual[key] = 0
                    break
                if answer.isdigit():
                    self.manual[key] = int(answer)
                    break
                print("    a whole number, please")
        try:
            self.manual["notes"] = input("  notes (ENTER = none): ").strip()
        except EOFError:
            self.manual["notes"] = ""

    def save(self) -> dict:
        s = self.summary()
        with open(self.path_json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        with open(self.path_odometry_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "t_test", "x", "y", "theta", "vx", "vy", "vtheta", "mode"])
            w.writerows(self._odometry_log)
        with open(self.path_regions_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_test", "stretch", "mode", "region", "stretch_dwell_s"])
            t0 = self.t_test_start or 0.0
            for i, st in enumerate(self._stretches, start=1):
                w.writerows((f"{t - t0:.4f}", i, st["mode"], r, st["dwell"]) for t, r in st["ticks"])
        self.saved = True
        print(f"[metrics] saved {self.path_json}, {os.path.basename(self.path_odometry_csv)} "
              f"({len(self._odometry_log)} odometry samples) and {os.path.basename(self.path_regions_csv)}")
        return s
