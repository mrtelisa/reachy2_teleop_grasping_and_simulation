"""
Per-trial kinematic metrics of the on-screen cursor reaching tests
(reaching_regions.py, reaching_blind.py). Plain numpy/scipy.

Metrics (None where not computable):
  reaction_time      target shown -> movement onset (speed > onset_speed)
  movement_time      movement onset -> entering the target for the last time
  reach_time         target shown -> entering the target for the last time
  path_length, straight_distance, normalized_path_length
                     path / straight-line displacement onset -> end (1 = straight)
  max_deviation      max perpendicular distance from the ideal line onset -> target centre
  dimensionless_jerk sqrt(0.5 * int |jerk|^2 dt * T^5 / L^2)  (Hogan & Sternad 2009)
  log_dimensionless_jerk  -ln(dimensionless_jerk)
  n_speed_peaks      local maxima of the (smoothed) speed profile above peak_threshold
  mean_speed, peak_speed

summarize() averages them over a whole session, block_summaries() over each
block of N consecutive targets (learning curve).
"""

import collections
import math

import numpy as np
from scipy.signal import find_peaks

METRIC_KEYS = (
    "reaction_time", "movement_time", "reach_time",
    "path_length", "straight_distance", "normalized_path_length",
    "max_deviation", "dimensionless_jerk", "log_dimensionless_jerk",
    "n_speed_peaks", "mean_speed", "peak_speed",
)


def compute_trial_metrics(samples, goal, t_shown, t_reach, onset_speed, peak_threshold, resample_hz):
    """
    samples: list of (t, x, y) or (t, x, y, vx, vy), from target shown to
    trial end. Without velocities they are derived from the positions.
    goal: (x, y). t_reach: time the target was entered for the last time
    (None if missed -> the whole trial is used). Units are whatever the
    positions are in (canvas px for the cursor).
    """
    m = {k: None for k in METRIC_KEYS}
    if len(samples) < 3:
        return m
    arr = np.asarray(samples, dtype=float)
    t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
    if arr.shape[1] >= 5:
        vx, vy = arr[:, 3], arr[:, 4]
    else:
        if np.any(np.diff(t) <= 0):
            return m
        vx, vy = np.gradient(x, t), np.gradient(y, t)
    speed = np.hypot(vx, vy)

    if t_reach is not None:
        m["reach_time"] = float(t_reach - t_shown)

    moving = np.flatnonzero(speed > onset_speed)
    if moving.size == 0:
        return m
    i_onset = int(moving[0])
    t_onset = float(t[i_onset])
    m["reaction_time"] = t_onset - t_shown

    t_end = t_reach if t_reach is not None else float(t[-1])
    i_end = int(np.searchsorted(t, t_end, side="right"))
    if i_end - i_onset < 3:
        return m
    m["movement_time"] = t_end - t_onset

    xy = np.column_stack([x[i_onset:i_end], y[i_onset:i_end]])
    seg = np.diff(xy, axis=0)
    path_length = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    p0 = xy[0]
    g = np.asarray(goal, dtype=float)
    # Straight-line distance actually covered (onset -> where the movement
    # ended, i.e. the target's edge when reached), so a perfectly straight
    # reach gives normalized_path_length == 1.
    straight = float(np.hypot(*(xy[-1] - p0)))
    m["path_length"] = path_length
    m["straight_distance"] = straight
    if straight > 1e-6:
        m["normalized_path_length"] = path_length / straight
    # Max deviation from the ideal line onset -> target centre
    ideal = float(np.hypot(*(g - p0)))
    if ideal > 1e-6:
        d = (g - p0) / ideal
        rel = xy - p0
        m["max_deviation"] = float(np.max(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])))

    sp = speed[i_onset:i_end]
    m["mean_speed"] = float(np.mean(sp))
    m["peak_speed"] = float(np.max(sp))

    # Uniform resampling before differentiating: sample timestamps jitter.
    dt = 1.0 / resample_hz
    tt = t[i_onset:i_end]
    tu = np.arange(tt[0], tt[-1], dt)
    if tu.size < 5:
        return m
    xu = np.interp(tu, tt, x[i_onset:i_end])
    yu = np.interp(tu, tt, y[i_onset:i_end])
    vxu, vyu = np.gradient(xu, dt), np.gradient(yu, dt)
    axu, ayu = np.gradient(vxu, dt), np.gradient(vyu, dt)
    jxu, jyu = np.gradient(axu, dt), np.gradient(ayu, dt)
    duration = tu[-1] - tu[0]
    # Dimensionless jerk (Hogan & Sternad 2009), amplitude = straight-line displacement
    if straight > 1e-6 and duration > 0:
        jerk_int = float(np.sum(jxu ** 2 + jyu ** 2) * dt)
        dj = math.sqrt(0.5 * jerk_int * duration ** 5 / straight ** 2)
        m["dimensionless_jerk"] = dj
        m["log_dimensionless_jerk"] = -math.log(dj) if dj > 0 else None

    # Speed peaks: light moving-average smoothing, then local maxima above threshold
    su = np.hypot(vxu, vyu)
    k = max(1, int(round(0.1 * resample_hz)))  # 100 ms window
    su_s = np.convolve(su, np.ones(k) / k, mode="same")
    peaks, _ = find_peaks(su_s, height=peak_threshold, prominence=peak_threshold * 0.25)
    m["n_speed_peaks"] = int(peaks.size)
    return m


# Metrics averaged by summarize()/block_summaries(): all of them, so adding a
# metric to METRIC_KEYS is enough to have it in the session and block summaries.
SUMMARY_KEYS = METRIC_KEYS


def summarize(results: list, keys=SUMMARY_KEYS) -> dict:
    """Success counts/rates plus the mean of each metric over successful trials."""
    n = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    summary = {
        "n_trials_done": n,
        "n_success": n_success,
        "success_rate": (n_success / n) if n else None,
    }
    for key in keys:
        vals = [r[key] for r in results if r.get("success") and r.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(vals)) if vals else None
    return summary


def block_summaries(results: list, block_size: int, keys=SUMMARY_KEYS,
                    group_key: str = "target_number") -> list:
    """One summarize() per block of `block_size` consecutive targets: the same
    statistics as the whole session, but computed on targets 1..block_size,
    block_size+1..2*block_size, and so on (so a session of 64 targets with
    block_size 8 gives 8 blocks, one of 96 gives 12 -- nothing to change when
    the number of targets changes, as long as it stays a multiple of
    block_size; a last, incomplete block is still reported, with
    complete=False).

    results: the trial dicts of the test; each one is assigned to a block by
    its `group_key` (target_number, 1-based). The optional keys t_shown/t_end
    (block duration), points (score) and over_time are used when the trials
    have them.
    """
    block_size = int(block_size)
    if not results or block_size <= 0:
        return []
    blocks = collections.OrderedDict()
    for r in results:
        number = r.get(group_key)
        if number is None:
            continue
        blocks.setdefault((int(number) - 1) // block_size + 1, []).append(r)
    summaries = []
    for b, rows in sorted(blocks.items()):
        numbers = sorted({int(r[group_key]) for r in rows})
        block = {
            "block": b,
            "block_size": block_size,
            "first_target": (b - 1) * block_size + 1,
            "last_target": b * block_size,
            "n_targets_in_block": len(numbers),
            # False only for the last block of an aborted session
            "complete": len(numbers) == block_size,
        }
        starts = [r["t_shown"] for r in rows if r.get("t_shown") is not None]
        ends = [r["t_end"] for r in rows if r.get("t_end") is not None]
        if starts and ends:
            block["t_start"] = min(starts)
            block["t_end"] = max(ends)
            block["duration"] = max(ends) - min(starts)
        if any("points" in r for r in rows):
            block["score"] = sum(r.get("points") or 0 for r in rows)
        block.update(summarize(rows, keys))
        if any("over_time" in r for r in rows):
            block["n_over_time"] = sum(1 for r in rows if r.get("over_time"))
            block["over_time_trials"] = [r["trial"] for r in rows if r.get("over_time")]
        summaries.append(block)
    return summaries
