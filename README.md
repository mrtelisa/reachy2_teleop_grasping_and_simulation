# reachy2_teleop_grasping_and_simulation

Hand-driven BoMI cursor and the **9-region blind reaching test** used to check whether a participant can learn the hand -> cursor map before teleoperating Reachy 2.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), and a calibrated autoencoder map turns the hand pose into a 2D cursor. The screen is split into the same 3×3 regions as the teleoperation interface. No robot is involved: everything runs on the operator PC.

---

## Requirements

```bash
python3 -m venv ~/.venvs/reachy_bomi && source ~/.venvs/reachy_bomi/bin/activate
pip install -r requirements.txt     # mediapipe, opencv-python, tensorflow, numpy, scipy
```

The MediaPipe **Tasks API** (`HandLandmarker`) needs a `hand_landmarker.task` model file. The `mediapipe` pip package doesn't include it, so a copy is tracked at `scripts/hand_landmarker.task`, which is the default for `--model`. If you need to download it again:

```bash
curl -o scripts/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

All the scripts are run from `reachy_bomi/`:

```bash
cd reachy_bomi
```

Every window the scripts open (cursor map, tests, webcam, calibration) goes on the **laptop's built-in screen**, whichever terminal or monitor you launch them from ([`display.py`](reachy_bomi/display.py)). Fullscreen windows fill that screen. To use another screen, set `BOMI_SCREEN` to its `xrandr` output name, e.g. `BOMI_SCREEN=DP-1 python3 reaching_blind.py ...`. `customize_bomi.py`, `reaching_regions.py` and `reaching_blind.py` also open a resizable **camera window** (webcam + tracked hand) on the **external monitor**, for the experimenter only (`BOMI_MONITOR` to choose another output). The keys (`ENTER`, `Q`, ...) are read by the window, not by the terminal: a new window takes the keyboard focus, and if you click back on the terminal, click the window (or `Alt+Tab`) before pressing them.

---

## Calibration maps

This follows the markerlessBoMI procedure. There is **one** autoencoder map, trained once on a single 90 s recording of the experimenter's hand. Each participant only gets a customization of it (rotation / gain / offset). The tools share the `calibrations/` folder, and names can be given without `.npz`.

```bash
python3 calibrate_bomi.py [NAME] [--cam 0] [--model PATH] [--duration 90] [--recalibrate]   # once
python3 customize_bomi.py [SUBJECT] [--base shared] [--cam 0] [--model PATH]                # per participant
python3 load_bomi.py SUBJECT|NAME [--cam 0] [--model PATH]                                  # try a map fullscreen (familiarisation)
```

**Once: `calibrate_bomi.py`** (markerlessBoMI's "Calibration" + "Calculate BoMI map"):
1. *Continuous calibration*: after `SPACE`, the hand is recorded continuously for `CALIB_DURATION_S` = 90 s. Every frame with a tracked hand is kept and no samples are picked by hand, so keep moving the hand through the whole workspace while the remaining time is shown. The raw 42-feature samples are saved to `calibrations/shared_calib.npy`. If that file already exists, the tool offers to reuse it and skip the recording (`--recalibrate` forces a new one). This is how you retrain after changing the AE hyperparameters in `bomi.py`.
2. *Offline training*: the autoencoder (32-32-2-32-32, tanh, Adam 0.02, 3001 full-batch epochs) is trained on the saved samples the same way markerlessBoMI's `train_ae` does it:
   - all-zero rows are dropped;
   - the samples are shuffled and split 80/20 into train and test;
   - the screen scale/offset comes from the training latent codes.

   The VAF and the latent-variance share of each code unit are printed for train and test. They are also stored in the `.npz` and shown again whenever the map is loaded. Training takes a couple of minutes on CPU. The map is saved to `calibrations/shared.npz` and previewed live (`Q` quits). With a `NAME` other than `shared`, the map is saved under that name (then use `customize_bomi.py --base NAME`).

**Per participant: `customize_bomi.py SUBJECT`** (markerlessBoMI's "Customization"):
- It loads `calibrations/shared.npz` and shows the fullscreen cursor map driven by the participant's hand.
- Rotate/flip/scale/offset the map (`[ ]`, `i`/`o`, `-`/`+`, `hjkl`, `r` resets) until the whole 3×3 grid is comfortably reachable and the rest position sits in region 5.
- `S` asks for the participant id (`SUBJECT` from the command line is the default, `-` cancels) and saves the map as `calibrations/<id>_<YYYYMMDD_HHMMSS>.npz`. For example, typing `elisa` gives `elisa_20260916_155827.npz`.
- `reaching_regions.py` and `load_bomi.py`, given `SUBJECT`, load the **latest** of these files. A redone customization therefore simply replaces the previous one, and the older files stay on disk.
- Retraining the shared map invalidates the existing customized maps, because they were made on the old one. `calibrate_bomi.py` warns about this.

`load_bomi.py SUBJECT` shows a participant's latest map (or any map by name, e.g. `shared`) on the fullscreen cursor map. Use it for familiarisation or to sanity-check a map.

---

## Blind reaching test

```bash
python3 reaching_regions.py --subject S001 [--calib <name>] [--cam 0] [--sequence file.csv]
```

The 1200×650 canvas is scaled to the whole screen (the map's 2550×1500 space is scaled onto it) and divided into the 9 regions of the teleoperation interface:

```
 1 | 2 | 3
 4 | 5 | 6      5 = home: a circle at the centre
 7 | 8 | 9      each outer region: a target circle at its centre (all circles: radius 70 px)
```

The sequence alternates home -> region -> home -> region .... There are **96 targets**, each of the 8 outer regions 12 times, in a seeded random order: every region appears once before any repeat, and never twice in a row. The sequence is frozen in [`config/cursor_regions.csv`](config/cursor_regions.csv), so it is the same for every participant. It is regenerated only if the file is missing (`N_REPETITIONS` in the script).

A session has **two parts**, each the whole sequence (96 targets), with a **3 min pause** in between (`N_PARTS`, `PAUSE_S`): 96 targets -> back to the home -> pause -> 96 targets.

How a part runs:
1. **Home**: the centre circle is shown. The **timer of the part starts the first time the cursor enters it**.
2. **Region**: as soon as the home is reached, it disappears and the **yellow target circle at the centre of the region** is shown. For the first **1 s** (`HIDDEN_S`) the **cursor is not drawn**. The goal is to see whether the participant can reach the target from the learned map alone, without visual feedback. After that second the cursor reappears.
3. Every goal, home or target, is reached when the cursor **stays inside its circle for 0.5 s** (`DWELL_S`, the same for both), whether the cursor is hidden or not. Leaving the circle restarts the count. A circle turns blue while the cursor is inside it (a target only once the cursor is visible). The home is then shown again for the return.
4. After the last target of part 1 and the return to the home, the **pause** starts: no goal, the cursor is visible, the time left is on screen, and nothing is recorded. When it is over the home appears again, and part 2 starts at the first entry into it.
   The **timer** at the bottom of the screen counts only the parts: it stops when the pause starts and stays still until the cursor enters the home again.
5. The session ends when the last target of part 2 is reached, or on `Q`/`Esc`.

Results go to `results_regions/`. A subject with previous sessions gets `_1`, `_2`, ... appended to the file names.

| File | Content |
|---|---|
| `<subject>_regions_trials.csv` | one row per goal (home or region), with its `part` (1 = before, 2 = after the pause): times and metrics |
| `<subject>_regions_blocks.csv` | one row per block of 8 targets (one repetition of every region), with its `part`: the learning curve, 12 blocks per part (24 in all) |
| `<subject>_regions_trajectory.csv` | every cursor sample: part, trial, kind, region, t, x, y, `cursor_visible` (nothing during the pause) |
| `<subject>_regions_summary.json` | `time_total` (= the on-screen timer), `time_part_1`, `time_part_2` (first entry into the home of the part -> its last goal); means over all region goals and the returns, **per part** (`parts`) and their difference (`part_2_vs_part_1`), per region and per block; the pause times (`pause`, `wall_clock_duration` = with the pause), plus the config |

Metrics for every goal ([`reaching_metrics.py`](reachy_bomi/reaching_metrics.py), in canvas px). For a region goal, the ideal target point is the centre of its target circle.

| Metric | Meaning |
|---|---|
| `reach_time` | goal shown -> entering the target (or home) circle, for the entry that completed the 0.5 s dwell |
| `reaction_time`, `movement_time` | goal shown -> movement onset, and onset -> entering |
| `path_length`, `normalized_path_length` | path / straight-line displacement (1 = perfectly straight) |
| `max_deviation` | max perpendicular distance from the ideal line onset -> goal centre |
| `dimensionless_jerk`, `log_dimensionless_jerk` | smoothness (Hogan & Sternad 2009) |
| `n_speed_peaks`, `mean_speed`, `peak_speed` | speed profile |

Metrics for region goals only:

| Metric | Meaning |
|---|---|
| `reached_hidden` | 100 if the target circle was entered (entry of the completed dwell) while the cursor was still hidden, else 0 |
| `region_at_reveal`, `region_at_reveal_correct` | region of the cursor when it reappears, and 100 if it is the target |
| `first_region`, `first_region_correct` | first region visited, i.e. stayed in for at least 0.25 s (the target counts at once), and 100 if it is the target |
| `n_wrong_regions` | distinct other regions visited before the target |

---

## 3-target blind reaching test (pre / post training)

```bash
python3 reaching_blind.py --subject S001 --phase pre   # before the training
python3 reaching_regions.py --subject S001             # training
python3 reaching_blind.py --subject S001 --phase post  # after the training
```

During the test the cursor is **never shown**. Only the **current target** is on screen: an empty yellow circle (radius 60 px), shown on its own like the targets of `reaching_regions.py`. Every **4 s** (`TRIAL_S`) it is replaced by the next one, whatever the cursor did. The cursor is still tracked, so the results show whether the participant got there from the learned map alone.

- There are **15 trials**: each of the 3 targets 5 times, all 3 once before any repeat, never twice in a row. Positions (one per third of the screen width, at least 400 px apart and 250 px apart vertically between the highest and the lowest, and **never overlapping a circle of `reaching_regions.py`** — its 8 targets and the home — with at least 20 px between the two, `REGION_CLEARANCE`) and order are drawn at random (seeded) and frozen in [`config/blind_targets.csv`](config/blind_targets.csv) (`trial,target,x,y`), so the pre and post sessions, and every participant, get the same targets.
- Before the start only the cursor is visible (no target), so the participant can see where the hand is. After 2 s (`START_CURSOR_S`), `ENTER` (experimenter) starts the session and the cursor disappears. `Q`/`Esc` aborts it, and the results so far are still saved.
- The dot in the top-right corner is green while the hand is tracked and grey when tracking is lost. It shows no position.
- `--show-cursor` draws the cursor. Use it only to test the setup, never with a participant.

Results go to `results_blind/<subject>_blind_<phase>_{trials,blocks,trajectory}.csv` and `_summary.json`. The blocks file has one row per block of 3 trials (each target once). A **post** session is compared with the subject's latest **pre** session: the differences (post − pre) of the main metrics are printed and stored under `comparison_with_pre` in the summary.

Per-trial metrics (canvas px, distances from the target centre). The `reaching_metrics.py` kinematics are also computed, up to the first entry:

| Metric | Meaning |
|---|---|
| `hit`, `reach_time` | 100 if the cursor entered the target circle, and the time to the first entry |
| `time_in_target`, `on_target_at_end` | % of the 5 s inside the circle, and 100 if inside when the target changes |
| `initial_error`, `final_error`, `end_error`, `min_error` | distance when shown, when the target changes, mean over the last 1 s, and closest approach |
| `relative_final_error` | `final_error / initial_error` (0 = perfect, 1 = did not get closer) |
| `chosen_target`, `chosen_correct` | the target closest to the cursor when the target changes, and 100 if it is the current one |
| `hand_lost` | % of the trial without a tracked hand |

---

## Package layout

```
reachy2_teleop_grasping_and_simulation/
├── reachy_bomi/
│   ├── bomi.py                 # shared hand -> cursor chain: MediaPipe, autoencoder map, filter, calibrations/ helpers
│   ├── display.py              # puts the OpenCV windows on the laptop screen, the camera view on the monitor
│   ├── calibrate_bomi.py       # once: 90 s continuous calibration + offline AE training -> calibrations/shared.npz
│   ├── customize_bomi.py       # per participant: rotate/flip/scale/offset the shared map -> calibrations/<SUBJECT>_<date>_<time>.npz
│   ├── load_bomi.py            # try a participant's / any saved map on the fullscreen cursor map
│   ├── reaching_regions.py     # 9-region blind reaching test
│   ├── reaching_blind.py       # 3-target blind reaching test, pre/post training
│   └── reaching_metrics.py     # per-trial kinematic metrics
├── config/
│   ├── cursor_regions.csv      # frozen region sequence of the training (96 targets, done twice)
│   └── blind_targets.csv       # frozen targets of the pre/post blind test (15 trials)
├── scripts/
│   └── hand_landmarker.task    # MediaPipe model (tracked, see Requirements)
├── calibrations/               # shared_calib.npy, shared.npz, <SUBJECT>_<date>_<time>.npz (not tracked)
├── requirements.txt
└── README.md
```
