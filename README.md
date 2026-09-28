# reachy2_teleop_grasping_and_simulation

`reachy_bomi` — a package that turns **hand movements into control of a Reachy 2 robot**: driving its mobile base, and selecting + grasping objects it sees through its torso depth camera.

A webcam tracks the operator's hand with [MediaPipe](https://developers.google.com/mediapipe), a calibrated autoencoder map converts the hand pose into a 2D cursor, and the cursor position drives either base velocity commands or object hover-selection. Grasp planning turns a YOLOv8 detection + depth camera point cloud into pre-grasp/grasp/lift end-effector poses, executed over [`reachy2_sdk`](https://github.com/pollen-robotics/reachy2-sdk) (gRPC over IP) — no ROS 2 networking is involved between the operator's PC and the robot.

This work started from a ROS 1 implementation written for the TIAGo robot and was ported to ROS 2 for Reachy 2, then moved off ROS 2 topics onto `reachy2_sdk` so the operator's PC doesn't need a ROS 2 distro matching the robot's.

---

## How it works

Everything runs on a **single PC** (any PC with a webcam and network access to the robot); the head- or torso-camera live feed runs in a small helper subprocess (`camera_viewer.py`) so its network round-trips never block the main cursor/velocity loop, but there's no ROS 2/robot-side process split:

```
webcam → MediaPipe → autoencoder cursor → 9-region velocity → reachy2_sdk (gRPC/IP) → mobile base
                                        ↓ (cursor dwell)
                          arms → pre-grasping pose → object hover-select ⇄ repositioning
                                        ↓                    (min-speed nav, torso camera)
      torso depth camera → YOLOv8 → point cloud → grasp planning → arm execution
```

`reachy_control.py` is the script with a CLI/`main()` for the real test on the robot — `bomi_teleop.py`, `reachy_detection.py`, `reachy_selection.py`, `reachy_pregrasp.py`, `reachy_grasp.py`, `camera_viewer.py`, `stream.py`, `safety.py`, `graphs.py` and `session_metrics.py` are library modules it's built from; `calibrate_bomi.py` (once) and `customize_bomi.py` (per participant) build the calibration maps on their own, without the robot.

- **`bomi_teleop.py`** — hand tracking → autoencoder cursor → 9-region velocity building blocks (continuous calibration phase, cursor preview, the BoMI map with its offline training, cursor filter, velocity helpers). `BoMIMap.fit` trains the autoencoder the way markerlessBoMI's `train_ae` does (80/20 split, VAF and latent-variance report in `BoMIMap.metrics`); `save_map_bomi`/`load_map_bomi` (de)serialize a fitted map to/from a `.npz` file; `resolve_calib_path` / `resolve_samples_path` / `resolve_subject_map_path` turn a map name or participant id into paths inside `CALIB_DIR` (the `calibrations/` folder next to the package; the saved `.npy`/`.npz` files are not tracked by git).
- **`calibrate_bomi.py`** — standalone tool, run **once**: 90 s continuous calibration saved raw to `calibrations/shared_calib.npy`, offline autoencoder training saved to `calibrations/shared.npz`, then a live preview (nothing sent anywhere, no robot needed). Offers to reuse the raw samples if they are already there (retraining without recording).
- **`customize_bomi.py`** — standalone tool, run **per participant**: rotate/flip/scale/offset the shared map live on the participant's hand and save it as `calibrations/<SUBJECT>_<date>_<time>.npz`, the map the robot scripts load from `--subject SUBJECT` (no robot).
- **`reachy_detection.py`** — torso camera → YOLOv8 detection → depth point cloud → grasp geometry building blocks: `capture_and_detect` (grab frame + detect, optionally pre-filtered down to presentable candidates via an injected predicate) and `build_object_point_cloud` (crop/fuse/isolate/measure once an object is confirmed).
- **`reachy_selection.py`** — hover-to-select/confirm UI on top of `reachy_detection.py`'s boxes: dwell-to-select, a **Repositioning** button, Yes/No confirm, and `presentable_filter` (the reachability/gripper-size pre-filter fed into `capture_and_detect`). Dwelling on Repositioning doesn't act itself — `select_object_to_grasp_bomi` returns the `REPOSITION_REQUESTED` sentinel and lets `reachy_control.py` drive the actual mode switch, so this module never depends back on it. `select_object_to_grasp_bomi`/`confirm_grasp_bomi` drive the UI from the BoMI cursor.
- **`reachy_pregrasp.py`** — moves both arms to the pre-grasping posture (non-blocking) and waits for it, showing a progress-bar window while holding the mobile base at zero speed and keeping the BoMI cursor alive.
- **`reachy_grasp.py`** — grasp planning/execution library: from an `ObjectGeometry` (position, axes, width/height, table normal), computes pre-grasp/grasp/lift end-effector poses and drives the arm through them. Also exposes `is_roughly_reachable`, the cheap single-point reachability check `reachy_selection.py` uses to pre-filter detections.
- **`camera_viewer.py`** — standalone script that shows a live feed from either the head/teleop camera or the torso/depth camera (`--camera teleop|torso`); spawned by `reachy_control.py` as its own OS process during Control/pre-grasping pose (teleop) and during repositioning (torso), so the camera's network round-trips stay out of the cursor/velocity loop. Its window is moved to a fixed screen position so it lands beside the cursor-map window instead of on top of it.
- **`stream.py`** — blocking camera live feed used by `camera_viewer.py`.
- **`graphs.py`** — matplotlib diagnostics (point cloud stages, planned grasp), called from `reachy_detection.py`/`reachy_control.py`; every figure is also saved to `graphs/`.
- **`safety.py`** — quit/shutdown safety net: a local `quit_requested` check (Q/ESC or window closed, while a cv2 window has focus) plus an OS-level global watcher (`pynput`, works regardless of focus, even mid-`arm.goto`) that triggers `emergency_shutdown`.
- **`session_metrics.py`** — session metrics (phase durations, path length and rotation per leg, command sequence, region shares, dwells, manual counts) and the 20 Hz odometry and region logs, written to `results_robot/` at the end of every `reachy_control.py` run. `plot_odometry.py` draws the path of a session from its odometry csv.

Dependencies between these run one way only, with no cycles: `reachy_grasp.py`/`bomi_teleop.py` have no dependency on the rest, `reachy_detection.py` depends only on `reachy_grasp.py` (for the shared `ObjectGeometry` type), `reachy_selection.py` depends on both, and `reachy_control.py` ties everything together.

---

## Requirements

- The real Reachy 2 robot reachable over the network, with its SDK server running (mobile base + arms + torso depth camera, depending on which script you run).
- A webcam (for `reachy_control.py`).
- Python deps, in the same environment used to run the scripts:

```bash
pip install reachy2-sdk mediapipe opencv-python tensorflow numpy scipy ultralytics matplotlib open3d pynput
```

`reachy_control.py` uses the MediaPipe **Tasks API** (`HandLandmarker`), which needs a `hand_landmarker.task` model file — it's not bundled with the `mediapipe` pip package, so a copy is tracked at the repo root (the default for `--model`). If you need to re-download it:

```bash
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

`reachy_control.py` uses YOLOv8 (`ultralytics`) for object detection — `yolov8n.pt` (COCO-pretrained) is tracked at the repo root (the default for `--yolo-model`, whatever folder you run from), or pass `--yolo-model` to use different weights.

No ROS 2 install is required to run these scripts (they only talk to the robot via `reachy2_sdk`). The package is still packaged as an `ament_python` ROS 2 package for convenience if you keep it in a ROS 2 workspace, but nothing in the runtime code imports `rclpy`.

---

## Installation

Clone the package into the `src/` folder of a (ROS 2, optional) workspace and build it:

```bash
cd ~/ros2_ws/src
git clone https://github.com/mrtelisa/reachy2_teleop_grasping_and_simulation.git
cd ~/ros2_ws
colcon build --packages-select reachy_bomi
source install/setup.bash
```

Or just run the scripts directly with `python3` — no build step required, since none of them depend on ROS 2 at runtime.

---

## Usage

Every window the scripts open (cursor map, cameras, confirmations, matplotlib graphs, the `camera_viewer.py` subprocess) goes on the **laptop's built-in screen**, whichever terminal or monitor you launch them from ([`display.py`](reachy_bomi/display.py)). Fullscreen windows fill that screen. To use another screen, set `BOMI_SCREEN` to its `xrandr` output name (e.g. `BOMI_SCREEN=DP-1`). The screens are read once at start-up: if you unplug/replug the monitor while a script runs, restart it.

### Full BoMI flow: teleop + grasp — `reachy_control.py`

The real entry point — everything else in the package exists to support this.

```bash
python3 reachy_bomi/reachy_control.py [robot_ip] [--cam 0] [--model hand_landmarker.task] [--yolo-model yolov8n.pt] [--conf 0.5] [--calib NAME] [--subject ID]
# or, from a built ROS 2 workspace:
ros2 run reachy_bomi reachy_control [robot_ip] [--cam 0] [--model hand_landmarker.task] [--yolo-model yolov8n.pt] [--conf 0.5] [--calib NAME] [--subject ID] --run 1|2
```

`robot_ip` is optional if you've set `DEFAULT_ROBOT_IP` in `reachy_control.py` to your robot's IP; otherwise pass it explicitly. `--subject` and `--run` (1 or 2: every participant does two runs) name the session metrics files (see [Session metrics](#session-metrics)).

**Phase 1 — Map loading:** no calibration happens on the robot. The participant's map `calibrations/<subject>_<date>_<time>.npz` is made beforehand with `customize_bomi.py SUBJECT` (see [Calibration maps](#calibration-maps--calibrate_bomipy-once-customize_bomipy-per-participant)) and picked up from `--subject` (the latest one for that id; if `calibrations/<subject>.npz` itself exists, e.g. `--subject shared` or a full map name with or without `.npz`, that exact file is loaded instead); `--calib NAME` loads any other saved map. A missing map fails fast, before connecting to the robot, listing the maps actually found.

**Phase 2 — Cursor preview:** the same cursor-map window used in Control is shown, but nothing is sent to the robot yet — also where Reachy's head/teleop camera live feed starts streaming, as its own subprocess (`camera_viewer.py`). Hold the cursor centered (region 5) for `SELECTION_HOLD_SECONDS` to proceed into Control, or `Q`/`Esc`/close a window to quit.

**Phase 3 — Control:** hand → autoencoder cursor → 9-region base velocity, mapped and sent to the robot. Hold the cursor centered (region 5) for `SELECTION_HOLD_SECONDS` straight: a Yes/No dialog ("Do you want to continue on the pipeline?") then either moves the arms into the pre-grasping pose or, on "No", goes back to driving through a new cursor preview.

```
 1 | 2 | 3      center (5)          → stop
 4 | 5 | 6      middle column (2,8) → linear only
 7 | 8 | 9      middle row (4,6)    → angular only
                corners (1,3,7,9)   → linear + angular
```

**Phase 3.5 — Pre-grasping pose:** both arms bend to `reachy_pregrasp.PRE_GRASP_ELBOW_PITCH_DEG` (`reachy_pregrasp.goto_pre_grasp_pose`/`wait_for_pre_grasp_pose`), the base holds zero speed; the head-camera feed from Phase 2 just keeps streaming throughout. The cursor/9-region map reappears as a cursor preview (no speed commands); holding the cursor centered resumes Control at reduced velocity (`HALVED_SPEED_FACTOR`). Hold the cursor centered again for `SELECTION_HOLD_SECONDS` (and answer "Yes") to open object selection (the head-camera subprocess is stopped right there, since Phase 4's own checkpoint streams the same camera differently, in-process); the base is held at zero speed here rather than powered off, so it's still ready to drive during Repositioning.

**Phase 4 — Object selection / grasp:** capture → hover-select → Yes/No confirm (`reachy_selection.select_object_to_grasp_bomi`/`confirm_grasp_bomi`) → build point cloud → live feed checkpoint → plan → execute, every hover point being the BoMI cursor (mapped into the window's pixel space) instead of a mouse. "No" re-offers hover-select on the same captured frame; the mobile base is only powered off once an object is confirmed with "Yes". After the object has been placed, a last Yes/No dialog asks whether to pick another object: "Yes" loops back to a fresh capture, "No" ends the session (`_finish_session`: back up, rotate, default posture). Quitting (`Q`/`Esc`/X) at any point in Phase 4, or a failed grasp/place, ends the run — there's no going back to Control.

**Placement height:** the destination table may be at a different height from the grasp table. When the placement grid is built, `reachy_selection.estimate_table_height_change` fits the destination table plane (RANSAC, points within `DEST_TABLE_MAX_HEIGHT_DIFF_M` = 20 cm of the grasp table and within reach) and `plan_place` shifts the transit and release poses by the height difference, so the object comes down by the lift distance onto the new table. A fit that is too tilted (> 15°), too far (> 20 cm) or not found falls back to the grasp table height, with a message on the terminal.

**Phase 4.5 — Repositioning (opened from object selection):** dwelling on the **Repositioning** button for `REPOSITIONING_HOVER_SECONDS` hands off to `reachy_control.repositioning_navigation`, which reuses the same 9-region cursor UI as Control but caps velocity to `bomi_teleop.MIN_LINEAR`/`MIN_ANGULAR` only (no ramp-up), and streams the **torso/depth camera** instead of the head one (`camera_viewer.py --camera torso`), so you can see what you're driving toward. A cursor-preview sub-phase runs first (nothing sent) until the cursor is re-centered; holding it centered again for `MODE_SWITCH_HOLD_SECONDS` (10 s) stops the base and returns to object selection with a **freshly recaptured** frame (the robot has moved, so the old detections/point positions no longer apply).

ESC/Q stop the robot from *any* window or the terminal, at any point — see `safety.py`.

### Session metrics

Every run of `reachy_control.py` writes `results_robot/<subject>_run<N>_session.json` (`--subject`, default `S000`; `--run` N = 1 or 2; a repeated run gets `_1`, `_2`, ... appended, e.g. `S001_run1_1`), collected by [`session_metrics.py`](reachy_bomi/session_metrics.py) from the mobile base odometry and the pipeline events. Written in `main()`'s `finally`, so it exists even after a quit or an abort (`end_reason`: `finished` / `quit` / `aborted: ...`).

At the end of the run, once the robot is off, the terminal asks the experimenter for what the software cannot see — collisions with obstacles, objects dropped by the robot during the grasp and during the transport/placement (ENTER = 0), plus free notes — and saves the JSON again (`manual`).

The task is split into phases by the pipeline events: **outbound** (start → pre-grasp dwell accepted, full speed), **approach** (→ object selection opened, reduced speed), **grasp** (→ object in the carry pose), **return** (→ placement grid opened, with the object), **placement** (→ object placed). Run 1 and run 2 are compared phase by phase (outbound with outbound, return with return).

| Field | Meaning |
|---|---|
| `success` | the object has been placed |
| `phase_durations`, `phase_completed` | duration [s] of every phase reached; the last one reached runs to the end of the test and is not completed |
| `test_duration` | from the start of Control after the cursor preview (Reachy starts moving) to the end of the test |
| `navigation_duration` | from the same start to the first time object selection opens (`reached_object_selection` says whether it did) |
| `n_repositioning`, `n_repositioning_per_phase` | how many times repositioning navigation was used, in the grasp / placement phase |
| `n_dwell`, `n_dwell_declined` (+ `_per_phase`) | dwells completed while driving, and how many of them were answered "No" to "Do you want to continue on the pipeline?" |
| `n_objects_moved`, `objects_moved` | objects picked **and** placed (YOLO class names, in order) |
| `manual` | `n_collisions`, `n_drops_grasp`, `n_drops_transport`, `notes`, typed in by the experimenter |
| `path_length_outbound`, `path_length_return` | base path [m] of the outbound and return legs |
| `cumulative_rotation_outbound`, `cumulative_rotation_return` | sum of \|Δθ\| [rad] on the two legs: the turning on the spot, which the path length does not see |
| `path_length_max_speed` / `_reduced_speed` / `_repositioning` / `_transport`, `path_length_navigation`, `path_length_total` | path [m] per driving mode; max + reduced; all |
| `log_dimensionless_jerk` | smoothness of the navigation trajectory (start → first object selection) |
| `region_time_percent`, `region_time_seconds` | share of the driving time the cursor spent in each of the 9 regions — previews and dialogs excluded, the completed dwells (10 s each, sum in `region5_dwell_time_removed`) removed from region 5 |
| `driving` | the driving metrics below, per driving phase (`outbound`, `approach`, `return`, `repositioning`) and over all the driving (`all`) |

`driving.<phase>`: `driving_time`, `path_length`, `cumulative_rotation`, `log_dimensionless_jerk`, `stop_time` / `stop_percent` (cursor in region 5, robot still, dwells removed), `region_time_*`, and the metrics of the **command sequence**. A command is a region the cursor stayed in for at least `MIN_COMMAND_S` = 0.25 s (shorter visits are jitter on a border and are merged into the surrounding command); changes never span two driving stretches (a preview or a dialog in between):

| Field | Meaning |
|---|---|
| `n_commands`, `n_command_changes`, `command_changes_per_min` | how much the control changes between regions |
| `mean_command_duration`, `median_command_duration` | how long a motion command (region ≠ 5) is held [s] |
| `non_adjacent_percent` | changes between regions that do not touch (e.g. 3 → 7): the cursor swept across the grid, out of control |
| `n_reversals`, `reversal_percent` | sign changes of the forward/back or left/right command between consecutive motion commands, a stop in between included (2-8, 2-5-8, 1-3, ...): overcorrections (as the steering reversal rate) |
| `stop_passage_percent` | motion → motion changes with a stop (5) in between: stop-and-go (5 2 5 2) vs fluid (2 3 2 1) |
| `sequence_entropy` | conditional entropy H(next \| current) of the command changes [bits] (as the steering entropy): 0 = predictable |
| `command_sequences` | the command sequence of every driving stretch |

The automatic back-up/rotation at the end is not part of any path length.

Next to the JSON, every session also writes `<subject>_run<N>_odometry.csv` (`odometry_file`): the raw mobile base odometry sampled at the control-loop rate (`PUBLISH_HZ` = 20 Hz) while driving — `t` (unix time), `t_test` (s since the test start), `x`, `y` [m], `theta` [rad], `vx`, `vy` [m/s], `vtheta` [rad/s] and the driving `mode` (`max_speed` / `reduced_speed` / `repositioning` / `transport`) — and `<subject>_run<N>_regions.csv` (`regions_file`): every cursor region tick while driving (`t_test`, `stretch`, `mode`, `region`), so any other trajectory or command metric can be computed afterwards.

### Calibration maps — `calibrate_bomi.py` (once), `customize_bomi.py` (per participant)

The same procedure markerlessBoMI was used with: **one** autoencoder map, trained once on a single 90 s recording of the experimenter's hand, and for each participant only the customization (rotation / gain / offset). Neither tool needs a robot connection — just the webcam and MediaPipe model, so they run on the operator PC on their own, before the sessions.

```bash
python3 reachy_bomi/calibrate_bomi.py [NAME] [--cam 0] [--model hand_landmarker.task] [--duration 90] [--recalibrate]   # once
python3 reachy_bomi/customize_bomi.py [SUBJECT] [--base shared] [--cam 0] [--model hand_landmarker.task]                # per participant
```

**Once — `calibrate_bomi.py`** (markerlessBoMI's "Calibration" + "Calculate BoMI map"):
1. *Continuous calibration*: after `SPACE`, the hand is recorded continuously for `CALIB_DURATION_S` = 90 s (every frame with a tracked hand, no sample picking by hand: keep moving the hand through the whole workspace while the time remaining is shown), then the raw 42-feature samples are saved to `calibrations/shared_calib.npy`. If that file already exists the tool offers to reuse it and skip the recording (`--recalibrate` forces a new one) — that's how to retrain after changing the AE hyperparameters in `bomi_teleop.py`.
2. *Offline training*: the autoencoder (32-32-2-32-32, tanh, Adam 0.02, 3001 full-batch epochs) is trained on the saved samples the way markerlessBoMI's `train_ae` does — all-zero rows dropped, shuffle, 80/20 train/test split, screen scale/offset from the training latent codes — and the VAF and latent-variance share of each code unit on train and test are printed (and stored in the `.npz`, shown again at the start of the robot scripts). Takes a couple of minutes on CPU. The map is saved to `calibrations/shared.npz` and previewed live (`Q` quits). A `NAME` other than `shared` saves under that name (then `customize_bomi.py --base NAME`).

**Per participant — `customize_bomi.py SUBJECT`** (markerlessBoMI's "Customization"): loads `calibrations/shared.npz` and shows the live cursor map on the participant's hand; rotate/flip/scale/offset it (`[ ]`, `i`/`o`, `-`/`=`, `hjkl`, `r` resets) until the whole 3×3 grid is comfortably reachable and the rest position sits in region 5, then `S` asks the participant id (`SUBJECT` from the command line is the default, `-` cancels) and saves it as `calibrations/<id>_<YYYYMMDD_HHMMSS>.npz` — type `elisa`, get `elisa_20260916_155827.npz`. `reachy_control.py --subject elisa` loads the **latest** of those, so a redone customization simply wins and the older ones stay on disk. The AE input being MediaPipe's normalized landmark coordinates, one hand's map transfers to another up to this affine customization (hand size/position → gain/offset, orientation → rotation). Retraining the shared map invalidates the existing customized maps (they were made on the old one) — `calibrate_bomi.py` warns about it.

---

## Package layout

```
reachy2_teleop_grasping_and_simulation/
├── reachy_bomi/                     # Python package
│   ├── __init__.py
│   ├── reachy_control.py            # THE entry point: ties bomi_teleop/reachy_detection/reachy_selection/reachy_pregrasp/reachy_grasp together under one BoMI cursor
│   ├── bomi_teleop.py               # library: webcam/MediaPipe → autoencoder cursor → 9-region velocity building blocks, BoMIMap save/load/customize
│   ├── session_metrics.py           # library: session metrics (durations, path lengths, region shares, dwells) + 20 Hz odometry csv, written to results_robot/
│   ├── plot_odometry.py             # standalone script: top-down PNG of the path driven in a session, from its odometry csv
│   ├── calibrate_bomi.py            # standalone script, once: 90 s continuous calibration, offline AE training -> calibrations/shared.npz, live preview -- no robot needed
│   ├── customize_bomi.py            # standalone script, per participant: rotate/flip/scale/offset the shared map live -> calibrations/<SUBJECT>_<date>_<time>.npz -- no robot needed
│   ├── reachy_detection.py          # library: torso camera → YOLOv8 → point cloud → grasp geometry building blocks
│   ├── reachy_selection.py          # library: hover-to-select/confirm UI (dwell-select, repositioning, Yes/No confirm, presentable_filter)
│   ├── reachy_pregrasp.py           # library: pre-grasping-pose goto + wait-with-progress-bar UI
│   ├── reachy_grasp.py              # library: grasp planning/execution (pose math, IK search, execute_grasp)
│   ├── camera_viewer.py             # standalone script: head or torso camera live feed (--camera), spawned by reachy_control.py as its own process
│   ├── stream.py                    # library: camera-streaming primitives (torso + teleop cameras)
│   ├── display.py                   # library: puts every window (cv2 + matplotlib) on the laptop screen
│   ├── graphs.py                    # library: matplotlib diagnostics (point cloud stages, grasp plan), figures saved to graphs/
│   └── safety.py                    # library: quit/shutdown safety net (local check + OS-level global watcher)
├── calibrations/                     # shared_calib.npy (raw samples), shared.npz (the AE map), <SUBJECT>_<date>_<time>.npz (per participant); folder tracked via .gitkeep, the files are not
├── results_robot/                    # session metrics JSON + odometry CSV of reachy_control.py (created on first run, not tracked by git)
├── hand_landmarker.task              # MediaPipe model (tracked, see Requirements)
├── yolov8n.pt                        # YOLOv8 weights (tracked; ultralytics re-downloads them if missing)
├── resource/
│   └── reachy_bomi                  # ament resource marker
├── package.xml
├── setup.py
├── setup.cfg
├── .gitignore
└── README.md
```

## Known limitations
- Requires a functioning webcam on the machine running `reachy_control.py`.
- Grasp planning only handles round objects (`cylinder`/`sphere` shapes in `reachy_detection.SHAPE_BY_CLASS`) — a box's hidden depth can't be recovered from a single camera view the same way.
- The operator's PC and the robot just need network (IP) reachability to each other — no ROS 2 distro matching is required, since communication goes through `reachy2_sdk`'s gRPC interface.
