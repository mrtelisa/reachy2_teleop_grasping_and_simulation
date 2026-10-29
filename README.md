# Reachy 2 BoMI Teleoperation

Markerless Body-Machine Interface (BoMI) to teleoperate the mobile base of the **Reachy 2** humanoid (Pollen Robotics) with hand movements, and to select objects that the robot then grasps and places autonomously.

The operator's hand is tracked by a webcam with MediaPipe. An autoencoder maps the hand landmarks to a 2D cursor, and the cursor position is converted into base velocity commands or into object selection. Grasping relies on YOLOv8 detection and the torso depth camera. All communication with the robot goes through `reachy2_sdk` (gRPC), so the operator PC does not need ROS 2.

## Branches

| Branch | Content |
|---|---|
| `main` | Teleoperation and grasping on the real Reachy 2 (this page) |
| [`new_protocol`](../../tree/new_protocol) | Experimental protocol to train and test participants on the BoMI before teleoperation: reaching tasks on screen and free driving of Reachy 2 in simulation (Gazebo) |

## Method

### 1. Body-Machine Interface

- **Input:** 21 MediaPipe hand landmarks (x, y), i.e. 42 features per frame.
- **Calibration (once):** 90 s of continuous free hand motion recorded from one person.
- **Map:** autoencoder 42 → 32 → 32 → **2** → 32 → 32 → 42 (tanh, Adam, lr 0.02, 3001 full-batch epochs, 80/20 train/test split). The 2D latent space is the cursor. VAF and latent variance are reported on the train and test sets.
- **Customization (per user):** rotation, flip, gain and offset applied to the shared map, so each user can reach the whole workspace comfortably.

### 2. Navigation

The cursor workspace is split into a 3×3 grid, and each region maps to a base command:

```
 1 | 2 | 3      5 (center)          → stop
 4 | 5 | 6      2, 8                → linear velocity only
 7 | 8 | 9      4, 6                → angular velocity only
                1, 3, 7, 9          → linear + angular
```

Holding the cursor in region 5 (dwell) switches between phases: full-speed driving → pre-grasp posture → reduced-speed approach → object selection. A low-speed repositioning mode, which streams the torso camera, is available from the selection screen.

### 3. Grasping

1. Torso camera RGB-D frame → YOLOv8 detection, pre-filtered by reachability and gripper size.
2. The user selects and confirms an object with the BoMI cursor.
3. Depth crop → object point cloud (Open3D) → position, principal axes and size.
4. Pre-grasp / grasp / lift end-effector poses → IK → arm execution.
5. Placement: the destination table plane is estimated with RANSAC, and the release pose is corrected for its height.

### 4. Evaluation metrics

Each session saves a JSON file with metrics per task phase (outbound, approach, grasp, return, placement): durations, path length, cumulative rotation, smoothness (log dimensionless jerk), time per grid region, and command-sequence metrics (reversal rate, stop-and-go, conditional entropy). Raw 20 Hz odometry and cursor regions are saved as CSV.

## Repository structure

```
reachy_bomi/
├── reachy_control.py     # main entry point: full teleoperation + grasping session
├── calibrate_bomi.py     # calibration recording + autoencoder training (run once)
├── customize_bomi.py     # per-user customization of the map
├── bomi_teleop.py        # hand tracking, autoencoder map, cursor → velocity
├── reachy_detection.py   # YOLOv8 detection, point cloud, object geometry
├── reachy_selection.py   # cursor-based object selection and confirmation UI
├── reachy_pregrasp.py    # pre-grasp arm posture
├── reachy_grasp.py       # grasp/place planning and execution
├── session_metrics.py    # session metrics and logs
├── plot_odometry.py      # plots the driven path from a session log
├── safety.py             # emergency stop (ESC/Q from any window)
├── camera_viewer.py      # head/torso camera feed in a separate process
├── stream.py, display.py, graphs.py   # camera streaming, window placement, debug plots
calibrations/             # saved maps (not tracked)
hand_landmarker.task      # MediaPipe hand model
yolov8n.pt                # YOLOv8 weights (COCO)
```

## Requirements

- Reachy 2 reachable over the network, with its SDK server running
- Webcam on the operator PC
- Python packages: `pip install -r requirements.txt`
  (reachy2-sdk, mediapipe, tensorflow, ultralytics, open3d, opencv-python, numpy, scipy, matplotlib, pynput)

ROS 2 is optional: the folder is also an `ament_python` package (`ros2 run reachy_bomi reachy_control`).

## Usage

```bash
# 1. Once: record 90 s of hand motion and train the shared map (no robot needed)
python3 reachy_bomi/calibrate_bomi.py

# 2. Per user: customize the shared map, saved as calibrations/<USER>_<date>_<time>.npz
python3 reachy_bomi/customize_bomi.py USER

# 3. Run a session on the robot (loads the latest map of USER)
python3 reachy_bomi/reachy_control.py <robot_ip> --subject USER --run 1
```

Main options of `reachy_control.py`: `--cam` (webcam index), `--calib` (load a specific map), `--yolo-model`, `--conf` (detection threshold), `--run` (1 or 2, session index).

Results are saved in `results_robot/` as `<USER>_run<N>_session.json`, `_odometry.csv` and `_regions.csv`.

## Limitations

- Grasp planning supports round objects only (cylinders, spheres).
- Single-view depth: object dimensions are estimated from one camera viewpoint.

## Acknowledgements

The project started from a ROS 1 implementation for navigating the TIAGo robot in simulation, which was then ported to ROS 2 and later to the Reachy 2 SDK.
