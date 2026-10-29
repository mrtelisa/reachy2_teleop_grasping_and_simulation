# Reachy 2 BoMI Teleoperation: Experimental Protocol

This branch contains the files used during the experimental protocol of the project. The protocol had several phases. The exercises run from this branch, all based on the markerless Body-Machine Interface (BoMI), are:

- **Blind test:** reaching targets without visual feedback of the cursor
- **Reaching training:** reaching targets in the 9 regions of the control interface
- **Free exploration in simulation:** driving the Reachy 2 mobile base in Gazebo

The teleoperation and grasping on the real robot are in the [`main`](../../tree/main) branch.

## Method

### Body-Machine Interface

Same interface as in `main`: 21 MediaPipe hand landmarks (42 features) → autoencoder (42 → 32 → 32 → **2** → 32 → 32 → 42) → 2D cursor. Before the exercises, the shared map is customized for each participant (rotation, flip, gain, offset). The screen is split into the same 3×3 regions used to command the robot.

### Blind test

- The cursor is **never shown**: only the current target is on screen (centres of regions 2, 4, 5 and 9).
- 12 trials of 4 s each (every target 3 times), in a fixed order (`config/blind_targets.csv`).
- Each test is compared with the previous tests of the same participant, to track how the map is learned.

### Reaching training

- Region 5 is the *home*, and each outer region has a target circle at its centre.
- 96 targets (8 outer regions × 12), alternating home → target → home, in a fixed order (`config/cursor_regions.csv`).
- For the first **1 s** after a target appears the cursor is **hidden**, then it is shown again.
- A goal is reached after a **0.5 s dwell** inside its circle.
- Two sessions, `pre` and `post`, which are compared automatically.

### Free exploration in simulation

The participant drives the simulated Reachy 2 mobile base in a room with obstacles (`worlds/familiarization.world`), with the same 9-region velocity map as the real teleoperation. The cursor runs on the operator PC and sends velocities over TCP to a ROS 2 node, which publishes them on `/cmd_vel`.

This exercise requires Pollen Robotics' **Reachy 2 Docker container** (ROS 2 + Gazebo), with this package built in its workspace.

### Metrics

Computed for every trial (`reaching_metrics.py`):

- **Kinematics:** reach, reaction and movement time, normalized path length, maximum deviation, smoothness (log dimensionless jerk), speed peaks, initial direction error.
- **Map learning:** target reached while the cursor was hidden, correct region when the cursor reappears, wrong regions visited.
- **Blind test:** hit rate, final and minimum error from the target, time in the target region.

Results are saved as per-trial, per-block and trajectory CSVs, plus a JSON summary. The simulation is recorded as a rosbag.

## Repository structure

```
reachy_bomi/
├── bomi.py                # hand tracking, autoencoder map, cursor filter
├── calibrate_bomi.py      # calibration recording + autoencoder training
├── customize_bomi.py      # per-participant customization of the map
├── load_bomi.py           # try a saved map
├── reaching_blind.py      # blind test
├── reaching_regions.py    # reaching training (pre/post)
├── reaching_metrics.py    # per-trial metrics
├── socket_client.py       # simulation, operator PC: cursor → velocities over TCP
├── socket_server.py       # simulation, container: TCP → ROS 2, launches the scenario
├── cmd_vel_publisher.py   # simulation, container: velocities → /cmd_vel
├── scenarios.py           # reads the simulation scenarios
├── display.py             # window placement
launch/                    # ROS 2 launch files
worlds/                    # Gazebo world
config/                    # fixed target sequences, simulation scenarios
scripts/                   # MediaPipe model, container setup
```

## Usage

Python packages: `pip install -r requirements.txt`. All scripts are run from `reachy_bomi/`.

```bash
# Per-participant customization of the shared map
python3 customize_bomi.py shared

# Blind test (N = test number)
python3 reaching_blind.py --test N --subject SXXX

# Reaching training
python3 reaching_regions.py --phase pre --subject SXXX
python3 reaching_regions.py --phase post --subject SXXX
```

Free exploration in simulation:

```bash
# Inside the Reachy 2 container
ros2 launch reachy_bomi bomi_bridge.launch.py

# On the operator PC
python3 socket_client.py <container_ip> --port 5051 --scenario familiarization --subject SXXX
```
