#!/usr/bin/env python3
"""
Navigation training for the reachy_control.py test: the same cursor preview
(9-region map) and head-camera stream, then BoMI driving of the mobile base
at the normal (max) speed. No arms, no object selection, no grasp.

Everything the participant sees and feels is the one of the test, taken from
reachy_control.py / bomi_teleop.py: same hand -> cursor chain, 9-region
velocity map, MAX_LINEAR / MAX_ANGULAR, PUBLISH_HZ, dwells, lidar
distances, startup posture and gaze.

Usage:
    python3 reachy_training.py [robot_ip] [--cam 0] [--model PATH] [--calib NAME] [--subject ID]

Flow:
  1. Load the participant's map (customize_bomi.py, run beforehand without the
     robot; --calib overrides), robot on, mobile base ODOMETRY RESET.
  2. Cursor preview (head camera streaming): hold the cursor in region 5 for
     3 s to start driving.
  3. Driving: cursor -> 9-region velocity -> mobile base. A 10 s dwell in
     region 5 opens the same Yes/No dialog as the test: Yes ends the
     training, No goes back to driving through a cursor preview.
Q / ESC = quit (robot stopped and powered off, metrics saved anyway).

Metrics: the same session_metrics as reachy_control.py, written to
results_training/<subject>_session.json, plus the 20 Hz odometry log
results_training/<subject>_odometry.csv.
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

# XWayland: cv2's fullscreen/topmost hints only work there (inherited by camera_viewer.py)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from reachy2_sdk import ReachySDK

import bomi_teleop
import reachy_control   # same windows, camera viewer, dwell and startup parameters as the test
import reachy_selection
import safety
import session_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_training")

PREVIEW_HOLD_SECONDS = reachy_control.SELECTION_HOLD_SECONDS   # 3 s, cursor preview -> driving (as in the test)
DWELL_SECONDS = reachy_control.MODE_SWITCH_HOLD_SECONDS        # 10 s, driving dwell -> end dialog (the test's Control dwell)
END_PROMPT = ["Do you want to end the training?"]

_metrics: Optional[session_metrics.SessionMetrics] = None   # session metrics, created in main()


def _sample_odometry(mobile_base) -> None:
    """Feed the base odometry to the session metrics (always at max speed here).
    A failed read (base off, gRPC hiccup) is reported and skipped, never fatal."""
    if _metrics is None:
        return
    try:
        _metrics.sample(mobile_base.get_current_odometry(degrees=False), session_metrics.MODE_MAX)
    except Exception as exc:
        print(f"[metrics] odometry read failed: {exc}")


def training_navigation(cap, landmarker, bomi_map, mobile_base, cursor_filter, crs_x, crs_y) -> None:
    """Driving loop of the training: cursor -> 9-region velocities -> mobile
    base at MAX_LINEAR / MAX_ANGULAR (reachy_control's Control before the
    pre-grasp pose). Holding the cursor in region 5 for DWELL_SECONDS opens
    the Yes/No dialog: Yes ends the training, No goes back to driving through
    a cursor preview. Q / ESC quit."""
    dt = 1.0 / bomi_teleop.PUBLISH_HZ
    last_publish = time.time()
    map_window = bomi_teleop.MAP_WINDOW_NAME

    region = bomi_teleop.check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"
    center_hold_start = None

    def _hold_base_still() -> None:
        mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
        mobile_base.send_speed_command()

    print("\n=== TRAINING: DRIVING ===  Q = quit  |  hold the cursor centered (region 5) "
          f"for {DWELL_SECONDS:.0f}s to end the training")
    if _metrics is not None:
        _metrics.start_test()   # the base starts moving after the cursor preview: training starts here

    while True:
        # A quit watcher may already be shutting down on another thread: stop
        # publishing speed commands
        if safety.shutdown_started():
            return

        hand_frame, crs_x, crs_y, hand_detected = bomi_teleop.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if hand_frame is None:
            continue

        if hand_detected:
            region = bomi_teleop.check_region_cursor(crs_x, crs_y)
            lin_vel, ang_vel = bomi_teleop.compute_dynamic_vel_from_cursor(
                crs_x, crs_y, max_linear=bomi_teleop.MAX_LINEAR, max_angular=bomi_teleop.MAX_ANGULAR,
            )
            lin_vel, ang_vel = bomi_teleop.apply_region_velocity_mask(region, lin_vel, ang_vel)
        else:
            lin_vel, ang_vel = 0.0, 0.0

        now = time.time()
        center_hold_start = (center_hold_start or now) if (hand_detected and region == 5) else None
        center_progress = min((now - center_hold_start) / DWELL_SECONDS, 1.0) if center_hold_start else 0.0
        if _metrics is not None:
            _metrics.region_tick(region, now)

        cv2.imshow(map_window, bomi_teleop.draw_cursor_map(crs_x, crs_y, region, message))
        cv2.moveWindow(map_window, *bomi_teleop.MAP_WINDOW_POS)  # re-pin, the WM can move it
        cv2.setWindowProperty(map_window, cv2.WND_PROP_TOPMOST, 1)  # re-pin (same-process windows only)
        safety.raise_window(map_window)  # actually wins over the cross-process fullscreen camera_viewer window

        if center_progress >= 1.0:
            decision, crs_x, crs_y = reachy_selection.confirm_bomi(
                cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
                lines=END_PROMPT, on_frame=_hold_base_still,
            )
            if _metrics is not None:
                _metrics.dwell(decision)
            if decision is None:
                break
            center_hold_start = None
            if decision:
                if _metrics is not None:
                    _metrics.end_test("finished")
                print("\nTraining over.")
                break
            # No: back to driving, through a cursor preview
            cursor_filter.reset(crs_x, crs_y)
            crs_x, crs_y = bomi_teleop.cursor_preview_phase(
                cap, landmarker, bomi_map, cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y, show_cam=False,
                hold_seconds=PREVIEW_HOLD_SECONDS,
            )
            continue

        if now - last_publish >= dt:
            message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
            # vtheta is in deg/s for reachy2_sdk
            mobile_base.set_goal_speed(vx=lin_vel, vy=0, vtheta=math.degrees(ang_vel))
            mobile_base.send_speed_command()
            last_publish = now
            _sample_odometry(mobile_base)

        key = cv2.waitKey(1) & 0xFF
        if safety.quit_requested(key, map_window):
            break

    # During a shutdown the watcher thread already zeroed the speed
    if not safety.shutdown_started():
        mobile_base.set_goal_speed(vx=0, vy=0, vtheta=0)
        mobile_base.send_speed_command()
        mobile_base.turn_off()
    safety.destroy_window(map_window)
    reachy_control.stop_camera_viewer()


# --- Entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Navigation training for the reachy_control.py test: cursor preview + "
                    "head camera, then BoMI driving of the mobile base at normal speed."
    )
    parser.add_argument("robot_ip", nargs="?", default=reachy_control.DEFAULT_ROBOT_IP,
                        help=f"IP address of the Reachy robot (default: {reachy_control.DEFAULT_ROBOT_IP})")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi_teleop.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model "
                             f"(default: {bomi_teleop.DEFAULT_MODEL_PATH}).")
    parser.add_argument("--calib", default=None,
                        help="Map to load (calibrations/<NAME>.npz) instead of the participant's "
                             "map resolved from --subject")
    parser.add_argument("--subject", default="S000",
                        help="Participant id: loads calibrations/<subject>.npz if it exists, else the latest "
                             "calibrations/<subject>_<date>_<time>.npz from customize_bomi.py; also names the "
                             "session metrics files in results_training/ (default: S000)")
    cli_args = parser.parse_args()
    cli_args.subject = bomi_teleop.strip_npz(cli_args.subject)  # "elisa.npz" -> "elisa" in the metrics file names

    global _metrics
    _metrics = session_metrics.SessionMetrics(cli_args.subject, dwell_seconds=DWELL_SECONDS, results_dir=RESULTS_DIR)

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

    # The map is the participant's customization of the shared autoencoder
    # (calibrate_bomi.py once, customize_bomi.py per participant), made
    # beforehand without the robot: no calibration/training happens here
    if cli_args.calib:
        calib_path = bomi_teleop.resolve_calib_path(cli_args.calib)
    else:
        calib_path = bomi_teleop.resolve_subject_map_path(cli_args.subject)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No calibration map for '{cli_args.subject}' found (neither {os.path.basename(calib_path)} "
              f"nor {bomi_teleop.strip_npz(cli_args.subject)}_<date>_<time>.npz): "
              f"run customize_bomi.py {cli_args.subject} first.")
        available = bomi_teleop.list_saved_maps()
        if available:
            print("        Available: " + ", ".join(available))
        sys.exit(1)

    reachy = ReachySDK(host=cli_args.robot_ip)
    if reachy.mobile_base is None:
        print(f"[ERROR] No mobile base reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)
    if reachy.cameras is None or reachy.cameras.teleop is None:
        print(f"[ERROR] No head/teleop camera reported by the robot at '{cli_args.robot_ip}'")
        reachy.disconnect()
        sys.exit(1)

    mobile_base = reachy.mobile_base

    # Same startup as the test (reachy_control.main)
    reachy.turn_on()
    reachy.goto_posture("default", duration=3.0, wait=True)
    mobile_base.lidar.safety_enabled = True
    mobile_base.lidar.safety_slowdown_distance = bomi_teleop.LIDAR_SLOWDOWN_DISTANCE
    mobile_base.lidar.safety_critical_distance = bomi_teleop.LIDAR_CRITICAL_DISTANCE
    mobile_base.turn_on()
    # Odometry from the starting pose: x, y, theta = 0 here, so the logged
    # trajectory of every session is in the same frame
    mobile_base.reset_odometry()
    print("[odometry] mobile base odometry reset")

    def _on_emergency_quit() -> None:
        reachy_control.stop_camera_viewer()
        safety.emergency_shutdown(reachy, mobile_base, rotate_base_before_shutdown=False)

    safety.start_global_quit_watcher(_on_emergency_quit)
    stop_terminal_watcher = safety.start_terminal_quit_watcher(_on_emergency_quit)

    cap = None
    landmarker = None
    try:
        cap = cv2.VideoCapture(cli_args.cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cli_args.cam}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame, not a growing backlog

        landmarker_options = hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=cli_args.model),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = hand_landmarker.HandLandmarker.create_from_options(landmarker_options)

        bomi_map = bomi_teleop.BoMIMap()
        bomi_map.load_map_bomi(calib_path)
        print(f"Loaded calibration map from {calib_path}")
        bomi_map.print_metrics()
        reachy_control.bring_window_to_front(bomi_teleop.MAP_WINDOW_NAME, bomi_teleop.MAP_WINDOW_POS)
        reachy_control.start_camera_viewer(cli_args.robot_ip)   # head camera
        reachy.head.rotate_by(pitch=-reachy_control.STARTUP_GAZE_PITCH_DEG, yaw=0, roll=0, wait=False)  # look down

        cursor_filter = bomi_teleop.CursorFilter()
        crs_x, crs_y = bomi_teleop.cursor_preview_phase(
            cap, landmarker, bomi_map, cursor_filter=cursor_filter, show_cam=False,
            hold_seconds=PREVIEW_HOLD_SECONDS,
        )
        training_navigation(cap, landmarker, bomi_map, mobile_base, cursor_filter, crs_x, crs_y)
    finally:
        reachy_control.stop_camera_viewer()
        if stop_terminal_watcher is not None:
            stop_terminal_watcher()
        # Session metrics: a run that did not end with "Yes" to the end dialog
        # is closed here as a quit; the files are written whatever happened
        _metrics.end_test("quit")
        summary = _metrics.save()
        print(f"[metrics] training {summary['test_duration'] or 0:.0f}s, "
              f"path {summary['path_length_total']:.2f} m, {summary['n_dwell']} dwells "
              f"({summary['n_dwell_declined']} declined)")
        safety.safe_robot_shutdown(reachy, mobile_base, rotate_base_before_shutdown=False)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()
        reachy.disconnect()


if __name__ == "__main__":
    main()
