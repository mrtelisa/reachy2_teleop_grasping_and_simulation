#!/usr/bin/env python3
"""
BoMI client for Reachy2 teleoperation.
Runs on the operator PC, NOT on the robot.

The hand -> cursor chain (MediaPipe, autoencoder map, filter) and the
calibrations/ helpers are the ones of bomi.py, shared with the other tools.

Dependencies:
    pip install mediapipe opencv-python tensorflow numpy scipy
    (optional: `sudo apt install wmctrl` to keep the cursor map above the browser)

Usage:
    # Beforehand, without the robot: calibrate_bomi.py once (shared autoencoder
    # map), then customize_bomi.py SUBJECT per participant.
    python3 socket_client.py <server_ip> --subject S001

    <server_ip> is optional; if omitted, HOST (set in this file) is used.

    Options:
        --subject ID           Participant id (or map name): loads
                               calibrations/<ID>.npz if it exists, else the latest
                               calibrations/<ID>_<date>_<time>.npz saved by
                               customize_bomi.py. Default: S000
        --calib NAME           Load calibrations/<NAME>.npz instead (e.g. "shared"
                               to try the uncustomized map).
        --model PATH           Path to the MediaPipe hand_landmarker.task model.
                               Default: scripts/hand_landmarker.task inside the package.
        --port PORT            Robot socket port. Default: 5051
        --cam INDEX            Webcam index. Default: 0
        --scenario NAME        Scenario name to send to the robot right after connecting
                               (e.g. "familiarization"). If omitted, no scenario message
                               is sent and --start-rviz/--record/--sim-wait are ignored.
        --start-rviz true|false  Whether the robot side should start RViz for this
                               scenario. Default: false
        --record true|false    Whether the robot side should record a ROS 2 bag for
                               this scenario. Default: true
        --sim-wait SECONDS     Seconds to wait after sending the scenario, to give the
                               robot side time to bring up the simulation before the
                               control loop starts sending velocities. Default: 10.0
        --sim-url URL          Page opened in the default browser right after the
                               scenario is sent (the noVNC view of Gazebo). Pass ""
                               to open nothing. Default: the noVNC URL on localhost.
        --show-cam             Also show the webcam feed with the tracked hand, in a
                               window on the experimenter's monitor (bomi.open_camera_window).

Phase 1 - Map loading:
    No calibration happens here: the participant's map is loaded from
    calibrations/ (fails fast if missing).

Phase 2 - Cursor preview:
    Same cursor map as Control, but nothing is sent to the robot. Hold the
    cursor in the centre region (5) for 5 s to start Control.

Phase 3 - Control:
    Hand movement -> autoencoder cursor -> 9-region velocity -> TCP socket to robot.
    Shows a small map of the virtual screen with the 9-region grid lines and a
    dot at the current cursor position, pinned to the top-left corner and kept
    above the other windows (e.g. the browser with the simulation).
    Q, ESC, or closing the window with the X = quit and stop robot.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser

import cv2
import numpy as np

import bomi
import display

MAX_LINEAR = 1.0      # m/s
MAX_ANGULAR = 0.8     # rad/s
DEAD_ZONE_PX = 200    # pixel radius around screen center before motion starts

SEND_HZ = 20 # frequency of sending lin/ang velocities to the second computer (Hz)

FORMAT = "utf-8"
DISCONNECT_MESSAGE = "!DISCONNECT"

# Fallback server IP -- replace with the robot PC's actual IP on your network.
DEFAULT_HOST = "192.168.1.100"

MAP_WINDOW_NAME = bomi.MAP_WINDOW_NAME
# Position (top-left corner, relative to the display.py screen) the cursor map
# is pinned to, so it sits in a corner above the browser showing the simulation.
MAP_WINDOW_POS = (0, 0)

# Cursor preview: seconds the cursor must stay in the centre region (5) before
# the control phase starts sending velocities to the robot.
PREVIEW_HOLD_SECONDS = 5.0

# Browser page showing the simulation (noVNC served by the Reachy container),
# opened right after the scenario request is sent.
DEFAULT_SIM_URL = "http://localhost:6080/vnc.html?autoconnect=1&resize=remote"


# --- Velocity helpers (adapted from reaching_functions.py) ---------
def compute_dynamic_vel_from_cursor(
    crs_x: float,
    crs_y: float,
    max_linear: float = MAX_LINEAR,
    max_angular: float = MAX_ANGULAR,
    dead_zone_px: float = DEAD_ZONE_PX,
    ang_right_is_negative: bool = True,
) -> tuple:
    """
    Continuous linear/angular velocity from cursor position.
    Cursor at screen center -> zero velocity (dead zone).
    Up from center -> positive linear; right from center -> negative angular.
    """
    cx = bomi.BASE_WIDTH / 2.0
    cy = bomi.BASE_HEIGHT / 2.0
    dx = crs_x - cx
    dy = crs_y - cy

    if np.hypot(dx, dy) < dead_zone_px:
        return 0.0, 0.0

    x_norm = float(np.clip(dx / cx, -1.0, 1.0))
    y_norm = float(np.clip(-dy / cy, -1.0, 1.0))  # up = positive

    if abs(x_norm) < dead_zone_px / cx:
        x_norm = 0.0
    if abs(y_norm) < dead_zone_px / cy:
        y_norm = 0.0

    lin_vel = max_linear * y_norm
    ang_sign = -1.0 if ang_right_is_negative else 1.0
    ang_vel = ang_sign * max_angular * x_norm
    return lin_vel, ang_vel


def apply_region_velocity_mask(region: int, lin_vel: float, ang_vel: float) -> tuple:
    """
    Enforce active DOFs per region:
      center (5)        -> stop
      middle col (2, 8) -> linear only
      middle row (4, 6) -> angular only
      corners (1,3,7,9) -> both
    """
    if region == 5:
        return 0.0, 0.0
    if region in (2, 8):
        ang_vel = 0.0
    if region in (4, 6):
        lin_vel = 0.0
    return lin_vel, ang_vel


_last_raise_time = 0.0
_RAISE_WINDOW_INTERVAL_S = 1.0
_wmctrl_missing_warned = False


def _pin_map_window(window_name: str) -> None:
    """Keep the cursor map in its corner and above the other windows (the
    browser showing the simulation). Safe to call every frame.

    cv2's WND_PROP_TOPMOST only wins over windows of the *same process*, so the
    window is also re-activated via wmctrl (same as alt-tabbing to it), throttled
    to once per second. No-ops (after one warning) if wmctrl isn't installed."""
    global _last_raise_time, _wmctrl_missing_warned
    display.move(window_name, *MAP_WINDOW_POS)  # a just-closed window can make the WM reclaim the position otherwise
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    now = time.time()
    if now - _last_raise_time < _RAISE_WINDOW_INTERVAL_S:
        return
    _last_raise_time = now
    try:
        subprocess.run(["wmctrl", "-a", window_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        if not _wmctrl_missing_warned:
            print("[WARN] wmctrl not installed -- the cursor map may not stay in front "
                  "(sudo apt install wmctrl).")
            _wmctrl_missing_warned = True


class RobotSocket:
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        print(f"[SOCKET] Connected to {host}:{port}")

    def send(self, msg: str) -> None:
        # '\n' delimiter so socket_server.py can tell separate messages apart
        # even if TCP coalesces/splits them across recv() calls.
        self._sock.sendall((msg + "\n").encode(FORMAT))

    def close(self) -> None:
        try:
            self.send(DISCONNECT_MESSAGE)
        except OSError:
            pass
        self._sock.close()


# --- Phases ---
def _cursor_preview_phase(cap, landmarker, bomi_map: bomi.BoMIMap, cursor_filter: bomi.CursorFilter,
                          show_cam: bool = False, hold_seconds: float = PREVIEW_HOLD_SECONDS) -> tuple:
    """
    Shows the same cursor/region view as the control phase, but nothing is
    sent to the robot -- lets the user get a feel for the cursor before it
    starts driving anything. Ends once the cursor has been held in the centre
    region (5) for hold_seconds; returns the final (crs_x, crs_y) so the
    control phase can continue from the same cursor state.
    """
    map_window = MAP_WINDOW_NAME

    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0
    region = bomi.check_region_cursor(crs_x, crs_y)
    center_hold_start = None

    print("\n=== CURSOR PREVIEW (robot not moving) ===")
    print(f"Get a feel for the cursor. Hold it centered (region 5) for {hold_seconds:.0f}s "
          "to start Control   |   Q = quit")

    while True:
        frame, crs_x, crs_y, hand_detected = bomi.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if frame is None:
            continue
        if show_cam:
            bomi.show_camera(frame, hand_detected)
        if hand_detected:
            region = bomi.check_region_cursor(crs_x, crs_y)

        now = time.time()
        # Only accrue while actively tracked and centered, so a dropped hand
        # while the stale cursor happens to sit in region 5 can't silently
        # trigger the switch.
        center_hold_start = (center_hold_start or now) if (hand_detected and region == 5) else None
        center_progress = min((now - center_hold_start) / hold_seconds, 1.0) if center_hold_start else 0.0

        cv2.imshow(map_window, bomi._draw_cursor_map(
            crs_x, crs_y, region, f"PREVIEW (not sent) - hold centered: {center_progress * 100:.0f}%",
        ))
        _pin_map_window(map_window)

        key = cv2.waitKey(1) & 0xFF
        if center_progress >= 1.0:
            break
        if bomi._quit_requested(key, map_window):
            print("Aborted.")
            sys.exit(0)

    # The map window is intentionally left open so it carries straight into
    # the control phase instead of flickering shut.
    return crs_x, crs_y


def _control_phase(cap, landmarker, bomi_map: bomi.BoMIMap, robot: RobotSocket,
                   show_cam: bool = False, cursor_filter: bomi.CursorFilter = None,
                   crs_x: float = None, crs_y: float = None) -> None:
    """show_cam=False (default) only shows the cursor map, pinned in its
    corner; True also shows the webcam feed with the tracked hand.
    Pass the preview phase's cursor_filter/crs_x/crs_y to continue from its
    cursor state instead of restarting centered."""
    dt = 1.0 / SEND_HZ
    last_send = time.time()
    cursor_filter = cursor_filter or bomi.CursorFilter()
    map_window = MAP_WINDOW_NAME

    # Start centered (region 5) until the first hand detection updates it.
    if crs_x is None or crs_y is None:
        crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0
    region = bomi.check_region_cursor(crs_x, crs_y)
    message = "lin_vel:0.000 ang_vel:0.000"

    print("\n=== CONTROL ===  Q = quit")
    robot.send("nine region")

    while True:
        frame, crs_x, crs_y, hand_detected = bomi.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        if frame is None:
            continue
        if show_cam:
            bomi.show_camera(frame, hand_detected)

        lin_vel, ang_vel = 0.0, 0.0
        if hand_detected:
            region = bomi.check_region_cursor(crs_x, crs_y)
            lin_vel, ang_vel = compute_dynamic_vel_from_cursor(crs_x, crs_y)
            lin_vel, ang_vel = apply_region_velocity_mask(region, lin_vel, ang_vel)

        now = time.time()
        if now - last_send >= dt:
            message = f"lin_vel:{lin_vel:.3f} ang_vel:{ang_vel:.3f}"
            robot.send(message)
            last_send = now

        cv2.imshow(map_window, bomi._draw_cursor_map(crs_x, crs_y, region, f"-> PC2: {message}"))
        _pin_map_window(map_window)

        key = cv2.waitKey(1) & 0xFF
        if bomi._quit_requested(key, map_window):
            break

    robot.send("lin_vel:0.000 ang_vel:0.000")
    cv2.destroyWindow(map_window)


# --- Entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(description="BoMI client for Reachy2")
    parser.add_argument("server_ip", nargs="?", default=DEFAULT_HOST,
                        help=f"IP address of the Reachy robot (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--subject", default="S000",
                        help="Participant id or map name: calibrations/<subject>.npz if it exists, else the "
                             "latest calibrations/<subject>_<date>_<time>.npz from customize_bomi.py (default: S000)")
    parser.add_argument("--calib", default=None,
                        help="Map to load (calibrations/<NAME>.npz) instead of the participant's "
                             "map resolved from --subject")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model "
                             f"(default: {bomi.DEFAULT_MODEL_PATH}).")
    parser.add_argument("--scenario", default=None,
                        help="Scenario name to send to the robot right after connecting "
                             "(e.g. 'familiarization'). Omit to skip sending a scenario.")
    parser.add_argument("--start-rviz", choices=["true", "false"], default="false",
                        help="Whether the robot side should start RViz for this scenario "
                             "(default: true). Only used if --scenario is set.")
    parser.add_argument("--record", choices=["true", "false"], default="true",
                        help="Whether the robot side should record a ROS 2 bag for this "
                             "scenario (default: true). Only used if --scenario is set.")
    parser.add_argument("--sim-wait", type=float, default=10.0,
                        help="Seconds to wait after sending the scenario before starting "
                             "the control loop, to let the robot side bring up the "
                             "simulation (default: 10.0).")
    parser.add_argument("--sim-url", default=DEFAULT_SIM_URL,
                        help="Page opened in the default browser right after the scenario "
                             "is sent (noVNC view of the simulation). Pass \"\" to open "
                             f"nothing (default: {DEFAULT_SIM_URL}).")
    parser.add_argument("--show-cam", action="store_true",
                        help="Also show the webcam feed with the tracked hand, on the experimenter's "
                             "monitor (default: only the cursor map is shown).")
    args = parser.parse_args()

    # The map is the participant's customization of the shared autoencoder
    # (calibrate_bomi.py once, customize_bomi.py per participant), made
    # beforehand: no calibration/training happens here
    if args.calib:
        calib_path = bomi._resolve_calib_path(args.calib)
    else:
        calib_path = bomi._resolve_subject_map_path(args.subject)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No calibration map for '{args.subject}' found (neither {os.path.basename(calib_path)} "
              f"nor {bomi._strip_npz(args.subject)}_<date>_<time>.npz): run customize_bomi.py {args.subject} first.")
        available = bomi._list_saved_maps()
        if available:
            print("        Available: " + ", ".join(available))
        sys.exit(1)

    # Fail early if the hand-landmarker model is missing
    if not os.path.exists(args.model):
        print(f"[ERROR] MediaPipe model not found: '{args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

    robot = RobotSocket(args.server_ip, args.port)
    cap = None
    landmarker = None
    try:
        if args.scenario:
            scenario_msg = f"scenario:{args.scenario} rviz:{args.start_rviz} record:{args.record}"
            print(f"[SCENARIO] Sending '{scenario_msg}' to the robot")
            robot.send(scenario_msg)
            if args.sim_url:
                # Open the simulation view right away, so it is on screen by the
                # time the cursor map appears on top of it.
                print(f"[SCENARIO] Opening {args.sim_url} in the browser")
                webbrowser.open_new_tab(args.sim_url)
            print(f"[SCENARIO] Waiting {args.sim_wait:.0f}s for the simulation to start...")
            time.sleep(args.sim_wait)

        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {args.cam}")
            sys.exit(1)

        landmarker = bomi.create_hand_landmarker(args.model)

        bomi_map = bomi.BoMIMap()
        bomi_map.load(calib_path)
        print(f"Loaded calibration map from {calib_path}")
        bomi_map.print_metrics()

        if args.show_cam:
            bomi.open_camera_window(cap)   # on the monitor, before the map (which keeps the focus)
        cursor_filter = bomi.CursorFilter()
        crs_x, crs_y = _cursor_preview_phase(cap, landmarker, bomi_map, cursor_filter,
                                             show_cam=args.show_cam)
        _control_phase(cap, landmarker, bomi_map, robot, show_cam=args.show_cam,
                       cursor_filter=cursor_filter, crs_x=crs_x, crs_y=crs_y)
    finally:
        robot.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()