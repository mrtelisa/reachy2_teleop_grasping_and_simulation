#!/usr/bin/env python3
"""
Step 2, once per participant (no robot needed): markerlessBoMI's
"Customization". Loads the shared autoencoder map (calibrations/shared.npz,
from calibrate_bomi.py), lets you rotate / flip / scale / offset it live on the
participant's hand and saves it as calibrations/<SUBJECT>_<YYYYMMDD_HHMMSS>.npz
-- reachy_control.py, given --subject SUBJECT, loads the
latest of those. The shared map is untouched.

Keys: [ / ] rotate, i / o flip X / Y, - / = scale, h j k l offset, r reset,
      s save (asks the participant id, default SUBJECT; '-' cancels), q quit

    python3 customize_bomi.py [SUBJECT] [--base NAME] [--cam INDEX] [--model PATH]
"""

import argparse
import os
import sys

# XWayland: cv2's fullscreen hint only works there (as in reachy_control.py)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

import bomi_teleop
import safety

ROT_STEP_DEG = 5.0
SCALE_STEP = 1.1     # multiplicative
OFFSET_STEP_PX = 10.0

HELP_TEXT = (
    "[ ]=rotate  i/o=flip X/Y  -/+=scale  hjkl=offset  r=reset  s=save  q=quit"
)


def _prompt_and_save(bomi_map: bomi_teleop.BoMIMap, subject: str) -> bool:
    """Asks the participant id (default: subject, if given on the command line)
    and saves the map as calibrations/<id>_<YYYYMMDD_HHMMSS>.npz, the file
    "--subject <id>" resolves to everywhere. Returns True once saved (False if
    the user cancels with '-', so the caller keeps previewing)."""
    hint = f" [{subject}]" if subject else ""
    while True:
        name = input(f"Participant id to save the map for{hint} ('-' = cancel): ").strip()
        if name == "-":
            print("Cancelled.")
            return False
        name = name or subject
        if name:
            break
        print("  Type a participant id (e.g. elisa or S001).")
    path = bomi_teleop.resolve_calib_path(bomi_teleop.custom_map_name(name))
    os.makedirs(bomi_teleop.CALIB_DIR, exist_ok=True)
    bomi_map.save_map_bomi(path)
    print(f"Saved to {path}")
    return True


def _customize_and_save(cap, landmarker, calib_path: str, subject: str) -> None:
    bomi_map = bomi_teleop.BoMIMap()
    bomi_map.load_map_bomi(calib_path)

    cursor_filter = bomi_teleop.CursorFilter()
    crs_x, crs_y = bomi_teleop.BASE_WIDTH / 2.0, bomi_teleop.BASE_HEIGHT / 2.0
    map_window = bomi_teleop.MAP_WINDOW_NAME

    print(f"\n=== CUSTOMIZE '{calib_path}' (nothing is sent anywhere) ===")
    print(HELP_TEXT)

    cv2.namedWindow(map_window, cv2.WINDOW_NORMAL)
    cv2.imshow(map_window, np.zeros((2, 2, 3), dtype="uint8"))  # FULLSCREEN only sticks after a first frame
    cv2.waitKey(1)
    cv2.setWindowProperty(map_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    safety.force_fullscreen(map_window)  # some Qt builds ignore the request above; ask the WM directly too

    while True:
        _, crs_x, crs_y, _ = bomi_teleop.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        region = bomi_teleop.check_region_cursor(crs_x, crs_y)
        cv2.imshow(map_window, bomi_teleop.draw_cursor_map(crs_x, crs_y, region, HELP_TEXT))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('['):
            bomi_map.customize(rot_deg=-ROT_STEP_DEG)
        elif key == ord(']'):
            bomi_map.customize(rot_deg=ROT_STEP_DEG)
        elif key == ord('i'):
            bomi_map.customize(gain_x=-1.0)
        elif key == ord('o'):
            bomi_map.customize(gain_y=-1.0)
        elif key == ord('-'):
            bomi_map.customize(gain_x=1.0 / SCALE_STEP, gain_y=1.0 / SCALE_STEP)
        elif key == ord('='):
            bomi_map.customize(gain_x=SCALE_STEP, gain_y=SCALE_STEP)
        elif key == ord('h'):
            bomi_map.customize(off_x=-OFFSET_STEP_PX)
        elif key == ord('l'):
            bomi_map.customize(off_x=OFFSET_STEP_PX)
        elif key == ord('k'):
            bomi_map.customize(off_y=-OFFSET_STEP_PX)
        elif key == ord('j'):
            bomi_map.customize(off_y=OFFSET_STEP_PX)
        elif key == ord('r'):
            bomi_map.load_map_bomi(calib_path)
            print("Reset to the original calibration.")
        elif key == ord('s'):
            if _prompt_and_save(bomi_map, subject):
                return
        elif safety.quit_requested(key, map_window):
            print("Closed without saving.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("subject", nargs="?", default=None,
                        help="Participant id, e.g. elisa: the map is saved as calibrations/elisa_<date>_<time>.npz "
                             "(can also be typed at the save prompt)")
    parser.add_argument("--base", default=bomi_teleop.SHARED_MAP_NAME,
                        help=f"Map to customize, calibrations/<NAME>.npz (default: {bomi_teleop.SHARED_MAP_NAME})")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi_teleop.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model.")
    cli_args = parser.parse_args()

    calib_path = bomi_teleop.resolve_calib_path(cli_args.base)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No map '{calib_path}' found: run calibrate_bomi.py first (or check --base).")
        available = bomi_teleop.list_saved_maps()
        if available:
            print("        Available: " + ", ".join(available))
        sys.exit(1)
    subject = bomi_teleop.strip_npz(cli_args.subject) if cli_args.subject else None

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

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
            num_hands=bomi_teleop.LANDMARKER_NUM_HANDS,
            min_hand_detection_confidence=bomi_teleop.LANDMARKER_MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=bomi_teleop.LANDMARKER_MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=bomi_teleop.LANDMARKER_MIN_TRACKING_CONFIDENCE,
        )
        landmarker = hand_landmarker.HandLandmarker.create_from_options(landmarker_options)

        _customize_and_save(cap, landmarker, calib_path, subject)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
