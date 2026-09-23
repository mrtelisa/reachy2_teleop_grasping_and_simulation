#!/usr/bin/env python3
"""
Step 2, once per participant (no robot needed): markerlessBoMI's
"Customization". Loads the shared autoencoder map (calibrations/shared.npz,
from calibrate_bomi.py), lets you rotate / flip / scale / offset it live on the
participant's hand, shown fullscreen, and saves it as
calibrations/<SUBJECT>_<YYYYMMDD_HHMMSS>.npz -- load_bomi.py and the
reaching test, given --subject SUBJECT, load the latest of those. The shared
map is untouched.

Keys:
    [ / ]   rotate -5 / +5 degrees
    i / o   flip X axis / flip Y axis
    - / +   scale down / up (both axes)
    h / l   nudge offset left / right
    k / j   nudge offset up / down
    r       reset (discard all changes, reload the shared map)
    s       save: asks the participant id (default: SUBJECT) and writes
            calibrations/<id>_<date>_<time>.npz ('-' at the prompt cancels)
    q       quit without saving

Usage:
    python3 customize_bomi.py [SUBJECT] [--base NAME] [--cam INDEX] [--model PATH]
"""

import argparse
import os
import sys

import cv2
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

import bomi

ROT_STEP_DEG = 5.0
SCALE_STEP = 1.1     # multiplicative
OFFSET_STEP_PX = 10.0

HELP_TEXT = (
    "[ ]=rotate  i/o=flip X/Y  -/+=scale  hjkl=offset  r=reset  s=save  q=quit"
)


def _prompt_and_save(bomi_map: bomi.BoMIMap, subject: str) -> bool:
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
    path = bomi._resolve_calib_path(bomi._custom_map_name(name))
    os.makedirs(bomi.CALIB_DIR, exist_ok=True)
    bomi_map.save(path)
    print(f"Saved to {path}")
    return True


def _customize_and_save(cap, landmarker, calib_path: str, subject: str) -> None:
    bomi_map = bomi.BoMIMap()
    bomi_map.load(calib_path)

    cursor_filter = bomi.CursorFilter()
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0
    map_window = bomi.MAP_WINDOW_NAME
    screen_w, screen_h = bomi.open_fullscreen_window(map_window)

    print(f"\n=== CUSTOMIZE '{calib_path}' (nothing is sent anywhere) ===")
    print(HELP_TEXT)

    while True:
        _, crs_x, crs_y, _ = bomi.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        region = bomi.check_region_cursor(crs_x, crs_y)
        cv2.imshow(map_window, bomi.draw_fullscreen_cursor_map(screen_w, screen_h, crs_x, crs_y, region, HELP_TEXT))

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
        elif key == ord('+'):
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
            bomi_map.load(calib_path)
            print("Reset to the original calibration.")
        elif key == ord('s'):
            if _prompt_and_save(bomi_map, subject):
                return
        elif bomi._quit_requested(key, map_window):
            print("Closed without saving.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("subject", nargs="?", default=None,
                        help="Participant id, e.g. elisa: the map is saved as calibrations/elisa_<date>_<time>.npz "
                             "(can also be typed at the save prompt)")
    parser.add_argument("--base", default=bomi.SHARED_MAP_NAME,
                        help=f"Map to customize, calibrations/<NAME>.npz (default: {bomi.SHARED_MAP_NAME})")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model.")
    cli_args = parser.parse_args()

    calib_path = bomi._resolve_calib_path(cli_args.base)
    if not os.path.exists(calib_path):
        print(f"[ERROR] No map '{calib_path}' found: run calibrate_bomi.py first (or check --base).")
        available = bomi._list_saved_maps()
        if available:
            print("        Available: " + ", ".join(available))
        sys.exit(1)
    subject = bomi._strip_npz(cli_args.subject) if cli_args.subject else None

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
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
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
