#!/usr/bin/env python3
"""
Step 1, done ONCE (no robot needed): build the shared autoencoder map, as
markerlessBoMI's "Calibration" + "Calculate BoMI map" on a single recording.

  1. continuous calibration: CALIB_DURATION_S (90 s) of hand recording, saved
     raw to calibrations/shared_calib.npy (reused if already there, e.g. to
     retrain after changing the AE hyperparameters, unless --recalibrate);
  2. offline autoencoder training on those samples (80/20 train/test, VAF and
     latent variance printed and stored in the map), saved to
     calibrations/shared.npz;
  3. live preview of the map (nothing sent anywhere), Q = quit.

Every participant then only customizes this map: customize_bomi.py SUBJECT.

    python3 calibrate_bomi.py [NAME] [--cam INDEX] [--model PATH] [--duration S] [--recalibrate]
"""

import argparse
import os
import sys

import cv2
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode

import socket_client as bomi

MIN_SAMPLES = 100  # fewer tracked frames than this in a recording = something is wrong with the webcam


def preview_map(cap, landmarker, bomi_map: bomi.BoMIMap, message: str = "(preview)") -> None:
    """Live cursor map with the given map, nothing sent anywhere; Q quits."""
    cursor_filter = bomi.CursorFilter()
    crs_x, crs_y = bomi.BASE_WIDTH / 2.0, bomi.BASE_HEIGHT / 2.0
    map_window = bomi.MAP_WINDOW_NAME

    print("\n=== PREVIEW (nothing is sent anywhere) ===  Q = quit")

    while True:
        _, crs_x, crs_y, _ = bomi.update_bomi_cursor(
            cap, landmarker, bomi_map, cursor_filter, crs_x, crs_y,
        )
        region = bomi.check_region_cursor(crs_x, crs_y)
        cv2.imshow(map_window, bomi._draw_cursor_map(crs_x, crs_y, region, message))

        key = cv2.waitKey(1) & 0xFF
        if bomi._quit_requested(key, map_window):
            return


def open_landmarker(model_path: str):
    landmarker_options = hand_landmarker.HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=model_path),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return hand_landmarker.HandLandmarker.create_from_options(landmarker_options)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name", nargs="?", default=bomi.SHARED_MAP_NAME,
                        help=f"Map name (default: {bomi.SHARED_MAP_NAME}, the one customize_bomi.py "
                             "loads): files are calibrations/<NAME>_calib.npy and <NAME>.npz")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", default=bomi.DEFAULT_MODEL_PATH,
                        help="Path to the MediaPipe hand_landmarker.task model.")
    parser.add_argument("--duration", type=float, default=bomi.CALIB_DURATION_S,
                        help=f"Recording duration in seconds (default: {bomi.CALIB_DURATION_S:.0f})")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Record again even if calibrations/<NAME>_calib.npy exists")
    cli_args = parser.parse_args()

    if not os.path.exists(cli_args.model):
        print(f"[ERROR] MediaPipe model not found: '{cli_args.model}'")
        print("        Download hand_landmarker.task and pass its path with --model.")
        sys.exit(1)

    samples_path = bomi._resolve_samples_path(cli_args.name)
    map_path = bomi._resolve_calib_path(cli_args.name)
    reuse = os.path.exists(samples_path) and not cli_args.recalibrate
    if reuse:
        # markerlessBoMI: "A previous calibration file was found. Do you want to reuse it?"
        answer = input(f"Found {samples_path}. Reuse it and skip the recording? [Y/n] ").strip().lower()
        reuse = answer in ("", "y", "yes")

    cap = None
    landmarker = None
    try:
        cap = cv2.VideoCapture(cli_args.cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cli_args.cam}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame, not a growing backlog
        landmarker = open_landmarker(cli_args.model)

        if reuse:
            samples = bomi.load_calib_samples(samples_path)
            print(f"Loaded {len(samples)} calibration samples from {samples_path}")
        else:
            samples = bomi._calibration_phase(cap, landmarker, duration_s=cli_args.duration)
            if len(samples) < MIN_SAMPLES:
                print(f"[ERROR] Only {len(samples)} samples with a tracked hand: check the webcam/lighting and retry.")
                sys.exit(1)
            bomi.save_calib_samples(samples_path, samples)
            print(f"Raw samples saved to {samples_path}")

        # Offline training (markerlessBoMI's "Calculate BoMI map")
        bomi_map = bomi.BoMIMap()
        bomi_map.fit(samples)
        os.makedirs(bomi.CALIB_DIR, exist_ok=True)
        bomi_map.save(map_path)
        print(f"Map saved to {map_path}")
        customized = [n for n in bomi._list_saved_maps() if bomi._is_custom_map_name(n)]
        if customized:
            print(f"[WARN] Existing customizations ({', '.join(customized)}) were made on the previous map: "
                  "redo customize_bomi.py for participants still to be tested.")

        preview_map(cap, landmarker, bomi_map, f"(preview {cli_args.name})")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
