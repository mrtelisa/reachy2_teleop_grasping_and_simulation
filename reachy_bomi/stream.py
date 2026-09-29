#!/usr/bin/env python3
"""Blocking live feed of a Reachy camera view (used by camera_viewer.py)."""

import time
from typing import Callable

import cv2
from reachy2_sdk.media.camera import CameraView, DepthCamera

STREAM_HZ = 12.0  # [Hz] frames requested from the robot (the camera itself runs faster)


def stream_blocking(
    depth_cam: DepthCamera, window_name: str, quit_requested: Callable[[int, str], bool],
    view: CameraView = CameraView.LEFT, rate_hz: float = STREAM_HZ,
) -> None:
    """Show frames at rate_hz until quit_requested(key, window_name) is True.
    The wait between two frames is spent in cv2.waitKey, so the window keeps
    reacting to Q/ESC."""
    print(f"\n=== LIVE RGB STREAM (no detection, {rate_hz:.0f} Hz) ===  Q = quit")
    period = 1.0 / rate_hz
    next_frame = time.monotonic()
    while True:
        result = depth_cam.get_frame(view=view)
        if result is not None:
            frame, _timestamp = result
            cv2.imshow(window_name, frame)

        # wait for the next frame slot (at least 1 ms, for the window events)
        next_frame += period
        wait_s = next_frame - time.monotonic()
        if wait_s < 0:   # the frame took longer than the period: restart the clock
            next_frame = time.monotonic()
        key = cv2.waitKey(max(1, int(wait_s * 1000))) & 0xFF
        if quit_requested(key, window_name):
            break
