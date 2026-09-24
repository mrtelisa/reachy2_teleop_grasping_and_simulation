#!/usr/bin/env python3
"""
BoMI hand -> cursor chain shared by the tools in this folder (calibrate_bomi.py,
customize_bomi.py, load_bomi.py, reaching_regions.py): webcam -> MediaPipe hand
landmarks -> autoencoder map -> Butterworth-filtered cursor on the
BASE_WIDTH x BASE_HEIGHT virtual screen, plus the calibrations/ helpers.
Nothing to run here.
"""

import datetime
import json
import os
import re
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
import scipy.signal as sgn
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import hand_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

import display  # every OpenCV window on the laptop screen (see display.py)

HAND_CONNECTIONS = hand_landmarker.HandLandmarksConnections.HAND_CONNECTIONS

# --- Virtual screen dimensions ---
BASE_WIDTH = 2550
BASE_HEIGHT = 1500

# Cursor low-pass filter: 3rd-order Butterworth, coefficients derived from
# the loop rate below rather than hardcoded for 50Hz.
CURSOR_FILTER_HZ = 30.0        # assumed webcam/control loop sample rate
CURSOR_FILTER_CUTOFF_HZ = 4.0  # cutoff frequency

# Calibration files live in the repo's 'calibrations/' folder. A bare --calib
# filename is placed there;
# a --calib value that already contains a path is used as-is.
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibrations")

# Continuous calibration, as in markerlessBoMI: the hand is recorded on every
# tracked frame for CALIB_DURATION_S (90 s there, at 50 Hz -> ~4500 samples)
CALIB_DURATION_S = 90.0

# One autoencoder map, trained once (calibrate_bomi.py) and customized per
# participant (customize_bomi.py) -- as markerlessBoMI was used
SHARED_MAP_NAME = "shared"      # calibrations/shared.npz (+ shared_calib.npy, its raw samples)
SAMPLES_SUFFIX = "_calib.npy"   # raw calibration samples (markerlessBoMI's SXXX_Calib.txt)
# A participant's customized map (markerlessBoMI's rotation/scale/offset_custom.txt):
# calibrations/<subject>_<YYYYMMDD_HHMMSS>.npz, so several customizations of the
# same participant coexist and "<subject>" alone resolves to the latest one
CUSTOM_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"
_CUSTOM_TIMESTAMP_RE = r"_\d{8}_\d{6}"

# MediaPipe Tasks hand-landmarker model (.task), in the repo's 'scripts/' folder.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "hand_landmarker.task"
)


def _resolve_calib_path(calib_arg: str) -> str:
    """Bare filename -> calibrations/ folder; explicit path -> used as given
    (.npz appended either way if missing, so 'elisa' -> calibrations/elisa.npz)."""
    if not calib_arg.endswith(".npz"):
        calib_arg += ".npz"
    if os.path.dirname(calib_arg):
        return calib_arg
    return os.path.join(CALIB_DIR, calib_arg)


def _resolve_samples_path(name: str) -> str:
    """Map name -> calibrations/<name>_calib.npy (the raw calibration samples)."""
    return os.path.join(CALIB_DIR, name + SAMPLES_SUFFIX)


def _strip_npz(name: str) -> str:
    return name[:-4] if name.endswith(".npz") else name


def _custom_map_name(subject: str) -> str:
    """New customized-map name for a participant: <subject>_<YYYYMMDD_HHMMSS>."""
    return f"{_strip_npz(subject)}_{datetime.datetime.now().strftime(CUSTOM_TIMESTAMP_FMT)}"


def _is_custom_map_name(name: str) -> bool:
    """True for a customize_bomi.py map name (<subject>_<YYYYMMDD_HHMMSS>)."""
    return re.search(_CUSTOM_TIMESTAMP_RE + r"$", name) is not None


def _custom_maps_of(subject: str) -> list:
    """Customized maps of a participant found in CALIB_DIR (names, oldest first)."""
    pattern = re.compile(re.escape(_strip_npz(subject)) + _CUSTOM_TIMESTAMP_RE + r"$")
    return sorted(n for n in _list_saved_maps() if pattern.match(n))


def _resolve_subject_map_path(subject: str) -> str:
    """What --subject means for a map: calibrations/<subject>.npz if that exact
    file exists (a map name, with or without .npz), else the latest
    customization calibrations/<subject>_<YYYYMMDD_HHMMSS>.npz saved by
    customize_bomi.py. Returns the exact-name path (not existing) if neither
    is there, so callers can report it."""
    exact = _resolve_calib_path(_strip_npz(subject))
    if os.path.exists(exact):
        return exact
    customs = _custom_maps_of(subject)
    if customs:
        return _resolve_calib_path(customs[-1])
    return exact


def _list_saved_maps() -> list:
    """Names of the .npz maps found in CALIB_DIR (for error messages)."""
    if not os.path.isdir(CALIB_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(CALIB_DIR) if f.endswith(".npz"))


def save_calib_samples(path: str, samples: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(samples, dtype=np.float64))


def load_calib_samples(path: str) -> np.ndarray:
    return np.load(path)


# Shared window name, so the calibrate/customize/load tools show the same
# cursor map window.
MAP_WINDOW_NAME = "BoMI - Cursor Map"

# Default cursor map size in pixels (the fullscreen map keeps its aspect ratio).
MAP_WINDOW_SIZE = (510, 300)


def check_region_cursor(crs_x: float, crs_y: float) -> int:
    """
    Returns region 1-9 based on 3x3 grid over BASE_WIDTH x BASE_HEIGHT.
    Layout:
        1 | 2 | 3   (top row)
        4 | 5 | 6   (middle row)
        7 | 8 | 9   (bottom row)
    """
    if crs_x < 847:
        col = 0
    elif crs_x <= 1697:
        col = 1
    else:
        col = 2

    if crs_y < 497:
        row = 0
    elif crs_y <= 997:
        row = 1
    else:
        row = 2

    return row * 3 + col + 1


# --- Autoencoder forward map ---
def _extract_hand_features(hand_landmarks) -> np.ndarray:
    """
    Flatten all 21 hand landmarks (x, y) into a 42-element vector. Left and
    right hands are not told apart: the participant always uses the same hand
    (the one the map was customized with).

    hand_landmarks is the list of NormalizedLandmark returned by MediaPipe
    Tasks (e.g. results.hand_landmarks[0]).
    """
    coords = [[lm.x, lm.y] for lm in hand_landmarks]
    return np.array(coords).flatten()


def _draw_hand_landmarks(frame, landmarks) -> None:
    """Draw MediaPipe Tasks hand landmarks/connections on a BGR OpenCV frame."""
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        points.append((x, y))

    for connection in HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 200, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def _quit_requested(key: int, window_name: str) -> bool:
    """True if Q/ESC was pressed, or the window was closed with the X button."""
    if key in (ord('q'), ord('Q'), 27):
        return True
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def _draw_cursor_map(crs_x: float, crs_y: float, region: int, message: str,
                      map_width: int = MAP_WINDOW_SIZE[0], map_height: int = MAP_WINDOW_SIZE[1]):
    """Rectangle representing the BASE_WIDTH x BASE_HEIGHT virtual screen, with
    the 9-region grid lines, a dot at the current cursor position and a
    message line (e.g. the tool's key help)."""
    canvas = np.full((map_height, map_width, 3), 30, dtype=np.uint8)
    sx = map_width / BASE_WIDTH
    sy = map_height / BASE_HEIGHT
    k = map_height / MAP_WINDOW_SIZE[1]   # lines, dot and text grow with the map (1 = default size)

    x1, x2 = int(847 * sx), int(1697 * sx)
    y1, y2 = int(497 * sy), int(997 * sy)
    for x in (x1, x2):
        cv2.line(canvas, (x, 0), (x, map_height), (90, 90, 90), max(1, round(k)))
    for y in (y1, y2):
        cv2.line(canvas, (0, y), (map_width, y), (90, 90, 90), max(1, round(k)))

    cx, cy = int(crs_x * sx), int(crs_y * sy)
    cv2.circle(canvas, (cx, cy), max(1, round(8 * k)), (0, 0, 255), -1)

    kt = min(k, 2.0)   # text grows less than the map, so the long help line still fits
    thick = max(1, round(2 * kt))
    cv2.putText(canvas, f"region={region}  cursor=({crs_x:.0f},{crs_y:.0f})",
                (round(10 * kt), round(25 * kt)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * kt, (255, 255, 255), thick)
    cv2.putText(canvas, message,
                (round(10 * kt), round(50 * kt)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * kt, (0, 255, 255), thick)
    return canvas


def open_fullscreen_window(window_name: str) -> tuple:
    """Create window_name in fullscreen and return its (width, height) in
    pixels. Used by the standalone tools (load_bomi.py, customize_bomi.py) to
    show the cursor map on the whole screen."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Show a frame BEFORE asking for fullscreen: some OpenCV builds (e.g. the
    # opencv-python 4.13 wheel, Qt backend) ignore the property on a window
    # that has not been mapped yet and leave a small window.
    cv2.imshow(window_name, np.zeros((MAP_WINDOW_SIZE[1], MAP_WINDOW_SIZE[0], 3), dtype=np.uint8))
    cv2.waitKey(100)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(300)   # let the window manager apply it before reading the size
    _, _, w, h = cv2.getWindowImageRect(window_name)
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) != cv2.WINDOW_FULLSCREEN:
        print(f"[WARNING] fullscreen not applied by this OpenCV/desktop, window is {w}x{h}")
    if w <= 0 or h <= 0:
        w, h = MAP_WINDOW_SIZE
    return w, h


CAM_WINDOW_NAME = "BoMI - Camera"
CAM_WINDOW_WIDTH = 640   # initial width in pixels (height from the camera aspect ratio), then resizable


def open_camera_window(cap) -> str:
    """Resizable window for show_camera() on the experimenter's monitor
    (display.use_monitor). Open it BEFORE the participant's window, so that
    one keeps the keyboard focus. Returns the window name."""
    display.use_monitor(CAM_WINDOW_NAME)
    cv2.namedWindow(CAM_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # Qt sizes the window on the first frame: show a black one of the camera
    # size, then set the initial size and centre it again at that size
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cv2.imshow(CAM_WINDOW_NAME, np.zeros((cam_h, cam_w, 3), dtype=np.uint8))
    cv2.waitKey(100)
    cv2.resizeWindow(CAM_WINDOW_NAME, CAM_WINDOW_WIDTH, round(CAM_WINDOW_WIDTH * cam_h / cam_w))
    cv2.waitKey(100)
    display.place(CAM_WINDOW_NAME)
    return CAM_WINDOW_NAME


def show_camera(frame, hand_detected: bool) -> None:
    """The webcam frame of update_bomi_cursor() (hand landmarks already drawn)
    in the open_camera_window() window, with the tracking status."""
    if frame is None:
        return
    status, colour = ("hand detected", (0, 255, 0)) if hand_detected else ("no hand", (0, 0, 255))
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.imshow(CAM_WINDOW_NAME, frame)


def draw_fullscreen_cursor_map(screen_w: int, screen_h: int, crs_x: float, crs_y: float,
                               region: int, message: str):
    """_draw_cursor_map() scaled to fill a screen_w x screen_h window: the
    virtual screen keeps its aspect ratio (BASE_WIDTH x BASE_HEIGHT), as large
    as possible and centred on a black frame."""
    scale = min(screen_w / BASE_WIDTH, screen_h / BASE_HEIGHT)
    map_w, map_h = max(1, int(BASE_WIDTH * scale)), max(1, int(BASE_HEIGHT * scale))
    frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    ox, oy = (screen_w - map_w) // 2, (screen_h - map_h) // 2
    frame[oy:oy + map_h, ox:ox + map_w] = _draw_cursor_map(crs_x, crs_y, region, message, map_w, map_h)
    return frame


class CursorFilter:
    """
    3rd-order Butterworth low-pass filter for the (crs_x, crs_y) cursor position.
    Coefficients are derived from the actual sample rate (CURSOR_FILTER_HZ) instead 
    of being hardcoded for a fixed frequency loop.
    """

    ORDER = 3

    def __init__(self, sample_hz: float = CURSOR_FILTER_HZ, cutoff_hz: float = CURSOR_FILTER_CUTOFF_HZ) -> None:
        nyquist = sample_hz / 2.0
        self._b, self._a = sgn.butter(self.ORDER, cutoff_hz / nyquist, btype="low")
        self._in_history = np.zeros((self.ORDER, 2))
        self._out_history = np.zeros((self.ORDER, 2))

    def update(self, crs_x: float, crs_y: float) -> tuple:
        new_input = np.array([crs_x, crs_y])
        new_output = self._b[0] * new_input
        for i in range(self.ORDER):
            new_output += self._b[i + 1] * self._in_history[i]
        for i in range(self.ORDER):
            new_output -= self._a[i + 1] * self._out_history[i]

        self._in_history = np.roll(self._in_history, 1, axis=0)
        self._in_history[0] = new_input
        self._out_history = np.roll(self._out_history, 1, axis=0)
        self._out_history[0] = new_output

        return float(new_output[0]), float(new_output[1])


# --- Autoencoder forward map parameters ---
# Same architecture/hyperparameters as Naji's compute_bomi_map.Autoencoder.train_network,
# as used for dr_mode="ae" in main_reaching_FullHand_MOD_ae.py's train_ae():
# Input -> Dense(32, tanh) -> Dense(32, tanh) -> Dense(2, linear) [latent/cursor]
#       -> Dense(32, tanh) -> Dense(32, tanh) -> Dense(n_features, linear) [reconstruction]
AE_N_STEPS = 3001      # training epochs (n_steps), full-batch
AE_LR = 0.02           # Adam learning rate
AE_HIDDEN_UNITS = 32   # nh1 = nh2
AE_ACTIVATION = "tanh"
AE_SEED = 20
AE_LATENT_DIM = 2      # cu: 2 code units -> (crs_x, crs_y)
AE_TEST_SPLIT = 0.2    # 80/20 train/test split of the shuffled calibration samples


def compute_vaf(x: np.ndarray, x_rec: np.ndarray) -> float:
    """Variance accounted for (%) by the reconstruction, both zero-meaned."""
    x_zm = x - x.mean(axis=0)
    x_rec_zm = x_rec - x_rec.mean(axis=0)
    return float(100.0 * (1.0 - np.sum((x_zm - x_rec_zm) ** 2) / np.sum(x_zm ** 2)))


def compute_latent_variance(latent: np.ndarray) -> list:
    """Share (%) of the latent variance carried by each code unit."""
    var = np.sum((latent - latent.mean(axis=0)) ** 2, axis=0)
    return [float(v) for v in 100.0 * var / var.sum()]


class BoMIMap:
    """
    Autoencoder forward map: raw hand landmarks -> 2D cursor in screen space.

    A dense autoencoder is trained on the raw landmark calibration samples
    (reconstruction loss only, no separate normalization), and its 2-unit
    latent layer is used standalone as the cursor space at inference time --
    exactly the dr_mode="ae" path in Naji's BoMI pipeline
    (compute_bomi_map.Autoencoder.train_network + reaching_functions.get_mapped_values).
    Scale/offset map the latent codes' peak-to-peak range onto the screen size,
    centered on the mean, same as train_ae()'s post-training step (with rot=0).

    The screen-space map is stored as a single affine transform cu_screen = A @
    cu_latent + b (A starts out diagonal, i.e. plain per-axis scale).

    The fitted parameters (encoder weights/biases, A, b) are plain numpy arrays,
    so the map can be saved to / loaded from a .npz file and reused without
    repeating calibration.
    """

    def __init__(self) -> None:
        self._w1 = self._b1 = None  # encoder layer 1 (Dense, tanh)
        self._w2 = self._b2 = None  # encoder layer 2 (Dense, tanh)
        self._w3 = self._b3 = None  # encoder layer 3 (Dense, linear) -> latent/cursor
        self._A = np.eye(2)     # latent -> screen affine map (starts diagonal = plain scale)
        self._b = np.zeros(2)
        self.metrics = {}       # training report of fit() (VAF, latent variance, sample counts)
        self.fitted = False

    def fit(self, samples, test_split: float = AE_TEST_SPLIT) -> dict:
        """Train the autoencoder on the calibration samples the way markerlessBoMI's
        train_ae does: drop all-zero rows, shuffle, hold out test_split for
        testing, train full-batch on the rest, then scale the training latent
        codes onto the screen. Returns (and stores in self.metrics) VAF and
        latent-variance share on train and test."""
        X = np.array(samples, dtype=np.float32)
        X = X[np.any(X != 0, axis=1)]  # frames recorded before the first detection
        n_features = X.shape[1]

        tf.keras.backend.clear_session()
        np.random.seed(AE_SEED)
        tf.random.set_seed(AE_SEED)
        initializer = tf.keras.initializers.GlorotNormal(seed=AE_SEED)

        np.random.shuffle(X)
        split = int(round(len(X) * (1.0 - test_split)))
        train_x, test_x = X[:split], X[split:]

        inputs = Input(shape=(n_features,))
        hidden1 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(inputs)
        hidden1 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(hidden1)
        latent = Dense(AE_LATENT_DIM, kernel_initializer=initializer)(hidden1)
        hidden2 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(latent)
        hidden2 = Dense(AE_HIDDEN_UNITS, activation=AE_ACTIVATION, kernel_initializer=initializer)(hidden2)
        predictions = Dense(n_features, kernel_initializer=initializer)(hidden2)

        encoder = Model(inputs=inputs, outputs=latent)
        autoencoder = Model(inputs=inputs, outputs=predictions)
        autoencoder.compile(loss="mse", optimizer=Adam(learning_rate=AE_LR))

        print(f"Training autoencoder BoMI map on {len(train_x)} samples "
              f"({len(test_x)} held out, {AE_N_STEPS} epochs)...")
        t0 = time.time()
        autoencoder.fit(x=train_x, y=train_x, epochs=AE_N_STEPS, verbose=0,
                        batch_size=len(train_x), shuffle=False)
        print(f"Autoencoder training done in {time.time() - t0:.0f}s.")

        # Keep only the encoder half (first 3 Dense layers) for standalone inference,
        # same as train_ae() only persisting weights1/2/3 + biases1/2/3.
        dense_layers = [layer for layer in autoencoder.layers if layer.get_weights()]
        self._w1, self._b1 = dense_layers[0].get_weights()
        self._w2, self._b2 = dense_layers[1].get_weights()
        self._w3, self._b3 = dense_layers[2].get_weights()

        train_cu = encoder.predict(train_x, verbose=0)

        extent = np.ptp(train_cu, axis=0)
        extent = np.where(extent > 1e-6, extent, 1.0)

        screen = np.array([BASE_WIDTH, BASE_HEIGHT], dtype=float)

        scale = screen / extent
        self._A = np.diag(scale)
        self._b = screen / 2.0 - (train_cu * scale).mean(axis=0)
        self.fitted = True

        # Training report, as train_ae prints/saves it (vaf.txt, latent_variance.txt)
        self.metrics = {
            "n_train": int(len(train_x)),
            "n_test": int(len(test_x)),
            "vaf_train": compute_vaf(train_x, autoencoder.predict(train_x, verbose=0)),
            "cu_train": compute_latent_variance(train_cu),
        }
        if len(test_x) > 1:
            test_cu = encoder.predict(test_x, verbose=0)
            self.metrics["vaf_test"] = compute_vaf(test_x, autoencoder.predict(test_x, verbose=0))
            self.metrics["cu_test"] = compute_latent_variance(test_cu)
        self.print_metrics()
        return self.metrics

    def print_metrics(self) -> None:
        m = self.metrics
        if not m:
            return
        print(f"  samples: {m['n_train']} train / {m['n_test']} test")
        print(f"  VAF: train {m['vaf_train']:.2f}%" + (f", test {m['vaf_test']:.2f}%" if "vaf_test" in m else ""))
        cu = " / ".join(f"{v:.1f}%" for v in m["cu_train"])
        print(f"  latent variance (CU x / CU y): train {cu}"
              + (", test " + " / ".join(f"{v:.1f}%" for v in m["cu_test"]) if "cu_test" in m else ""))

    def transform(self, features: np.ndarray) -> tuple:
        """
        Autoencoder BoMI map: raw hand landmarks -> 2D cursor in screen space,
        via the trained encoder's forward pass (2 tanh hidden layers + linear
        latent layer), same as reaching_functions.get_mapped_values(dr_mode="ae").
        Returns (crs_x, crs_y) in pixels, clipped to the screen size.
        """
        h = np.tanh(np.dot(features, self._w1) + self._b1)
        h = np.tanh(np.dot(h, self._w2) + self._b2)
        cu = np.dot(h, self._w3) + self._b3  # latent code == raw cursor position
        cu = self._A @ cu + self._b  # map latent extent onto the screen size
        crs_x = float(np.clip(cu[0], 0, BASE_WIDTH))
        crs_y = float(np.clip(cu[1], 0, BASE_HEIGHT))
        return crs_x, crs_y

    def customize(self, rot_deg: float = 0.0, gain_x: float = 1.0, gain_y: float = 1.0,
                  off_x: float = 0.0, off_y: float = 0.0) -> None:
        """
        Compose an extra screen-space rotation/gain/offset on top of the current
        map, exactly like Naji's rotation_custom/scale_custom/offset_custom
        (applied in reaching_functions.get_mapped_values after the base AE/PCA
        map): recentre on the screen middle, rotate (screen space is left-handed,
        hence the sign flip), apply a per-axis gain -- negative flips that axis --
        then offset. Composes into the same (A, b) used by transform(), so calling
        this repeatedly (e.g. from customize_bomi.py) keeps stacking correctly, and
        the result is saved/loaded like any other BoMIMap.
        """
        center = np.array([BASE_WIDTH, BASE_HEIGHT]) / 2.0
        rad = -np.radians(rot_deg)  # left-handed screen space
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        gain_rot = np.array([gain_x, gain_y])[:, None] * rot  # diag(gain) @ rot

        self._A = gain_rot @ self._A
        self._b = gain_rot @ (self._b - center) + center + np.array([off_x, off_y])

    def save(self, path: str) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted BoMIMap.")
        np.savez(
            path,
            w1=self._w1, b1=self._b1,
            w2=self._w2, b2=self._b2,
            w3=self._w3, b3=self._b3,
            A=self._A,
            b=self._b,
            metrics=json.dumps(self.metrics),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        if "w1" not in data:
            raise ValueError(
                f"'{path}' is not an autoencoder calibration (keys: {list(data.keys())}). "
                "It was probably saved by the old PCA map: rebuild it with calibrate_bomi.py."
            )
        self._w1, self._b1 = data["w1"], data["b1"]
        self._w2, self._b2 = data["w2"], data["b2"]
        self._w3, self._b3 = data["w3"], data["b3"]
        self._A = data["A"]
        self._b = data["b"]
        self.metrics = json.loads(str(data["metrics"])) if "metrics" in data else {}
        self.fitted = True


# --- Hand tracking ---
def create_hand_landmarker(model_path: str = DEFAULT_MODEL_PATH):
    """MediaPipe Tasks hand landmarker in VIDEO mode, one hand."""
    return hand_landmarker.HandLandmarker.create_from_options(
        hand_landmarker.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def update_bomi_cursor(cap, landmarker, bomi_map: BoMIMap, cursor_filter: CursorFilter,
                       crs_x: float, crs_y: float):
    """One iteration of hand tracking: reads a webcam frame, runs the hand
    landmarker, and returns (frame_with_landmarks, crs_x, crs_y, hand_detected).
    Cursor position is carried over unchanged when no hand is detected."""
    ret, frame = cap.read()
    if not ret:
        return None, crs_x, crs_y, False

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

    if not results.hand_landmarks:
        return frame, crs_x, crs_y, False

    hl = results.hand_landmarks[0]
    _draw_hand_landmarks(frame, hl)
    crs_x, crs_y = bomi_map.transform(_extract_hand_features(hl))
    crs_x, crs_y = cursor_filter.update(crs_x, crs_y)
    return frame, crs_x, crs_y, True


# --- Phases ---
def _calibration_phase(cap, landmarker, duration_s: float = CALIB_DURATION_S,
                       window_name: str = "BoMI - Calibration") -> list:
    """Continuous calibration, as in markerlessBoMI's compute_calibration: once
    started (SPACE), the hand features are recorded on every frame where the
    hand is tracked until duration_s has elapsed, with the time remaining on
    screen. No sample is picked by hand: the participant just keeps moving the
    hand through the whole workspace. Q/Esc/closing the window quits.
    Returns the list of 42-element samples."""
    samples = []

    print("\n=== CALIBRATION ===")
    print(f"Continuous recording for {duration_s:.0f}s: keep moving your hand through all the "
          "positions you intend to use, at the speed you will use them.")
    print("SPACE = start   |   Q = quit")

    start_time = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        hand = results.hand_landmarks[0] if results.hand_landmarks else None
        if hand is not None:
            _draw_hand_landmarks(frame, hand)

        now = time.time()
        if start_time is None:
            label = f"Get ready: SPACE starts {duration_s:.0f}s of recording   Q=quit"
            color = (0, 255, 255)
        else:
            if hand is not None:
                samples.append(_extract_hand_features(hand))
            remaining = duration_s - (now - start_time)
            if remaining <= 0:
                print(f"  Calibration done ({len(samples)} samples in {duration_s:.0f}s)")
                break
            label = f"Calibration time: {remaining:4.0f}s   samples: {len(samples)}"
            color = (0, 255, 0) if hand is not None else (0, 0, 255)
            if hand is None:
                cv2.putText(frame, "no hand detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and start_time is None:
            start_time = time.time()
            print("  Recording...")
        elif _quit_requested(key, window_name):
            print("Aborted.")
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return samples
