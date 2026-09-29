"""
Shows every OpenCV window of these scripts on one screen (by default the
laptop's built-in one), whatever terminal/monitor they are launched from.

Importing this module is enough (safety.py does it, so every script that
opens windows): it wraps cv2.namedWindow / cv2.imshow so that every new window
is moved to that screen (centred) the first time it is shown, and gets the
keyboard focus. Fullscreen windows then fill that screen. matplotlib figures
are moved with place_figure(). Nothing to run here.

Screen: $BOMI_SCREEN if set (an xrandr output name, e.g. DP-1, or its prefix),
else the built-in panel (eDP / LVDS / DSI), else the primary output. If xrandr
finds none, windows are left where the window manager puts them.

Windows passed to use_monitor() (e.g. the experimenter's camera view) go
instead on the monitor: $BOMI_MONITOR if set, else the first other connected
output (the screen above if there is only one). They do not take the focus.
"""

import os
import re
import subprocess

import cv2

SCREEN_ENV = "BOMI_SCREEN"
MONITOR_ENV = "BOMI_MONITOR"
BUILTIN_PREFIXES = ("eDP", "LVDS", "DSI")

_outputs = None      # [(name, primary, x, y, w, h)] of the connected outputs, from xrandr
_screen = False      # (name, x, y, w, h) once looked up, None if not found
_monitor = False     # same, for use_monitor() windows
_placed = set()      # windows already moved to their screen
_on_monitor = set()  # windows that go on the monitor

_named_window = cv2.namedWindow
_imshow = cv2.imshow
_destroy_window = cv2.destroyWindow
_destroy_all_windows = cv2.destroyAllWindows


def outputs() -> list:
    """[(name, primary, x, y, w, h)] of the connected outputs ([] without xrandr)."""
    global _outputs
    if _outputs is None:
        try:
            out = subprocess.run(["xrandr", "--query"], capture_output=True, text=True, timeout=5).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("[display] xrandr not available: windows are not moved")
            out = ""
        _outputs = [(m[0], m[1] == " primary", int(m[4]), int(m[5]), int(m[2]), int(m[3]))
                    for m in re.findall(r"^(\S+) connected( primary)? (\d+)x(\d+)\+(\d+)\+(\d+)", out, re.M)]
    return _outputs


def screen():
    """(name, x, y, w, h) of the screen the windows go to, or None."""
    global _screen
    if _screen is not False:
        return _screen
    _screen = None
    outputs_ = outputs()
    wanted = os.environ.get(SCREEN_ENV)
    if wanted:
        match = [o for o in outputs_ if o[0].startswith(wanted)]
        if not match:
            print(f"[display] {SCREEN_ENV}={wanted} not among {[o[0] for o in outputs_]}: windows are not moved")
            return None
    else:
        match = [o for o in outputs_ if o[0].startswith(BUILTIN_PREFIXES)] or [o for o in outputs_ if o[1]]
    if match:
        name, _, x, y, w, h = match[0]
        _screen = (name, x, y, w, h)
        print(f"[display] windows on {name} ({w}x{h}+{x}+{y})")
    return _screen


def monitor():
    """(name, x, y, w, h) of the screen use_monitor() windows go to, or None."""
    global _monitor
    if _monitor is not False:
        return _monitor
    s = screen()
    wanted = os.environ.get(MONITOR_ENV)
    if wanted:
        match = [o for o in outputs() if o[0].startswith(wanted)]
        if not match:
            print(f"[display] {MONITOR_ENV}={wanted} not among {[o[0] for o in outputs()]}")
    else:
        match = [o for o in outputs() if s is None or o[0] != s[0]]
    if match:
        name, _, x, y, w, h = match[0]
        _monitor = (name, x, y, w, h)
        print(f"[display] experimenter windows on {name} ({w}x{h}+{x}+{y})")
    else:
        _monitor = s
    return _monitor


def use_monitor(window_name: str) -> None:
    """Put window_name on the monitor instead of the screen, without the focus.
    Call it before the window is created."""
    _on_monitor.add(window_name)


def move(window_name: str, x: int, y: int) -> None:
    """Move window_name to (x, y) relative to the top-left corner of its screen."""
    s = monitor() if window_name in _on_monitor else screen()
    ox, oy = (s[1], s[2]) if s else (0, 0)
    cv2.moveWindow(window_name, ox + x, oy + y)


def place(window_name: str) -> None:
    """Centre window_name on its screen and (if not on the monitor) give it the
    keyboard focus."""
    _placed.add(window_name)
    s = monitor() if window_name in _on_monitor else screen()
    if s is None:
        return
    _, _, _, w, h = s
    try:
        _, _, ww, wh = cv2.getWindowImageRect(window_name)
    except cv2.error:
        ww, wh = 0, 0
    move(window_name, max(0, (w - ww) // 2), max(0, (h - wh) // 2))
    cv2.waitKey(1)
    if window_name in _on_monitor:
        return
    try:   # focus (keys are read by the window, not by the terminal); optional
        subprocess.run(["wmctrl", "-a", window_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def place_figure(fig) -> None:
    """Centre a matplotlib figure window on the screen (TkAgg or Qt backends;
    no-op for the others). Call it before plt.show()."""
    s = screen()
    window = getattr(getattr(fig.canvas, "manager", None), "window", None)
    if s is None or window is None:
        return
    _, sx, sy, w, h = s
    fw, fh = (fig.get_size_inches() * fig.dpi).astype(int)
    x, y = sx + max(0, (w - fw) // 2), sy + max(0, (h - fh) // 2)
    if hasattr(window, "wm_geometry"):   # TkAgg
        window.wm_geometry(f"+{x}+{y}")
    elif hasattr(window, "move"):        # QtAgg
        window.move(x, y)


def _named_window_on_screen(winname, flags=cv2.WINDOW_AUTOSIZE):
    _named_window(winname, flags)
    if winname not in _placed:
        place(winname)


# If set, called as imshow_listener(window_name, image_shape) after every cv2.imshow
# (reachy_control.py: where the cursor has just been drawn on the screen)
imshow_listener = None


def _imshow_on_screen(winname, mat):
    _imshow(winname, mat)
    if winname not in _placed:
        place(winname)
    if imshow_listener is not None:
        imshow_listener(winname, getattr(mat, "shape", None))


def _destroy_window_on_screen(winname):
    _placed.discard(winname)
    _destroy_window(winname)


def _destroy_all_windows_on_screen():
    _placed.clear()
    _destroy_all_windows()


cv2.namedWindow = _named_window_on_screen
cv2.imshow = _imshow_on_screen
cv2.destroyWindow = _destroy_window_on_screen
cv2.destroyAllWindows = _destroy_all_windows_on_screen
