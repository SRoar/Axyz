"""One command for the Dev 2 live setup.

Usage:
    python tools/setup_live.py                 # the whole thing: find page -> track pen tip / fingertip
    python tools/setup_live.py --tune-marker   # first tune a coloured pen marker (optional, more robust)
    python tools/setup_live.py --calib-window  # first run the stand-alone calibration window
    python tools/setup_live.py --source webcam # any camera flag is passed through

What happens (no keys needed):
  1. The tracker looks for the page: whole sheet in view, hands out, hold still ~1.5 s.
     Orange outline + green bar while it waits; it locks by itself and learns the empty scene.
  2. Reach in with the pen (or point with a finger). The tip is the end of whatever reaches in
     from the edge of the view. It is followed off the page too ("OFF PAGE: LEFT", ...).
  Keys in the tracker: c = find page again, b = re-learn background, 1-9 = box, ESC = quit.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def run(title, args, help_text, passthrough):
    print("\n" + "=" * 78 + f"\n  {title}\n  {help_text}\n" + "=" * 78, flush=True)
    return subprocess.call([sys.executable, "-u", *args, *passthrough])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-window", action="store_true", help="run tools/calibrate_page.py first")
    ap.add_argument("--tune-marker", action="store_true", help="run tools/tune_marker.py first")
    a, passthrough = ap.parse_known_args()

    if a.calib_window:
        run("CALIBRATE PAGE", ["tools/calibrate_page.py"],
            "Whole sheet in view, hands out, hold still. It saves by itself.", passthrough)
    if a.tune_marker:
        run("TUNE PEN MARKER", ["tools/tune_marker.py"],
            "Click the marker in 'frame', check 'mask' shows one blob, press s, then ESC.", passthrough)
    return run("LIVE TRACKER", ["-m", "src.tracker"],
               "Hands out, whole page in view, hold still until the outline turns green.\n"
               "  Then reach in with the pen / finger. ESC to quit.", passthrough)


if __name__ == "__main__":
    sys.exit(main())
