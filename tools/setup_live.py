"""One command for the Dev 2 live setup.

Usage:
    python tools/setup_live.py                 # find page -> track the pen tip -> distances to the boxes
    python tools/setup_live.py --prescan       # first find the answer boxes on a blank sheet (Gemini)
    python tools/setup_live.py --tune-marker   # first tune a coloured pen marker (optional)
    python tools/setup_live.py --calib-window  # first run the stand-alone calibration window
    python tools/setup_live.py --source webcam # any camera flag is passed through

What happens (no keys needed):
  0. (--prescan, or no data/layout.json yet) ANSWER BOXES: blank sheet in view, all 4 corners,
     hands out. The page locks, Gemini finds the numbered answer boxes, ENTER saves data/layout.json.
  1. The tracker looks for the page: whole sheet in view, hands out, hold still ~1.5 s.
     Orange outline + green bar while it waits; it locks by itself and learns the empty scene.
  2. Reach in with the pen. Its tip is located on the paper (cm from the top-left corner) and the
     "tracker: page" window lists the distance and direction from the tip to every answer box,
     with an arrow to the selected one. It is followed off the page too ("OFF PAGE: LEFT", ...).
  Keys in the tracker: 1-9 = arrow to box N, c = find page again, b = re-learn background, ESC = quit.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LAYOUT_PATH = "data/layout.json"


def run(title, args, help_text, passthrough):
    print("\n" + "=" * 78 + f"\n  {title}\n  {help_text}\n" + "=" * 78, flush=True)
    return subprocess.call([sys.executable, "-u", *args, *passthrough])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prescan", action="store_true", help="find the answer boxes first (python -m src.prescan)")
    ap.add_argument("--calib-window", action="store_true", help="run tools/calibrate_page.py first")
    ap.add_argument("--tune-marker", action="store_true", help="run tools/tune_marker.py first")
    a, passthrough = ap.parse_known_args()

    if a.prescan or not os.path.exists(LAYOUT_PATH):
        run("ANSWER BOXES", ["-m", "src.prescan"],
            "BLANK sheet in view (all 4 corners), hands out. Check the green boxes, ENTER = save.", passthrough)
    if a.calib_window:
        run("CALIBRATE PAGE", ["tools/calibrate_page.py"],
            "Whole sheet in view, hands out, hold still. It saves by itself.", passthrough)
    if a.tune_marker:
        run("TUNE PEN MARKER", ["tools/tune_marker.py"],
            "Click the marker in 'frame', check 'mask' shows one blob, press s, then ESC.", passthrough)
    return run("LIVE TRACKER", ["-m", "src.tracker"],
               "Hands out, whole page in view, hold still until the outline turns green.\n"
               "  Then write with the pen: tip position + distance to each answer box. ESC to quit.", passthrough)


if __name__ == "__main__":
    sys.exit(main())
