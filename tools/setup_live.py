"""Run the Dev 2 setup back to back: calibrate the page -> tune the pen marker -> live tracker.

Usage:
    python tools/setup_live.py                 # all three steps
    python tools/setup_live.py --skip-calib    # keep data/page_calibration.json, start at the marker
    python tools/setup_live.py --only-live     # just the live tracker view
    python tools/setup_live.py --source webcam # any camera flag is passed to every step

Each step opens its own window; close it (as described below) and the next one starts.
  1. calibrate  page found automatically, saves itself after 1.5 s of holding still
                (click a corner to switch to manual: TL, TR, BR, BL, then ENTER).  ESC = cancel.
  2. marker     click the pen marker in "frame", check "mask" shows one blob, press s, then ESC.
  3. live       move the pen; watch the dot, FPS and guidance.  1/2 = box, ESC = quit.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CALIB = os.path.join("data", "page_calibration.json")
MARKER = os.path.join("data", "marker_hsv.json")

STEPS = [
    ("1/3 CALIBRATE PAGE", ["tools/calibrate_page.py"],
     "Whole sheet in view on a darker surface, hands out of the frame, hold still.\n"
     "  It saves by itself after ~1.5 s (green outline + countdown). Manual: click TL, TR, BR, BL, ENTER."),
    ("2/3 TUNE PEN MARKER", ["tools/tune_marker.py"],
     "Hold the pen over the page. CLICK THE MARKER in the 'frame' window.\n"
     "  'mask' window: the marker should be the only white blob. Press s to save, then ESC."),
    ("3/3 LIVE TRACKER", ["-m", "src.tracker"],
     "Move the pen over the page: red circle on the marker, dot + trail on the page view,\n"
     "  ~60 FPS at the top. Press 1/2 to pick a box and follow the guidance text. ESC to quit."),
]


def run(title, args, help_text, passthrough):
    print("\n" + "=" * 78 + f"\n  STEP {title}\n  {help_text}\n" + "=" * 78, flush=True)
    return subprocess.call([sys.executable, "-u", *args, *passthrough])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-calib", action="store_true", help="reuse data/page_calibration.json")
    ap.add_argument("--skip-marker", action="store_true", help="reuse data/marker_hsv.json")
    ap.add_argument("--only-live", action="store_true", help="just the live tracker")
    a, passthrough = ap.parse_known_args()

    skip = set()
    if a.skip_calib or a.only_live:
        skip.add(0)
    if a.skip_marker or a.only_live:
        skip.add(1)

    for i, (title, args, help_text) in enumerate(STEPS):
        if i in skip:
            continue
        rc = run(title, args, help_text, passthrough)
        if i == 0 and not os.path.exists(CALIB):
            print("\nNo page calibration was saved. Fix the view (whole page visible, darker surface) "
                  "and re-run, or click the 4 corners manually.")
            return 1
        if i == 1 and not os.path.exists(MARKER):
            print("\nNo marker colour saved (press s in the tuner). The tracker will use the default "
                  "green range.")
        if rc not in (0, None) and i < 2:
            print(f"\nstep exited with code {rc}; continuing.")
    print("\nDone. Re-run any step alone: tools/calibrate_page.py, tools/tune_marker.py, python -m src.tracker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
