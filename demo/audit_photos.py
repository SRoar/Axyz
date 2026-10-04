"""
Check the auditor against a folder of saved, PAGE-RECTIFIED photos (plan task 4.5: 6/6 agreement).

    python -m demo.make_audit_samples                         # synthetic set -> demo/audit_samples
    python -m demo.audit_photos --dir demo/audit_samples --no-gemini     # pixel fallback only
    python -m demo.audit_photos --dir my_photos                          # with Gemini (needs GEMINI_API_KEY)

Folder format: images + labels.json:
    [{"file": "a.png", "box_id": "box1", "ink_present": true, "ink_outside": false}, ...]
Real photos: run the tracker's snapshot() on the taped sheet, save the PNGs, write the labels by eye.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

from src.audit import GeminiAuditor
from src.contracts import Question, SimClock, load_layout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="demo/audit_samples")
    ap.add_argument("--layout", default=None, help="layout json (default: demo sheet geometry)")
    ap.add_argument("--no-gemini", action="store_true", help="pixel fallback only (offline)")
    a = ap.parse_args()

    if a.layout:
        layout = load_layout(a.layout)
    else:
        from demo import sheet_spec
        layout = sheet_spec.layout()
    with open(os.path.join(a.dir, "labels.json")) as f:
        labels = json.load(f)

    auditor = GeminiAuditor(SimClock(), use_gemini=not a.no_gemini)
    agree = 0
    print(f"{'file':<28}{'want in/out':<14}{'got in/out':<14}{'conf':<6}note")
    for lab in labels:
        img = cv2.imread(os.path.join(a.dir, lab["file"]))
        if img is None:
            print(f"{lab['file']:<28}UNREADABLE")
            continue
        box = layout[lab.get("box_id", "box1")]
        r = auditor.audit_image(img, box, Question("q", "(photo check)", box.id))
        ok = r.ink_present == lab["ink_present"] and r.ink_outside == lab.get("ink_outside", r.ink_outside)
        agree += ok
        print(f"{lab['file']:<28}{str(lab['ink_present'])[0]}/{str(lab.get('ink_outside', '-'))[0]:<12}"
              f"{str(r.ink_present)[0]}/{str(r.ink_outside)[0]:<12}{r.confidence:<6.2f}"
              f"{'OK ' if ok else 'BAD'} {r.note}")
    print(f"\nagreement: {agree}/{len(labels)}")
    return 0 if agree == len(labels) else 1


if __name__ == "__main__":
    sys.exit(main())
