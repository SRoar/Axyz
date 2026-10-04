"""
Synthetic "saved photos" for the auditor (Dev 4, task 4.5): page-rectified test sheets with
handwriting-like strokes, so the auditor can be tested with NO hardware and NO network.

    python -m demo.make_audit_samples                 # writes demo/audit_samples/*.png + labels.json
    python -m demo.audit_photos --dir demo/audit_samples

Replace/extend with REAL photos for the final check: put page-rectified images in a folder with a
labels.json like the generated one (see demo/audit_photos.py) -- the plan's bar is 6/6 agreement.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from demo import sheet_spec
from src.contracts import PAGE_H_CM, PAGE_W_CM, Box

W, H = 850, 1100            # letter aspect (21.59 x 27.94 cm)
INK = (110, 40, 20)         # blue-black ink, BGR
PENCIL = (125, 125, 125)


def _scribble(img: np.ndarray, x0: int, x1: int, y: int, rng: np.random.Generator,
              color=INK, thick: int = 2, amp: int = 9) -> None:
    """A line of fake cursive: connected loops with word gaps."""
    x = x0
    while x < x1 - 40:
        wlen = int(rng.integers(60, 160))
        xs = np.arange(x, min(x + wlen, x1 - 10), 2.0)
        phase = rng.uniform(0, 6.28)
        ys = y + amp * np.sin(xs / rng.uniform(3.0, 5.0) + phase) + rng.normal(0, 1.2, xs.size)
        pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)
        x += wlen + int(rng.integers(18, 40))


def blank_sheet(layout: Dict[str, Box], rng: np.random.Generator) -> np.ndarray:
    """Looks like the printed demo sheet: thick boxes + question line above each (see sheet_spec)."""
    img = np.full((H, W, 3), 238, np.uint8)
    for i, ((qbase, _, _), text) in enumerate(zip(sheet_spec.ROWS_CM, sheet_spec.QUESTION_TEXT), start=1):
        b = layout.get(f"box{i}")
        if b is None:                                    # layouts with fewer boxes (e.g. the 2-box sample)
            continue
        x0, x1, y0, y1 = int(b.xmin * W), int(b.xmax * W), int(b.ymin * H), int(b.ymax * H)
        cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 20), 6)                      # thick printed border
        cv2.putText(img, f"{i}. {text}", (x0, int(qbase / PAGE_H_CM * H)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return img


def _finish(img: np.ndarray, rng: np.random.Generator, shadow: bool = False) -> np.ndarray:
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f *= (0.93 + 0.07 * (xx / W))[..., None]                                          # lighting gradient
    if shadow:
        f *= (1.0 - 0.45 * np.clip((xx - 0.45 * W) / (0.3 * W), 0, 1))[..., None]    # big soft shadow
    f += rng.normal(0, 3.0, f.shape)                                                  # sensor noise
    return cv2.GaussianBlur(np.clip(f, 0, 255).astype(np.uint8), (3, 3), 0)


def make_samples(layout: Dict[str, Box] | None = None, seed: int = 3
                 ) -> List[Tuple[str, np.ndarray, Dict[str, object]]]:
    """-> [(name, image, label)] label = {box_id, ink_present, ink_outside}."""
    layout = layout or sheet_spec.layout()
    rng = np.random.default_rng(seed)
    b = layout["box1"]
    x0, x1, y0, y1 = int(b.xmin * W), int(b.xmax * W), int(b.ymin * H), int(b.ymax * H)
    mid = (y0 + y1) // 2
    out_y = y1 + int(1.0 / PAGE_H_CM * H)                    # ~1.0 cm below the box border
    S: List[Tuple[str, np.ndarray, Dict[str, object]]] = []

    def new() -> np.ndarray:
        return blank_sheet(layout, rng)

    img = new()
    for k in range(3):
        _scribble(img, x0 + 30, x1 - 30, mid - 40 + k * 38, rng)
    S.append(("1_ink_in_box", _finish(img, rng), dict(box_id="box1", ink_present=True, ink_outside=False)))

    img = new()
    _scribble(img, x0 + 30, x1 - 100, out_y, rng)
    S.append(("2_ink_outside_only", _finish(img, rng), dict(box_id="box1", ink_present=False, ink_outside=True)))

    S.append(("3_empty", _finish(new(), rng), dict(box_id="box1", ink_present=False, ink_outside=False)))

    img = new()
    for k in range(2):
        _scribble(img, x0 + 30, x1 - 30, mid - 20 + k * 38, rng)
    _scribble(img, x0 + 30, x1 - 100, out_y, rng)
    S.append(("4_ink_in_and_out", _finish(img, rng), dict(box_id="box1", ink_present=True, ink_outside=True)))

    img = new()
    for k in range(2):
        _scribble(img, x0 + 30, x1 - 30, mid - 20 + k * 38, rng, color=PENCIL, thick=2)
    S.append(("5_faint_pencil_in_box", _finish(img, rng), dict(box_id="box1", ink_present=True, ink_outside=False)))

    S.append(("6_shadow_only_no_ink", _finish(new(), rng, shadow=True),
              dict(box_id="box1", ink_present=False, ink_outside=False)))
    return S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/audit_samples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    labels = []
    for name, img, lab in make_samples():
        cv2.imwrite(os.path.join(a.out, name + ".png"), img)
        labels.append(dict(file=name + ".png", **lab))
    with open(os.path.join(a.out, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)
    print(f"wrote {len(labels)} samples + labels.json to {a.out}")


if __name__ == "__main__":
    main()
