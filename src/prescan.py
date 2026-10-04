"""
Prescan: find the answer box of every question on the blank test sheet -> data/layout.json.  Owner: Dev 2.

    python -m src.prescan                         # camera -> rectified page -> Gemini -> preview -> save
    python -m src.prescan --manual                # click two opposite corners per box instead of Gemini
    python -m src.prescan --image page.png        # use a saved page-rectified image instead of the camera
    python -m src.prescan --yes                   # skip the preview confirmation

Needs data/page_calibration.json (python tools/calibrate_page.py) for the camera path: the image
sent to Gemini is page-rectified, so its 0-1000 coordinates ARE page-normalized coordinates x 1000.
Box ids come from data/questions.json (question N -> its box_id), else "box<N>".
COMMIT data/layout.json so the demo never depends on the network.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.contracts import Box, RealClock, load_layout, load_questions, save_layout

PRESCAN_IMAGE = "data/prescan_page.png"
GEMINI_MODEL = "gemini-2.5-flash"
PROMPT = (
    "This image is a photo of a paper test sheet, cropped and rectified so the image edges are the "
    "page edges. The sheet has numbered free-response questions, each followed by a large rectangular "
    "answer box drawn with thick dark lines where the student writes. For EVERY answer box return the "
    "question number it belongs to and the box's INNER rectangle (inside the border lines) as "
    "ymin, xmin, ymax, xmax normalized to 0-1000 over the full image. Do not return the question text "
    "area, only the empty writing box. Order by question number."
)


# --------------------------------------------------------------------------- box ids
def box_ids_for(questions_path: str, n: int) -> List[str]:
    try:
        qs = load_questions(questions_path)
    except (FileNotFoundError, ValueError):
        qs = []
    ids = [q.box_id for q in qs]
    return ids[:n] + [f"box{i + 1}" for i in range(len(ids), n)]


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def make_box(box_id: str, x0: float, y0: float, x1: float, y1: float) -> Box:
    return Box(box_id, _clamp01(min(x0, x1)), _clamp01(min(y0, y1)), _clamp01(max(x0, x1)), _clamp01(max(y0, y1)))


# --------------------------------------------------------------------------- Gemini
def gemini_boxes(image: np.ndarray, questions_path: str) -> Dict[str, Box]:
    from pydantic import BaseModel, Field
    from google import genai
    from google.genai import types
    from src.config import config

    class AnswerBox(BaseModel):
        question_number: int = Field(description="Number printed next to the question (1, 2, 3, ...)")
        ymin: int = Field(description="Top edge, 0-1000")
        xmin: int = Field(description="Left edge, 0-1000")
        ymax: int = Field(description="Bottom edge, 0-1000")
        xmax: int = Field(description="Right edge, 0-1000")

    class AnswerBoxes(BaseModel):
        boxes: List[AnswerBox]

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set (.env); use --manual")
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode the page image")
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=AnswerBoxes, temperature=0.0),
    )
    found = sorted(AnswerBoxes.model_validate_json(resp.text).boxes, key=lambda b: b.question_number)
    print(f"[prescan] Gemini found {len(found)} boxes in {time.perf_counter() - t0:.1f}s")
    ids = box_ids_for(questions_path, max([b.question_number for b in found], default=0))
    out: Dict[str, Box] = {}
    for b in found:
        if 1 <= b.question_number <= len(ids):
            bid = ids[b.question_number - 1]
            out[bid] = make_box(bid, b.xmin / 1000, b.ymin / 1000, b.xmax / 1000, b.ymax / 1000)
    return out


# --------------------------------------------------------------------------- manual
def manual_boxes(image: np.ndarray, questions_path: str, n: Optional[int]) -> Dict[str, Box]:
    """Click two opposite corners per box. u = undo, ENTER = finish, ESC = cancel."""
    if n is None:
        try:
            n = len(load_questions(questions_path))
        except (FileNotFoundError, ValueError):
            n = 2
    ids = box_ids_for(questions_path, n)
    h, w = image.shape[:2]
    clicks: List[Tuple[float, float]] = []
    win = "prescan --manual"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2 * len(ids):
            clicks.append((x / w, y / h))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = image.copy()
        for i in range(0, len(clicks) - 1, 2):
            (a, b), (c, d) = clicks[i], clicks[i + 1]
            cv2.rectangle(view, (int(a * w), int(b * h)), (int(c * w), int(d * h)), (255, 120, 0), 2)
            cv2.putText(view, ids[i // 2], (int(min(a, c) * w) + 4, int(min(b, d) * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
        for (a, b) in clicks[len(clicks) // 2 * 2:]:
            cv2.circle(view, (int(a * w), int(b * h)), 5, (0, 0, 255), -1)
        k = len(clicks) // 2
        msg = (f"{ids[k]}: click {'first' if len(clicks) % 2 == 0 else 'opposite'} corner"
               if k < len(ids) else "ENTER = save")
        cv2.putText(view, msg + "  (u=undo ESC=cancel)", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyWindow(win)
            return {}
        if key == ord("u") and clicks:
            clicks.pop()
        if key in (13, 10) and len(clicks) >= 2:
            break
    cv2.destroyWindow(win)
    out = {}
    for i in range(0, len(clicks) - 1, 2):
        bid = ids[i // 2]
        (a, b), (c, d) = clicks[i], clicks[i + 1]
        out[bid] = make_box(bid, a, b, c, d)
    return out


# --------------------------------------------------------------------------- capture
def capture_page(camera_args) -> np.ndarray:
    from src.page_calibration import CALIB_PATH, PageCalibration
    from src.tracker import PenTracker
    from src.webcam import config_from_args

    if PageCalibration.load(CALIB_PATH) is None:
        raise SystemExit(f"no {CALIB_PATH}: run `python tools/calibrate_page.py` first")
    tracker = PenTracker(RealClock(), config_from_args(camera_args), hud_frame=False)
    tracker.start()
    try:
        time.sleep(1.5)   # let exposure settle and fill the sharpness buffer
        snap = tracker.snapshot()
    finally:
        tracker.stop()
    if snap is None:
        raise SystemExit("camera delivered no frames")
    return snap


def preview(image: np.ndarray, boxes: Dict[str, Box]) -> bool:
    view = image.copy()
    h, w = view.shape[:2]
    for b in boxes.values():
        cv2.rectangle(view, (int(b.xmin * w), int(b.ymin * h)), (int(b.xmax * w), int(b.ymax * h)), (0, 180, 0), 3)
        cv2.putText(view, b.id, (int(b.xmin * w) + 6, int(b.ymin * h) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
    cv2.putText(view, "ENTER = save   ESC = discard", (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    scale = min(1.0, 900.0 / h)
    cv2.imshow("prescan result", cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            ok = True
            break
        if key == 27:
            ok = False
            break
    cv2.destroyWindow("prescan result")
    return ok


def main() -> int:
    from src.webcam import add_camera_args

    ap = argparse.ArgumentParser(description="Find answer boxes on the blank test sheet -> data/layout.json")
    add_camera_args(ap)
    ap.add_argument("--manual", action="store_true", help="click the boxes instead of asking Gemini")
    ap.add_argument("--image", default=None, help="page-rectified image file instead of the camera")
    ap.add_argument("--boxes", type=int, default=None, help="number of boxes for --manual (default: #questions)")
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--out", default="data/layout.json")
    ap.add_argument("--yes", action="store_true", help="save without the preview window")
    a = ap.parse_args()

    if a.image:
        image = cv2.imread(a.image)
        if image is None:
            raise SystemExit(f"could not read {a.image}")
    else:
        image = capture_page(a)
        os.makedirs(os.path.dirname(PRESCAN_IMAGE), exist_ok=True)
        cv2.imwrite(PRESCAN_IMAGE, image)
        print(f"[prescan] saved the rectified page to {PRESCAN_IMAGE}")

    if a.manual:
        boxes = manual_boxes(image, a.questions, a.boxes)
    else:
        try:
            boxes = gemini_boxes(image, a.questions)
        except Exception as e:
            print(f"[prescan] Gemini failed ({e}); falling back to --manual")
            boxes = manual_boxes(image, a.questions, a.boxes)
    if not boxes:
        print("[prescan] no boxes; nothing saved")
        return 1
    for b in boxes.values():
        print(f"  {b.id}: x {b.xmin:.3f}-{b.xmax:.3f}  y {b.ymin:.3f}-{b.ymax:.3f}")
    try:
        missing = {q.box_id for q in load_questions(a.questions)} - set(boxes)
        if missing:
            print(f"[prescan] WARNING: questions reference boxes that were not found: {sorted(missing)}")
    except (FileNotFoundError, ValueError):
        pass
    if not a.yes and not preview(image, boxes):
        print("[prescan] discarded")
        return 1
    if os.path.exists(a.out):
        try:
            old = load_layout(a.out)
            print(f"[prescan] replacing {a.out} (had {sorted(old)})")
        except Exception:
            pass
    save_layout(a.out, boxes)
    print(f"[prescan] wrote {a.out}. Commit it so the demo works offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
