"""
Prescan: find the answer box of every question on the blank test sheet -> data/layout.json.  Owner: Dev 2.

    python -m src.prescan                         # camera -> rectified page -> Gemini -> preview -> save
    python -m src.prescan --manual                # click two opposite corners per box instead of Gemini
    python -m src.prescan --image page.png        # use a saved page-rectified image instead of the camera
    python -m src.prescan --yes                   # skip the preview confirmation

Camera path: a live window shows the view until the sheet is found (all 4 corners visible, hands
out, still); the photo is then page-rectified, so Gemini's 0-1000 coordinates ARE page-normalized
coordinates x 1000. The new page position is saved to data/page_calibration.json for the tracker.
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
MIN_PAGE_BRIGHTNESS = 40.0
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_TRIES = 5
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
    for attempt in range(GEMINI_TRIES):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   response_schema=AnswerBoxes, temperature=0.0),
            )
            break
        except Exception as e:      # 503 "high demand" spikes are short: back off and retry
            if attempt == GEMINI_TRIES - 1 or "503" not in str(e) and "UNAVAILABLE" not in str(e):
                raise
            wait = 3.0 * 2 ** attempt
            print(f"[prescan] Gemini busy, retrying in {wait:.0f}s ({attempt + 1}/{GEMINI_TRIES - 1})")
            time.sleep(wait)
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

    tracker = PenTracker(RealClock(), config_from_args(camera_args), hud_frame=False)
    tracker.start()
    snap = None
    win = "prescan: frame the WHOLE sheet (all 4 corners), hands out   SPACE = take it now   ESC = cancel"
    try:
        # live view until the page locks FRESH in this session (never the old saved position:
        # the photo for Gemini must be cropped to where the sheet is now)
        locked_since = None
        while True:
            raw = tracker.raw_frame()
            if raw is None:
                if cv2.waitKey(30) & 0xFF == 27:
                    break
                continue
            searching, progress, guess = tracker.calibration_status()
            if not searching and tracker.page_locked_t is None:
                tracker.recalibrate()          # startup search timed out: keep looking
            view = raw.copy()
            h, w = view.shape[:2]
            fs = max(0.5, w / 1400.0)
            locked = tracker.page_locked_t is not None
            if locked:
                cv2.polylines(view, [np.int32(tracker.calibration.page_polygon_px())], True, (0, 255, 0), 3)
                msg, col = "PAGE FOUND - taking the photo", (0, 200, 0)
            elif float(raw.mean()) < MIN_PAGE_BRIGHTNESS:
                msg, col = "TOO DARK - lens covered? lights on?", (0, 0, 255)
            elif guess is not None:
                cv2.polylines(view, [np.int32(guess)], True, (0, 165, 255), 3)
                cv2.rectangle(view, (10, h - 30), (10 + int(progress * (w - 20)), h - 18), (0, 255, 0), -1)
                msg, col = "page seen - hold still, or SPACE if the outline is right", (0, 165, 255)
            else:
                msg, col = "NO PAGE - show the whole sheet, all 4 corners, desk around it", (0, 0, 255)
            cv2.putText(view, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2)
            scale = min(1.0, 900.0 / h)
            cv2.imshow(win, cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == ord(" ") and not locked and guess is not None:
                calib = PageCalibration(np.float32(guess), (w, h))
                calib.save(CALIB_PATH)
                tracker.reload()
                snap = calib.rectify(raw, PenTracker.SNAPSHOT_PX_PER_CM, cv2.INTER_CUBIC)
                print(f"[prescan] took the photo with the outlined page (saved to {CALIB_PATH})")
                break
            if locked:
                locked_since = locked_since or time.monotonic()
                if time.monotonic() - locked_since >= 0.5:   # sharpness buffer holds the final view
                    snap = tracker.snapshot()
                    break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
    if snap is None:
        raise SystemExit("cancelled: no photo of the page")
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
