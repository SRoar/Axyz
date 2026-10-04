"""
GeminiAuditor -- "did the answer land in the box?"   Owner: Dev 4.

    src.audit.GeminiAuditor(clock)          (constructor signature fixed by the contract)

submit() returns IMMEDIATELY.  A worker thread:
    1. crops the page-rectified snapshot to the answer box (+10 % margin),
    2. asks Gemini 2.5 Flash (structured JSON output) whether there is handwriting
       INSIDE the box and/or just OUTSIDE it,
    3. waits at most `timeout_s` (6 s) -- on timeout / error / no API key / no network it
       falls back to a PIXEL check (dark-pixel fraction inside the inset box),
    4. queues an AuditResult.  poll() hands each result out exactly once.

If there is no snapshot at all (camera has no frame) NO result is produced: the brain's own
8 s audit timeout then says "I could not check your answer" and moves on -- the designed path.

The pixel check is also useful on its own:   pixel_audit(image, box) -> PixelStats
Everything here works without the network and without google-genai installed.

Test it on saved photos (no hardware):   python -m demo.audit_photos --help
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from src.contracts import PAGE_H_CM, PAGE_W_CM, AuditResult, Box, Clock, Question

GEMINI_MODEL = "gemini-3.8-flash"  # gemini-2.5-flash is retired for new API keys (404 NOT_FOUND)
GEMINI_TIMEOUT_S = 6.0

CROP_MARGIN = 0.10          # Gemini crop = box grown by 10 % of its size on every side
BORDER_EXCLUDE_CM = 0.45    # ignore this band at the box border (printed border + calibration slop)
RING_START_CM = 0.60        # "just outside the box" ring starts this far from the box edge ...
RING_END_CM = 1.30          # ... and ends this far out (sheet rule: nothing printed within 1.4 cm of a box)
INK_MIN_FRAC = 0.002        # >= 0.2 % dark pixels in the inset box  => handwriting present
INK_OUTSIDE_MIN_FRAC = 0.010  # >= 1.0 % dark pixels in the ring       => writing outside (advisory)
INK_REL_DARKNESS = 0.78     # pixel counts as ink if < 78 % of the local paper brightness
MIN_SPECK_PX = 4            # connected dark blobs smaller than this are sensor noise


# =========================================================================== pixel check
@dataclass
class PixelStats:
    inside_frac: float      # dark-pixel fraction inside the inset box
    outside_frac: float     # dark-pixel fraction in the ring just outside the box
    ink_present: bool
    ink_outside: bool


def _to_gray(image: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.GaussianBlur(g, (3, 3), 0)


def _px_rect(box: Box, w: int, h: int, grow_x_cm: float = 0.0, grow_y_cm: Optional[float] = None
             ) -> Tuple[int, int, int, int]:
    """Box (page-normalized) -> pixel rect, grown (+) or shrunk (-) by a physical distance in cm."""
    gy = grow_x_cm if grow_y_cm is None else grow_y_cm
    gx_n, gy_n = grow_x_cm / PAGE_W_CM, gy / PAGE_H_CM
    x0 = int(round((box.xmin - gx_n) * w))
    x1 = int(round((box.xmax + gx_n) * w))
    y0 = int(round((box.ymin - gy_n) * h))
    y1 = int(round((box.ymax + gy_n) * h))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """
    Ink = pixels clearly darker than the LOCAL paper level.  The paper level is estimated with a
    morphological closing (removes thin dark strokes, keeps paper), so lighting gradients and soft
    shadows divide out and faint pencil still shows up.
    """
    k = int(np.clip(min(gray.shape) // 8, 9, 41)) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (k, k), 0).astype(np.float32)
    g = gray.astype(np.float32)
    return ((g < bg * INK_REL_DARKNESS) & (bg - g > 25.0)).astype(np.uint8)


def _clean(mask: np.ndarray, line_span: float) -> np.ndarray:
    """Drop speckle noise and any blob that spans >= line_span of the region (printed border lines)."""
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        cw, ch, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area < MIN_SPECK_PX:
            continue
        if cw >= line_span * w or ch >= line_span * h:
            continue
        keep[i] = True
    return keep[labels].astype(np.uint8)


def pixel_audit(image: np.ndarray, box: Box) -> PixelStats:
    """Fallback ink detector on a PAGE-RECTIFIED image (box coordinates are page-normalized)."""
    h, w = image.shape[:2]
    gray = _to_gray(image)

    # inside: the box shrunk by the border band (never more than 15 % per side)
    bw_cm, bh_cm = (box.xmax - box.xmin) * PAGE_W_CM, (box.ymax - box.ymin) * PAGE_H_CM
    ix = min(BORDER_EXCLUDE_CM, 0.15 * bw_cm)
    iy = min(BORDER_EXCLUDE_CM, 0.15 * bh_cm)
    x0, y0, x1, y1 = _px_rect(box, w, h, -ix, -iy)
    inside = 0.0
    if x1 - x0 > 4 and y1 - y0 > 4:
        roi = _clean(_ink_mask(gray[y0:y1, x0:x1]), line_span=0.92)
        inside = float(roi.sum()) / roi.size

    # outside: ring between RING_START_CM and RING_END_CM from the box edge
    ox0, oy0, ox1, oy1 = _px_rect(box, w, h, RING_END_CM)
    ix0, iy0, ix1, iy1 = _px_rect(box, w, h, RING_START_CM)
    outside = 0.0
    if ox1 - ox0 > 4 and oy1 - oy0 > 4:
        region = gray[oy0:oy1, ox0:ox1]
        m = _clean(_ink_mask(region), line_span=0.5)
        hole = np.zeros_like(m)
        hole[iy0 - oy0:iy1 - oy0, ix0 - ox0:ix1 - ox0] = 1
        m[hole == 1] = 0
        ring_area = m.size - int(hole.sum())
        outside = float(m.sum()) / max(1, ring_area)

    return PixelStats(inside, outside, inside >= INK_MIN_FRAC, outside >= INK_OUTSIDE_MIN_FRAC)


# =========================================================================== gemini
def crop_for_gemini(image: np.ndarray, box: Box) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop to the box +10 % margin.  Returns (crop, box location inside the crop in 0..1000)."""
    h, w = image.shape[:2]
    mx, my = (box.xmax - box.xmin) * CROP_MARGIN, (box.ymax - box.ymin) * CROP_MARGIN
    cx0, cy0 = max(0.0, box.xmin - mx), max(0.0, box.ymin - my)
    cx1, cy1 = min(1.0, box.xmax + mx), min(1.0, box.ymax + my)
    crop = image[int(cy0 * h):max(int(cy1 * h), int(cy0 * h) + 2), int(cx0 * w):max(int(cx1 * w), int(cx0 * w) + 2)]
    cw, ch = max(1e-6, cx1 - cx0), max(1e-6, cy1 - cy0)
    loc = (int(1000 * (box.xmin - cx0) / cw), int(1000 * (box.ymin - cy0) / ch),
           int(1000 * (box.xmax - cx0) / cw), int(1000 * (box.ymax - cy0) / ch))
    return crop, loc


def build_prompt(loc: Tuple[int, int, int, int], question_text: str) -> str:
    x0, y0, x1, y1 = loc
    return (
        "You are checking a photo of a paper test sheet written by a blind student who cannot see "
        "where the pen is. The image is a crop around ONE answer box. The answer box is the rectangle "
        "with a thick dark printed border; inside this crop it spans x from "
        f"{x0} to {x1} and y from {y0} to {y1} (both normalized 0-1000, origin top-left).\n"
        f"The question for this box was: \"{question_text}\".\n"
        "HANDWRITING means pen or pencil strokes made by the student. The printed box border, printed "
        "numbers/labels, printed question text, shadows, creases and paper texture are NOT handwriting.\n"
        "Answer strictly:\n"
        "- ink_inside: true if ANY handwriting is inside the box.\n"
        "- ink_outside: true if ANY handwriting is visible in the crop but OUTSIDE the box border.\n"
        "- confidence: 0.0-1.0 for your answer.\n"
        "- note: at most 12 words describing what you see.\n"
        "Be conservative: faint smudges and shadows are not handwriting."
    )


def _verdict_schema():
    """Pydantic schema for Gemini structured output (None if pydantic is missing)."""
    try:
        from pydantic import BaseModel, Field
    except ImportError:  # pragma: no cover
        return None

    class AuditVerdict(BaseModel):
        ink_inside: bool = Field(description="handwriting present inside the answer box")
        ink_outside: bool = Field(description="handwriting present outside the box border")
        confidence: float = Field(description="0.0 to 1.0")
        note: str = Field(description="at most 12 words")

    return AuditVerdict


def parse_verdict(text: str) -> Dict[str, Any]:
    """Parse Gemini's JSON (tolerates ```json fences).  Raises ValueError on garbage."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    d = json.loads(t)
    if not isinstance(d, dict) or "ink_inside" not in d:
        raise ValueError(f"unexpected verdict: {text[:120]!r}")
    return {
        "ink_inside": bool(d["ink_inside"]),
        "ink_outside": bool(d.get("ink_outside", False)),
        "confidence": float(min(1.0, max(0.0, d.get("confidence", 0.8)))),
        "note": str(d.get("note", ""))[:120],
    }


# =========================================================================== the auditor
@dataclass
class _Job:
    image: np.ndarray
    box: Box
    question: Question


class GeminiAuditor:
    """Non-blocking answer auditor (see module docstring)."""

    def __init__(self, clock: Clock, api_key: Optional[str] = None, model: str = GEMINI_MODEL,
                 timeout_s: float = GEMINI_TIMEOUT_S, use_gemini: bool = True) -> None:
        self.clock = clock
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key if api_key is not None else self._env_key()
        self.use_gemini = use_gemini
        self._client: Any = None
        self._client_failed = False
        self._jobs: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._results: Deque[AuditResult] = deque()
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._closed = False
        self.history: Deque[AuditResult] = deque(maxlen=20)   # for the HUD / debugging

    # ------------------------------------------------------------------ public contract
    def submit(self, snapshot: Any, box: Box, question: Question) -> None:
        if snapshot is None or not hasattr(snapshot, "shape") or snapshot.size == 0:
            print(f"[auditor] no snapshot for {question.id}; brain will time out and move on",
                  file=sys.stderr)
            return
        self._ensure_worker()
        self._jobs.put(_Job(np.array(snapshot, copy=True), box, question))

    def poll(self) -> Optional[AuditResult]:
        with self._lock:
            return self._results.popleft() if self._results else None

    # ------------------------------------------------------------------ synchronous core
    def audit_image(self, image: np.ndarray, box: Box, question: Question) -> AuditResult:
        """Blocking audit used by the worker (and by demo/audit_photos.py and the tests)."""
        t0 = time.monotonic()
        px = pixel_audit(image, box)
        why = self._gemini_unavailable_reason()
        if why is None:
            holder: Dict[str, Any] = {}
            th = threading.Thread(target=self._gemini_worker, args=(image, box, question, holder),
                                  daemon=True, name="gemini-call")
            th.start()
            th.join(self.timeout_s)
            if th.is_alive():
                why = f"gemini timeout >{self.timeout_s:.0f}s"
            elif "error" in holder:
                why = f"gemini error: {holder['error']}"
            else:
                v = holder["verdict"]
                dt = time.monotonic() - t0
                return AuditResult(
                    question.id, v["ink_inside"], v["ink_outside"], v["confidence"],
                    f"gemini {dt:.1f}s: {v['note']} | px={px.inside_frac:.2%}/{px.outside_frac:.2%}")
        return AuditResult(
            question.id, px.ink_present, px.ink_outside, 0.5,
            f"pixel fallback ({why}) inside={px.inside_frac:.2%} outside={px.outside_frac:.2%}")

    def close(self) -> None:
        self._closed = True
        self._jobs.put(None)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _env_key() -> str:
        try:
            from src.config import config
            return config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        except Exception:  # noqa: BLE001
            return os.getenv("GEMINI_API_KEY", "")

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._closed = False
                self._worker = threading.Thread(target=self._run, daemon=True, name="auditor")
                self._worker.start()

    def _run(self) -> None:
        while not self._closed:
            job = self._jobs.get()
            if job is None:
                return
            try:
                result = self.audit_image(job.image, job.box, job.question)
            except Exception as e:  # noqa: BLE001 -- the brain must always get an answer
                print(f"[auditor] audit crashed: {e}", file=sys.stderr)
                result = AuditResult(job.question.id, True, False, 0.0, f"audit crashed: {e}")
            print(f"[auditor] {job.question.id}: ink_in_box={result.ink_present} "
                  f"outside={result.ink_outside} ({result.note})", file=sys.stderr)
            with self._lock:
                self._results.append(result)
                self.history.append(result)

    def _gemini_unavailable_reason(self) -> Optional[str]:
        if not self.use_gemini:
            return "gemini disabled"
        if not self.api_key and not self._overridden():
            return "no GEMINI_API_KEY"
        if self._client_failed:
            return "gemini client unavailable"
        return None

    def _overridden(self) -> bool:
        """True when a subclass / test replaced _ask_gemini (so no key is needed)."""
        return type(self)._ask_gemini is not GeminiAuditor._ask_gemini

    def _gemini_worker(self, image: np.ndarray, box: Box, question: Question, out: Dict[str, Any]) -> None:
        try:
            crop, loc = crop_for_gemini(image, box)
            long_side = max(crop.shape[:2])
            if long_side > 1600:
                s = 1600.0 / long_side
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("jpeg encode failed")
            text = self._ask_gemini(jpg.tobytes(), build_prompt(loc, question.text))
            out["verdict"] = parse_verdict(text)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"[:160]

    def _ask_gemini(self, jpeg: bytes, prompt: str) -> str:
        """One structured Gemini call -> JSON text.  (Tests override this method.)"""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client_failed = True
                raise
        from google.genai import types
        cfg: Dict[str, Any] = dict(response_mime_type="application/json", temperature=0.0)
        schema = _verdict_schema()
        if schema is not None:
            cfg["response_schema"] = schema
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)   # fast: no thinking
        except Exception:  # noqa: BLE001
            pass
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
            config=types.GenerateContentConfig(**cfg),
        )
        return resp.text
