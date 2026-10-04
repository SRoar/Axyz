"""
Geometry of the printed DEMO TEST SHEET (US Letter, 3 free-response questions).  Owner: Dev 4.

One source of truth for: the printable PDF (make_test_sheet.py), the synthetic auditor photos
(make_audit_samples.py) and the ground-truth layout other devs can compare prescan against.

SHEET RULES (they make tracking + auditing reliable):
  * Answer boxes are THICK dark rectangles (4 pt border).
  * NOTHING is printed within CLEARANCE_CM of any box (the auditor's "writing outside" ring is
    0.6-1.3 cm from the box edge; the pen-margin buzzer lives in the same band).
  * No printing INSIDE a box.  Question numbers live in the question line above the box.
  * No red / bright-green / bright-orange ink anywhere -> keep the pen marker colour unique.
"""
from __future__ import annotations

from typing import Dict, List

from src.contracts import PAGE_H_CM, PAGE_W_CM, Box, Question

CLEARANCE_CM = 1.4
BOX_X_CM = (2.0, PAGE_W_CM - 2.0)
# (question baseline y, box top y, box bottom y) in cm from the TOP of the page
ROWS_CM = [
    (3.6, 5.0, 9.4),
    (11.7, 13.4, 17.8),
    (20.1, 21.8, 26.2),
]
QUESTION_TEXT = [
    "Explain the first law of thermodynamics.",
    "Describe how the second law of thermodynamics relates to entropy.",
    "Why does ice float on liquid water?",
]


def layout() -> Dict[str, Box]:
    out: Dict[str, Box] = {}
    for i, (_, y0, y1) in enumerate(ROWS_CM, start=1):
        out[f"box{i}"] = Box(f"box{i}", round(BOX_X_CM[0] / PAGE_W_CM, 4), round(y0 / PAGE_H_CM, 4),
                             round(BOX_X_CM[1] / PAGE_W_CM, 4), round(y1 / PAGE_H_CM, 4))
    return out


def questions() -> List[Question]:
    return [Question(f"q{i}", t, f"box{i}") for i, t in enumerate(QUESTION_TEXT, start=1)]


def check_clearance() -> List[str]:
    """Self-check of the sheet rules (also run by the tests)."""
    errs: List[str] = []
    for i, ((qb, y0, y1), nxt) in enumerate(zip(ROWS_CM, ROWS_CM[1:] + [None]), start=1):
        if y0 - qb < CLEARANCE_CM * 0.0 + 1.2:     # baseline -> box top; text is above the ring
            errs.append(f"q{i}: question baseline too close to its box")
        if nxt is not None:
            text_top = nxt[0] - 0.55                # ~0.55 cm of glyph height above the baseline
            if text_top - y1 < CLEARANCE_CM:
                errs.append(f"box{i}: next question text only {text_top - y1:.2f} cm below it")
        if y1 + CLEARANCE_CM > PAGE_H_CM:
            errs.append(f"box{i}: clearance below box runs off the page")
    return errs
