"""
Generate the printable DEMO TEST SHEET.  Owner: Dev 4 (plan task 4.6).

    pip install reportlab
    python -m demo.make_test_sheet            # -> demo/test_sheet.pdf  (PRINT AT 100 %, Letter, no 'fit to page')
                                              #    demo/layout.sheet.json     ground-truth boxes (page-normalized)
                                              #    demo/questions.demo.json   the 3 questions (ids q1..q3 -> box1..box3)

Geometry comes from demo/sheet_spec.py (single source of truth).  Dev 2's prescan writes the real
data/layout.json from a photo of this sheet; layout.sheet.json is what it should roughly match.
Dev 3: demo/questions.demo.json is ready to copy into data/questions.json if you want the 3 demo FRQs.
"""
from __future__ import annotations

import argparse
import os

from demo import sheet_spec
from src.contracts import PAGE_H_CM, PAGE_W_CM, save_layout, save_questions

CM = 28.3465  # points per cm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    layout, questions = sheet_spec.layout(), sheet_spec.questions()
    save_layout(os.path.join(a.out, "layout.sheet.json"), layout)
    save_questions(os.path.join(a.out, "questions.demo.json"), questions)

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("wrote layout.sheet.json + questions.demo.json  (install reportlab for the PDF: pip install reportlab)")
        return

    pw, ph = letter
    assert abs(pw / CM - PAGE_W_CM) < 0.05 and abs(ph / CM - PAGE_H_CM) < 0.05
    c = canvas.Canvas(os.path.join(a.out, "test_sheet.pdf"), pagesize=letter)
    c.setTitle("Illumin demo test sheet")
    c.setFillGray(0.1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(2.0 * CM, ph - 1.6 * CM, "Illumin demo test  -  free response")
    c.setFont("Helvetica", 9)
    c.drawRightString(pw - 2.0 * CM, ph - 1.6 * CM, "Name: ______________________")
    for i, ((qbase, y0, y1), text) in enumerate(zip(sheet_spec.ROWS_CM, sheet_spec.QUESTION_TEXT), start=1):
        c.setFont("Helvetica", 11)
        c.drawString(sheet_spec.BOX_X_CM[0] * CM, ph - qbase * CM, f"{i}. {text}")
        c.setLineWidth(4)                                   # THICK dark border (camera + auditor friendly)
        c.setStrokeGray(0.05)
        c.rect(sheet_spec.BOX_X_CM[0] * CM, ph - y1 * CM,
               (sheet_spec.BOX_X_CM[1] - sheet_spec.BOX_X_CM[0]) * CM, (y1 - y0) * CM)
    c.showPage()
    c.save()
    print(f"wrote {a.out}/test_sheet.pdf, layout.sheet.json, questions.demo.json")


if __name__ == "__main__":
    main()
