"""Auditor tests (Dev 4): synthetic saved photos, fake Gemini, timeout/error fallback, async contract."""
import json
import time

import numpy as np
import pytest

from demo.make_audit_samples import make_samples
from src.audit import GeminiAuditor, parse_verdict, pixel_audit, crop_for_gemini
from demo import sheet_spec
from src.contracts import Auditor, Question, SimClock

LAYOUT = sheet_spec.layout()
Q = Question("q1", "Explain the first law of thermodynamics.", "box1")
SAMPLES = make_samples()


@pytest.mark.parametrize("name,img,label", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_pixel_fallback_agrees_with_labels(name, img, label):
    px = pixel_audit(img, LAYOUT[label["box_id"]])
    assert px.ink_present == label["ink_present"], (name, px)
    assert px.ink_outside == label["ink_outside"], (name, px)


class FakeGemini(GeminiAuditor):
    """Overrides the network call; verdict text / delay / failure are scripted."""
    reply = json.dumps({"ink_inside": True, "ink_outside": False, "confidence": 0.9, "note": "two lines"})
    delay = 0.0
    boom = False

    def _ask_gemini(self, jpeg, prompt):
        assert isinstance(jpeg, bytes) and jpeg[:2] == b"\xff\xd8" and "answer box" in prompt
        time.sleep(self.delay)
        if self.boom:
            raise RuntimeError("503 unavailable")
        return self.reply


def _wait(a, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = a.poll()
        if r:
            return r
        time.sleep(0.01)
    raise AssertionError("no result")


def test_contract_and_nonblocking():
    a = FakeGemini(SimClock())
    assert isinstance(a, Auditor)
    a.delay = 0.3
    t0 = time.time()
    a.submit(SAMPLES[0][1], LAYOUT["box1"], Q)
    assert time.time() - t0 < 0.1, "submit() must return immediately"
    assert a.poll() is None
    r = _wait(a)
    assert r.question_id == "q1" and r.ink_present and not r.ink_outside and r.confidence == 0.9
    assert "gemini" in r.note
    assert a.poll() is None, "a result is delivered exactly once"
    a.close()


def test_gemini_verdict_is_used_over_pixels():
    a = FakeGemini(SimClock())
    a.reply = json.dumps({"ink_inside": False, "ink_outside": True, "confidence": 0.7, "note": "outside"})
    r = a.audit_image(SAMPLES[0][1], LAYOUT["box1"], Q)      # photo HAS ink inside; Gemini says no
    assert (r.ink_present, r.ink_outside) == (False, True)


def test_timeout_falls_back_to_pixels():
    a = FakeGemini(SimClock(), timeout_s=0.2)
    a.delay = 2.0
    t0 = time.time()
    r = a.audit_image(SAMPLES[0][1], LAYOUT["box1"], Q)
    assert time.time() - t0 < 1.0
    assert r.ink_present and "timeout" in r.note and "pixel" in r.note


@pytest.mark.parametrize("reply", ["not json at all", "{}", "[1,2]"])
def test_garbage_reply_falls_back(reply):
    a = FakeGemini(SimClock())
    a.reply = reply
    r = a.audit_image(SAMPLES[2][1], LAYOUT["box1"], Q)      # empty sheet
    assert not r.ink_present and "pixel" in r.note


def test_gemini_error_falls_back():
    a = FakeGemini(SimClock())
    a.boom = True
    r = a.audit_image(SAMPLES[0][1], LAYOUT["box1"], Q)
    assert r.ink_present and "gemini error" in r.note


def test_no_api_key_uses_pixels_without_network():
    a = GeminiAuditor(SimClock(), api_key="")
    r = a.audit_image(SAMPLES[2][1], LAYOUT["box1"], Q)
    assert not r.ink_present and "no GEMINI_API_KEY" in r.note


def test_no_snapshot_produces_no_result():
    a = GeminiAuditor(SimClock(), api_key="")
    a.submit(None, LAYOUT["box1"], Q)
    time.sleep(0.1)
    assert a.poll() is None          # brain's 8 s timeout handles it


def test_results_keep_order_and_ids():
    a = GeminiAuditor(SimClock(), api_key="")
    q2 = Question("q2", "x", "box2")
    a.submit(SAMPLES[0][1], LAYOUT["box1"], Q)
    a.submit(SAMPLES[2][1], LAYOUT["box2"], q2)
    r1, r2 = _wait(a), _wait(a)
    assert (r1.question_id, r2.question_id) == ("q1", "q2")
    a.close()


def test_parse_verdict_handles_fences_and_clamps():
    v = parse_verdict('```json\n{"ink_inside": true, "confidence": 7, "note": "x"}\n```')
    assert v["ink_inside"] is True and v["ink_outside"] is False and v["confidence"] == 1.0


def test_crop_has_ten_percent_margin_and_box_location():
    img = SAMPLES[0][1]
    crop, loc = crop_for_gemini(img, LAYOUT["box1"])
    b = LAYOUT["box1"]
    exp_w = (b.xmax - b.xmin) * 1.2 * img.shape[1]
    assert abs(crop.shape[1] - exp_w) < 3
    assert 60 < loc[0] < 110 and 890 < loc[2] < 940      # box sits ~8 % in from each crop edge


def test_demo_sheet_follows_its_own_clearance_rules():
    assert sheet_spec.check_clearance() == []
