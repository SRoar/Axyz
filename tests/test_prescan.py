import json

from src.prescan import box_ids_for, make_box


def test_box_ids_follow_questions_then_default(tmp_path):
    q = tmp_path / "questions.json"
    q.write_text(json.dumps([{"id": "q1", "text": "a", "box_id": "boxA"},
                             {"id": "q2", "text": "b", "box_id": "boxB"}]))
    assert box_ids_for(str(q), 3) == ["boxA", "boxB", "box3"]
    assert box_ids_for(str(tmp_path / "missing.json"), 2) == ["box1", "box2"]


def test_make_box_orders_and_clamps_corners():
    b = make_box("box1", 0.9, 0.5, 0.1, -0.2)
    assert (b.xmin, b.ymin, b.xmax, b.ymax) == (0.1, 0.0, 0.9, 0.5)
