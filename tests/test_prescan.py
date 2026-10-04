import json

from src.prescan import box_ids_for, clean_question_text, make_box, make_questions


def test_box_ids_follow_questions_then_default(tmp_path):
    q = tmp_path / "questions.json"
    q.write_text(json.dumps([{"id": "q1", "text": "a", "box_id": "boxA"},
                             {"id": "q2", "text": "b", "box_id": "boxB"}]))
    assert box_ids_for(str(q), 3) == ["boxA", "boxB", "box3"]
    assert box_ids_for(str(tmp_path / "missing.json"), 2) == ["box1", "box2"]


def test_clean_question_text_drops_number_and_extra_spaces():
    assert clean_question_text("  3.  What is frozen\nliquid water called? ") == "What is frozen liquid water called?"
    assert clean_question_text("Q2) What planet do humans live on?") == "What planet do humans live on?"
    assert clean_question_text("Question 4: What gas?") == "What gas?"
    assert clean_question_text("What organ pumps blood in humans?") == "What organ pumps blood in humans?"


def test_make_questions_in_number_order_with_box_ids():
    qs = make_questions({2: "2. What planet?", 1: "What color is the sky?", 9: "out of range", 3: "  "},
                        ["box1", "box2", "box3"])
    assert [(q.id, q.text, q.box_id) for q in qs] == [
        ("q1", "What color is the sky?", "box1"), ("q2", "What planet?", "box2")]


def test_make_box_orders_and_clamps_corners():
    b = make_box("box1", 0.9, 0.5, 0.1, -0.2)
    assert (b.xmin, b.ymin, b.xmax, b.ymax) == (0.1, 0.0, 0.9, 0.5)
