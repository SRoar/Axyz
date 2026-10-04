"""Speech phrase constants for TactileReader / Illumin."""

def question_intro(index: int, text: str) -> str:
    return f"Question {index}. {text}"

def moving_to_question(index: int) -> str:
    return f"Moving to question {index}."

def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")

def test_complete(answered: int, skipped: int) -> str:
    summary = f" You answered {_count(answered, 'question')}"
    summary += f" and skipped {skipped}." if skipped else "."
    return TEST_COMPLETE + summary

TAP_INSTRUCTIONS = "Ready. Tap once to hear the question. Tap twice to go to the next one."
LAST_QUESTION = "That was the last question."
IN_ANSWER_BOX = "In the answer box. Start writing."
ANSWER_RECORDED = "Answer recorded."
ANSWER_RECORDED_OUTSIDE = "Answer recorded. Some writing went outside the box."
NO_INK_FOUND = "I did not find writing in the box. Move to the box and try again."
AUDIT_TIMEOUT = "I could not check your answer. Moving on."
TEST_COMPLETE = "That was the last question. Test complete."
SKIPPING_QUESTION = "Skipping question."
TARGET_CLEARED = "Target cleared."
