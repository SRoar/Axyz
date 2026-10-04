"""
Tap-driven question reader.

    python -m src.tap_reader                 # pen on the auto-detected Arduino port
    python -m src.tap_reader --port /dev/cu.usbmodem1101
    python -m src.tap_reader --keyboard      # no pen: type 1 or 2 then Enter

One tap  = read the current question out loud.
Two taps = move on to the next question.
"""
from __future__ import annotations

import argparse
import threading
import time

from src import speech
from src.contracts import Tap, load_questions
from src.voice import VoiceEngine


class TapReader:
    def __init__(self, questions, voice: VoiceEngine) -> None:
        self.questions = questions
        self.voice = voice
        self.index = 0

    def on_tap(self, tap: Tap) -> None:
        if tap == Tap.SINGLE:
            q = self.questions[self.index]
            self.voice.speak(speech.question_intro(self.index + 1, q.text), interrupt=True)
        elif tap == Tap.DOUBLE:
            if self.index + 1 >= len(self.questions):
                self.voice.speak(speech.LAST_QUESTION, interrupt=True)
                return
            self.index += 1
            self.voice.speak(speech.moving_to_question(self.index + 1), interrupt=True)


def keyboard_taps(handler) -> None:
    while True:
        try:
            line = input().strip()
        except EOFError:
            return
        if line in ("1", "2"):
            handler(Tap(int(line)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--port", default=None)
    ap.add_argument("--keyboard", action="store_true", help="simulate taps from the keyboard")
    a = ap.parse_args()

    voice = VoiceEngine()
    reader = TapReader(load_questions(a.questions), voice)
    imu = None

    if a.keyboard:
        threading.Thread(target=keyboard_taps, args=(reader.on_tap,), daemon=True).start()
        print("Keyboard mode: type 1 (single tap) or 2 (double tap), then Enter. Ctrl+C to quit.")
    else:
        from src.arduino_link import ArduinoImuLink
        imu = ArduinoImuLink(port=a.port)
        imu.start()
        print("Tap the pen once to hear the question, twice for the next one. Ctrl+C to quit.")

    voice.speak(speech.TAP_INSTRUCTIONS)
    try:
        while True:
            if imu:
                for e in imu.poll():
                    if e.tap != Tap.NONE:
                        print(f"[tap] {e.tap.name}")
                        reader.on_tap(e.tap)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        if imu:
            imu.stop()
        voice.stop()


if __name__ == "__main__":
    main()
