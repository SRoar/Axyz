# Submission text (edit names / links before pasting)

## Title
**Illumin (TactileReader) — a paper test a blind student can take without a scribe**

## One-liner
A pen probe that *feels* what the hand is doing, an overhead camera that *sees* where it is, and a voice that guides, then stays silent while the student writes.

## Inspiration
Blind and low-vision students taking a paper free-response exam usually need a human scribe: it costs independence and privacy. We wanted the student to write the answer **themselves**, on paper, in the right box.

## What it does
1. Hold the pen still → the question is read aloud (ElevenLabs).
2. Move the hand → left/right buzzers and short spoken cues ("Move 2 inches down") steer the pen to the answer box.
3. Pen settles in the box → a double-tone **LOCK** and "Start writing."
4. While writing the voice is **silent**; drifting past the margin triggers a harsh buzz.
5. Triple-tap the pen → Gemini checks the photo of the box; "Answer recorded."

## How we built it
* **Pen probe (Arduino 101):** accelerometer classifies STILL / MOVING / WRITING / LIFTED on-device and detects 1/2/3 taps; two buzzers play non-blocking haptic patterns with a 1.5 s dead-man switch; light sensor streams for future edge detection.
* **Overhead camera:** marker tracking + 4-corner homography gives the pen tip in page coordinates; occlusion handling holds the position for 300 ms.
* **Design rule:** *the camera knows WHERE the pen is; the accelerometer knows WHAT it is doing; the brain acts on the combination.* The camera cannot tell hovering from writing, cannot see the pen under the hand, and cannot receive "I'm done" — the IMU provides all three.
* **Brain:** a pure-logic state machine (IDLE → READING → NAVIGATING → WRITING → AUDITING) that never speaks while writing and never fires the margin buzzer unless the IMU says WRITING.
* **Auditor:** Gemini 2.5 Flash structured output on a rectified crop of the answer box, with a 6 s timeout and an offline pixel-based fallback.
* **Engineering:** one contract file, fakes for every component driven by a single scripted timeline, and a headless simulation that must pass before every merge — four people integrated hardware, vision, voice and AI in 12 hours.

## Challenges
Separating writing from travelling in noisy accelerometer data; keeping tracking alive when the hand covers the pen; making speech and mic never overlap; getting everything to degrade gracefully when the network or a sensor fails.

## Accomplishments
Every sensor has a kill-switch fallback, the whole flow runs headless in a second, and the HUD shows the key idea live: the camera loses the pen under the hand while the IMU still knows it is writing.

## What's next
Edge detection from the light sensor for sub-10 ms border buzz, multi-page tests, handwriting transcription, other languages.

## Built with
Arduino 101 · LIS3DH-class accelerometer · OpenCV · Python · Gemini 2.5 Flash · ElevenLabs
