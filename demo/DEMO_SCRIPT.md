# Demo script — 90 seconds (Dev 4 owns this)

**Roles:** *Operator* (laptop, `--debug-keys` hand on the keyboard) · *Student* (blindfolded, holds the pen) · *Narrator* (talks to judges).
**Screen:** HUD full-screen, readable from 2 m. **Say latency honestly:** "sub-10 ms haptic response on the device; ~50 ms end-to-end guidance."

## One-time setup on the demo laptop (Windows)
1. **Python 3.13** (record3d and pyaudio have no 3.14 wheels): `py -3.13 -m venv .venv313`, then
   `.venv313\Scripts\python -m pip install -r requirements.txt`. Run every command below with that Python.
2. **Apple's USB driver** for the iPhone: install the *Apple Devices* app (Microsoft Store) or iTunes. Without it
   Record3D prints `usbmuxd ... error opening socket` and no iPhone is ever found.
3. **iPhone:** Record3D app with *USB Streaming* (Settings), phone unlocked, "Trust this computer" accepted.
4. **Pen:** UNO Q flashed with `firmware/illumin_pen`; it shows up as a COM port (COM4 here). Optional:
   `ARDUINO_PORT=COM4` in `.env` (otherwise it is auto-detected). Close the Arduino IDE Serial Monitor.
5. `.env` from `.env.example`. Without keys the demo still runs: offline Windows voice, pixel answer check.
6. **Which sheet?** `data/questions.json` + `data/layout.json` are the sheet Dev 2 prescanned (5 short questions).
   To use the printed `demo/test_sheet.pdf` instead, add
   `--questions demo/questions.demo.json --layout demo/layout.sheet.json` to the run command
   (or re-run `python -m src.prescan` with that sheet under the camera).

## Pre-flight (10 min before, every time)
1. iPhone streaming in Record3D (red button), pen plugged in, sheet taped down, diffuse light, no glare.
2. `python tools/preflight.py --taps` → no `FAIL`. It checks the packages, sheet files, keys, the pen stream
   (~33 lines/s, a LOCK beep, then asks for one tap and two taps), the iPhone frames + LiDAR depth, the
   page lock (hands out!) and the pen tip, and speaks a test line. Look at `captures/preflight_page.png`.
3. Probe taped on the pen and **not touched again**.
4. Run: `python -m src.main --real all --debug-keys` (add the sheet flags from step 6 above if needed). One window: the HUD.
   * HUD camera panel: "LOOKING FOR THE PAGE" until the sheet locks (hands out, ~2 s still), then the
     straightened page with the answer boxes and a red dot on the pen tip (off the paper: red arrow + side).
   * HUD pen panel: every tap stays on screen ("last: DOUBLE TAP, 3 s ago") with running counts; keyboard
     taps are labelled "(keyboard)". Bottom line: `COM4 connected 33 lines/s`, or why it is not connected.
   * Health chips all `ok` (`IMU no link` = pen unplugged). The camera retries by itself if the phone
     wasn't streaming yet. (A browser view also exists, `--web`, but the HUD is the demo UI.)
5. Press `s` once to confirm screenshots land in `captures/`. Start the screen recording (`--record demo_run.mp4` or OS recorder).

## The 90 seconds
| t | Student / Operator does | Narrator says | HUD shows |
|---|---|---|---|
| 0:00 | Pen held over sheet, still | "A blind student, no scribe, a paper test." | `IDLE`, motion `STILL` |
| 0:08 | (auto) | "Hold still — it reads the question." | `READING`, voice log fills |
| 0:20 | Student moves the hand | "Buzzers pulse left/right and the voice says *Move 2 inches down*. The pen tells us it's **moving**." | `NAVIGATING`, L/R buzzer lights, trail |
| 0:35 | Pen enters the box, stops | "Double tone — locked in." | haptic `LOCK`, "In the answer box" |
| 0:42 | Student writes | "The voice goes **silent** — we never talk over a writing student." | `WRITING_LOCKED`, voice `quiet` |
| 0:52 | Drift past the margin | "Harsh buzz — but only because the **IMU says WRITING**." | `OUT_OF_BOUNDS`, red WARN |
| 1:00 | Correct back inside | "Stops the moment they're back." | back to `WRITING_LOCKED` |
| 1:05 | Hand covers the pen | "The camera loses the pen under the hand — the IMU doesn't." | camera panel: **CAMERA LOST THE PEN / but the IMU still says: WRITING** |
| 1:12 | Student stops writing and rests the pen | "Two seconds without writing — the pen knows the answer is finished." | `AUDITING` |
| 1:18 | (auto) | "Gemini checks the photo of the box." | `LAST AUDIT: INK IN BOX` |
| 1:22 | (auto) | "Answer recorded — ascending tone." | `COMPLETE` haptic, next question |
| 1:30 | — | "**The pen knows what it's doing; the camera knows where it is.**" | IMU panel highlighted |

## Kill switches (Operator, no panic)
| Symptom | Do |
|---|---|
| WRITING never detected / flickers | `3` = force WRITING, `1` still, `2` moving, `4` lifted, `0` = give control back to the sensor |
| Taps not detected | `r` = tap 1 (read / repeat), `t` or `n` = tap 2 (next question). No third tap: an answer ends when writing stops for 2 s (press `1` after `3`) |
| Accelerometer dead | Operator drives everything with `1`–`4` / `t` while the student acts it out |
| Camera loses the pen / bad light | restart with `--real imu,voice,brain,auditor` (fake tracker) and hand-steer with keys |
| Gemini slow/down | nothing — pixel fallback answers after 6 s (HUD note says `pixel fallback`) |
| ElevenLabs down | Dev 3's `say` fallback; if silent too → read the HUD aloud |
| Everything on fire | play `demo/backup_run.mp4` ("this is the recorded run from earlier") |

## Backup video (record on the FIRST clean run, ~H20–23)
`python -m src.main --real all --record demo/backup_run.mp4` → save a second copy on the phone + USB.
No hardware at all? `python -m src.main --fast --record demo/backup_fakes.mp4` renders the scripted world in seconds (label it honestly as a simulation).

## Q&A cheat sheet
* **Why IMU + camera?** Camera can't tell hover from writing, can't see the pen under the hand, can't receive "I'm done". The IMU gives all three.
* **What if the network dies?** Layout is pre-saved; auditor falls back to a pixel check; voice falls back to `say`.
* **Is the buzzer safe if the laptop crashes?** Firmware dead-man: guidance/warn buzzers expire after 1.5 s without a refresh.
* **Why does it never talk while writing?** The brain is gated by the IMU: speech is cancelled the moment WRITING starts and the sim asserts it never speaks in that phase.
