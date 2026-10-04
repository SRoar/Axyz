# Demo script — 90 seconds (Dev 4 owns this)

**Roles:** *Operator* (laptop, `--debug-keys` hand on the keyboard) · *Student* (blindfolded, holds the pen) · *Narrator* (talks to judges).
**Screen:** HUD full-screen, readable from 2 m. **Say latency honestly:** "sub-10 ms haptic response on the device; ~50 ms end-to-end guidance."

## Pre-flight (10 min before, every time)
1. `git pull`, `python -m src.sim` → **SIM PASS**; `pytest -q` green.
2. Print `demo/test_sheet.pdf` at **100 %** (Letter, *no* "fit to page"). Tape it down. Diffuse light, no glare on the boxes.
3. Pen probe taped on and **not touched again**. Pen marker colour appears nowhere else on the desk.
4. Camera fixed → re-run page calibration (4 clicks). Check `data/layout.json` matches the sheet (`demo/layout.sheet.json` is the ground truth).
5. `.env` has both API keys; volume up; headphones **out**; mic on.
6. Dry run: `python -m src.main --real all --debug-keys` → HUD shows all chips `ok` (BRA IMU TRA GUI VOI AUD).
7. Press `s` once to confirm screenshots land in `captures/`. Start the screen recording (`--record demo_run.mp4` or OS recorder).

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
