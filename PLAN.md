# TactileReader — Master Plan (hackathon hours 12 → 24)

Blind / low-vision student takes a **paper** free-response test. No scribe.
A **pen probe** (Arduino UNO Q: accelerometer + light sensor + 2 buzzers) and an **overhead camera** guide the pen to each answer box, keep the writing inside it, and a **voice** reads the questions and confirms the answer.

> **Design rule: the camera knows WHERE the pen is. The accelerometer knows WHAT the pen is doing. The brain acts on the combination.**
> The IMU is essential because the camera cannot tell hovering from writing, cannot see the pen under the hand, and has no way to receive a "I'm done" command. The IMU provides all three.

---

## 0. Rules of the road (read once)

1. **Everything talks through `src/contracts.py`.** Types, serial protocol, interfaces, class names. Nobody edits it without telling Dev 4 (contract owner). Changes bump `CONTRACT_VERSION` and get announced in chat.
2. **You own your files. Do not edit anyone else's.** (Ownership list is in each workstream below.)
3. **Develop against fakes.** `src/fakes.py` has a fake of every interface, all driven by one scripted timeline (`FakeWorld`), so they agree with each other.
4. **`python -m src.sim` must PASS before every merge to `main`.** It runs the full flow headless in ~1 second and asserts the behaviour.
5. **Your real class replaces its fake with one flag:** `python -m src.sim --real brain` / `--real imu` / `--real tracker,guidance` etc. Nobody else has to be ready.
6. **Branches:** `dev1-hw`, `dev2-vision`, `dev3-brain`, `dev4-glue`. Merge to `main` only at checkpoints C1, C2, C3 (or any time the sim passes and you only touched your own files).
7. **Real class locations are fixed** (the factory imports them by name):

| Component | Real class (import path) | Constructor | Owner |
|---|---|---|---|
| IMU + haptics link | `src.arduino_link.ArduinoImuLink` | `(clock)` | Dev 1 |
| Pen tracker | `src.tracker.PenTracker` | `(clock)` | Dev 2 |
| Guidance geometry | `src.guidance.GuidanceEngine` | `()` | Dev 2 |
| Voice (TTS + STT) | `src.voice.VoiceEngine` | `(clock)` | Dev 3 |
| State machine | `src.state_machine.StateMachine` | `(questions, layout)` | Dev 3 |
| Answer auditor | `src.audit.GeminiAuditor` | `(clock)` | Dev 4 |

Drop-in: copy `src/contracts.py fakes.py factory.py system.py sim.py` and `data/` into the repo. The existing `src/main.py` is broken (`draw_hud` is undefined) and gets replaced by Dev 4. Existing `camera.py`, `voice.py`, `vision_agent.py`, `config.py` are inputs to Dev 2 / Dev 3 — reuse what works.

---

## 1. Data flow

```
 ┌────────────── PEN PROBE (Arduino UNO Q) ────────────┐        ┌───────── OVERHEAD CAMERA ─────────┐
 │ MMA7660 accel 120Hz ─► classifier ─► STILL/MOVING/  │        │ frame ─► marker detect ─► homography│
 │                        WRITING/LIFTED + taps 1/2/3  │        │          ─► PenState (page coords)  │
 │ light sensor A0 ─► raw stream                       │        └───────────────┬────────────────────┘
 │ buzzers D3(L) D4(R) ◄── haptic pattern engine       │                        │
 └──────────┬───────────────────────────▲──────────────┘                        │
   S/M/T lines│                         │ H,<CMD>                               ▼
            ▼ │                         │                              GuidanceEngine(pen, box)
      ArduinoImuLink  ◄───────────────────┘                                     │ Guidance
            │ ImuEvents + motion                                                │
            └──────────────►  ┌────────────────────────────┐  ◄────────────────┘
                              │   System.step()  (50 Hz)    │
   Voice.poll_command ──────► │   builds Inputs ─► BRAIN    │ ──Speak/Silence──► Voice (ElevenLabs)
   Voice.is_speaking ───────► │   StateMachine.update()     │ ──Haptic─────────► ArduinoImuLink.send
   Auditor.poll ────────────► │   executes Actions          │ ──SetBox─────────► active answer box
                              └─────────────┬──────────────┘ ──Snapshot───────► Tracker.snapshot ─► Auditor.submit
                                            ▼
                                       HUD (cv2) for judges
```

Coordinate space everywhere: **page-normalized** (0,0 top-left of paper → 1,1 bottom-right, y down). Camera pixels never leave the tracker.

## 2. Who knows what

| Question | Answered by | Never by |
|---|---|---|
| Where is the pen tip on the page? | Camera (`PenState`) | IMU (drifts) |
| Is the pen resting / travelling / writing / picked up? | **IMU only** (`MotionState`) | Camera (hand occlusion) |
| Did the student give a command (repeat / skip / done)? | **IMU taps**, voice as backup | Camera |
| Is the pen inside the box? | Guidance (camera + box) | — |
| May the voice speak / may the margin buzzer fire? | **Brain, gated by IMU state** | — |
| Did the answer land in the box? | Auditor (Gemini on a snapshot) | — |

---

## 3. Contract summary (full detail in `contracts.py`)

**Motion states:** `STILL`, `MOVING`, `WRITING`, `LIFTED`. **Taps:** 1 = repeat question, 2 = skip, 3 = "I'm done, audit".
**Haptic commands:** `LOCK` (one-shot), `WARN` (continuous), `COMPLETE` (one-shot), `GUIDE_LEFT`, `GUIDE_RIGHT`, `GUIDE_BOTH`, `OFF`.

**Serial (115200, `\n`-terminated):**

| Direction | Line | Meaning |
|---|---|---|
| Arduino→PC | `READY` | booted (also reply to `PING`) |
| Arduino→PC | `S,<ms>,<ax>,<ay>,<az>,<light>` | raw sample, ~50 Hz, g and raw ADC |
| Arduino→PC | `M,<STATE>` | motion state **changed** |
| Arduino→PC | `T,<1\|2\|3>` | tap gesture |
| PC→Arduino | `H,<CMD>` | play haptic pattern |
| PC→Arduino | `PING` | liveness |

**Haptic patterns (firmware must implement; non-blocking):**

| Cmd | Pattern |
|---|---|
| `LOCK` | two 80 ms pulses, 1000 Hz then 1500 Hz, both buzzers |
| `WARN` | continuous 2500 Hz, alternating buzzers every 100 ms (harsh) |
| `COMPLETE` | ascending 600 / 900 / 1200 Hz, 120 ms each |
| `GUIDE_LEFT` | D3 only, 1000 Hz, 100 ms on / 150 ms off |
| `GUIDE_RIGHT` | D4 only, same |
| `GUIDE_BOTH` | both, 800 Hz, 100 ms on / 400 ms off |
| `OFF` | silence |

`GUIDE_*` and `WARN` **auto-expire after 1500 ms without a refresh** (the brain refreshes every 0.5 s). If the PC dies, the buzzers stop.

---

## 4. Brain transition table (Dev 3 owns; `ReferenceStateMachine` implements the happy path)

| Phase | Event / condition | Actions | Next |
|---|---|---|---|
| IDLE | STILL ≥ 0.8 s (and ≥ 0.8 s in phase) **or** tap 1 | `SetBox(box)`, `Speak("Question N. …")` | READING |
| IDLE / READING / NAVIGATING | tap 2 or "skip" | `Speak("Skipping question.")` | next question's IDLE (or COMPLETE) |
| READING | tap 1 or "repeat" | re-`Speak` (interrupt) | READING |
| READING | speech finished | — | NAVIGATING |
| NAVIGATING | motion = MOVING | `Haptic(guidance.cmd)` (refresh 0.5 s); spoken cue every 3.5 s if not speaking | NAVIGATING |
| NAVIGATING | motion = STILL/LIFTED | `Haptic(OFF)` | NAVIGATING |
| NAVIGATING | STILL ≥ 0.3 s **and** `in_box` | `Haptic(LOCK)`, `Speak("In the answer box. Start writing.")` (once) | NAVIGATING (locked) |
| NAVIGATING | motion = **WRITING** | `Silence()` | WRITING |
| WRITING | motion = WRITING **and** write_status = OUTSIDE | `Haptic(WARN)` | WRITING |
| WRITING | otherwise | `Haptic(OFF)` | WRITING |
| WRITING | not WRITING for ≥ 2.0 s **or** tap 3 / "done" | `Haptic(OFF)`, `Snapshot(q.id)` | AUDITING |
| AUDITING | result.ink_present | `Speak("Answer recorded.")`, `Haptic(COMPLETE)` | IDLE (next q) / COMPLETE |
| AUDITING | result.ink_present = False | `Speak("I did not find writing…")` | NAVIGATING |
| AUDITING | timeout 8 s | `Speak("I could not check…")`, `Haptic(COMPLETE)` | next |

The brain **never speaks during WRITING** and **never fires WARN unless the IMU says WRITING.** The sim asserts both.

---

## 5. What EVERY component does

### 5.1 Firmware (Arduino UNO Q) — Dev 1
- Initialise `Wire.h`, **MMA7660 at I²C `0x4C`** (found by the bus scan; the module is the Grove 3-Axis Digital Accelerometer, not a LIS3DH), up to 120 Hz, ±1.5 g. The output is only 6-bit (about 21.33 counts per g, ~47 mg steps), so it is coarse: check early that WRITING jitter and taps are still visible (see the hardware notes below). Serial 115200.
- **Non-blocking loop, no `delay()`.** Sample at 100 Hz, emit `S,…` every 2nd sample.
- **Motion classifier** over a ~0.3 s rolling window of |a|: STILL (tiny variance), MOVING (large, low-frequency), WRITING (small-to-medium, sustained high-frequency jitter, 5–20 Hz), LIFTED (gravity vector shifted from baseline / z spike). Hysteresis: a state must hold ~100 ms to be emitted; emit `M,<STATE>` on change only.
- **Tap detector:** spike in |a| deviation > ~0.4 g lasting < 60 ms; group spikes within 500 ms; emit `T,n` after 500 ms of quiet.
- **Haptic engine:** the table in §3, non-blocking via `tone()` timing, with the 1500 ms dead-man for repeating patterns.
- Light sensor A0 value goes out in `S` (HUD + future edge detection). *Stretch:* local sub-10 ms edge buzz when the light value jumps while WRITING (box border crossing), only if everything else is solid.

### 5.2 `ArduinoImuLink` (PC side of the pen) — Dev 1
- Auto-detect the serial port (fallback to `config.ARDUINO_PORT`), reader thread, `parse_device_line()` from the contract, thread-safe event queue.
- `motion` property = latest `M,` state (starts STILL). `poll()` returns events since last call. `send()` is non-blocking (write queue), reconnects on unplug. `recent_samples(n)` for the HUD waveform. 20 Hz `PING` watchdog optional.

### 5.3 `PenTracker` — Dev 2
- Capture thread (reuse `camera.py`: Record3D if present, else webcam). Always keep the **latest** frame; `read()` never blocks.
- Detect the **pen marker** (bright cap/sticker via HSV mask, tuned in a one-shot `tools/tune_marker.py`), compute the tip centroid, smooth lightly.
- **Page calibration:** user clicks the 4 page corners once → homography → page-normalized coords; saved to `data/page_calibration.json`, reloaded on start.
- **Occlusion handling:** if the marker is lost, hold/extrapolate for ≤ 300 ms with decaying `confidence`, then return `pen=None`.
- `snapshot()`: page-rectified, sharp frame (warpPerspective) for the auditor.

### 5.4 `GuidanceEngine` — Dev 2
- Pure function `(PenState, Box) → Guidance`. Distance to the box edge in cm (`PAGE_W_CM/H_CM`), dominant-axis spoken phrase ("Move 2 inches Down."), `GUIDE_LEFT/RIGHT` when horizontally off, `GUIDE_BOTH` when aligned horizontally, `in_box`, `write_status` with a small hysteresis margin so jitter at the edge doesn't flap WARN.

### 5.5 Prescan / layout — Dev 2
- `python -m src.prescan`: capture the rectified blank test page, ask Gemini for the answer box of each question (the test sheet has numbered boxes), write `data/layout.json` (`Dict[box_id, Box]`). `--manual` mode: click two corners per box. **The layout file is committed so the demo never depends on the network.**

### 5.6 `StateMachine` — Dev 3
- The table in §4 plus the error paths: pen lost mid-navigation (keep last cue, then OFF), retry limit on empty audit (2 retries, then accept), tap debounce, ignore taps during READING-speech start, WRITING seen while IDLE/READING (queue it, don't crash), phase timeouts, COMPLETE summary.
- Pure logic: no I/O, no sleeps, time comes from `Inputs.t`. That is what makes it unit-testable.

### 5.7 `VoiceEngine` + phrases — Dev 3
- ElevenLabs TTS with `eleven_flash_v2_5` (streaming, play via mpv/pyaudio; macOS `afplay` as a fallback; **macOS `say` as an offline fallback** if the API fails).
- `speak(text, interrupt)`, `silence()`, `is_speaking()` (**True from call until all queued audio is done**, no gaps), `stop()`.
- STT in the background (reuse existing recognizer): `poll_command()` returns lowercase phrases. **Mic muted while speaking and while the brain is in WRITING** (avoid hearing itself / the student).
- `src/speech.py`: short, consistent phrase library (every spoken line in one file).

### 5.8 `GeminiAuditor` — Dev 4
- `submit()` returns immediately; a worker thread crops the snapshot to the box (+10 % margin), asks Gemini 2.5 Flash (structured output) "is there handwriting inside the box / just outside it?", `poll()` returns the `AuditResult` once. 6 s timeout → **pixel fallback** (dark-pixel fraction inside the inset box vs a threshold).

### 5.9 `System` + `main.py` + HUD — Dev 4
- `System.step()` (already written) is the only place components meet. `main.py` = CLI (`--real …`), `RealClock`, start/stop, 50 Hz loop, `--debug-keys` (keys 1–4 inject STILL/MOVING/WRITING/LIFTED, `t` injects tap 3, so a dead sensor can't kill the demo).
- **HUD (cv2):** camera view + answer boxes + pen dot + 2 s trail; big phase banner (READING / NAVIGATING / WRITING_LOCKED / OUT_OF_BOUNDS / AUDITING); current question text; **IMU panel** (motion state, accel waveform, tap flashes); haptic log; last audit result.

### 5.10 Physical setup — shared (assigned below)
Probe rigidly fixed to the pen (**do not change mounting after Dev 1 records data**); overhead camera fixed, page position taped, diffuse light; test sheet with 2–3 numbered thick dark answer boxes and the pen marker colour that appears nowhere else.

---

## 6. The four workstreams (≈ 10 focused hours each)

Times are hackathon hours; if you start later than H12, shrink the first block.

### DEV 1 — Pen Hardware: firmware + serial link  ·  *the critical path*
**Owns:** `firmware/illumin_pen/illumin_pen.ino`, `src/arduino_link.py`, `tools/imu_record.py`, `tests/test_link.py`.
**Mission:** the probe reports motion state and taps, and plays haptic patterns, reliably.

| # | Task | Est |
|---|---|---|
| 1.1 | Sketch skeleton: `Wire.h`, MMA7660 @0x4C, ~100 Hz, non-blocking loop, `S` stream @50 Hz, `READY` (a working accelerometer-only sketch is already in `accel_stream/`) | 1.0 h |
| 1.2 | **Mount the probe on the pen**, then `tools/imu_record.py`: record labeled CSV for STILL / MOVING / WRITING / LIFTED with the real pen on real paper (≥ 30 s each) | 1.5 h |
| 1.3 | Motion classifier + hysteresis, thresholds tuned from the recordings; `M,` events | 2.0 h |
| 1.4 | Tap detector (1/2/3) → `T,n` | 1.5 h |
| 1.5 | Haptic pattern engine (§3) + dead-man timeout | 1.5 h |
| 1.6 | `ArduinoImuLink` (thread, parser, reconnect, `recent_samples`, `send`) | 1.5 h |
| 1.7 | Tests + hardening: replay recorded lines through `parse_device_line`; unplug/replug; 10-min soak | 1.0 h |

**Test without anyone else:** serial monitor — type `H,LOCK` and hear it; hold still / move / write and watch `M,` lines; tap 3× and see `T,3`. In Python: `python -m src.sim --real imu` (fake everything else; haptics you hear are the brain's real commands from the scripted world).
**Done when:** classifier gets ≥ 90 % of a 60 s scripted routine (still → move → write → lift) right, taps are detected ≥ 9/10, patterns are audibly distinct, link survives replug, sim passes with `--real imu`.
**Hard deadline: working classifier by H15.** Fallback if WRITING can't be separated from MOVING: emit only STILL/MOVING/LIFTED and let tap-3 mean "done" (the brain already handles this).

### DEV 2 — Vision & Geometry: tracker, calibration, guidance, prescan
**Owns:** `src/tracker.py`, `src/guidance.py`, `src/prescan.py`, `tools/calibrate_page.py`, `tools/tune_marker.py`, `data/page_calibration.json`, `data/layout.json`, `tests/test_guidance.py`. (Reuses/edits `camera.py`, `vision_agent.py`.)
**Mission:** turn camera frames into a clean page-space pen position, guidance, and the answer-box layout.

| # | Task | Est |
|---|---|---|
| 2.1 | **Camera rig**: fix camera, tape page position, lighting; capture thread from `camera.py`, latest-frame `read()` | 1.5 h |
| 2.2 | Marker detection (HSV mask, `tune_marker.py`), tip centroid, smoothing; replaces skin-tone hand tracking | 2.0 h |
| 2.3 | Page calibration (4 clicks → homography → page coords), persisted; `snapshot()` rectified image | 2.0 h |
| 2.4 | Occlusion hold/extrapolate (≤ 300 ms, decaying confidence) + 30 FPS check | 1.0 h |
| 2.5 | `GuidanceEngine`: cm distances, phrasing, GUIDE_* choice, in-box hysteresis; unit tests | 1.5 h |
| 2.6 | `prescan.py`: Gemini box finder (+ `--manual` fallback) → `data/layout.json`; commit it | 2.0 h |

**Test without anyone else:** `python -m src.tracker` live view prints/draws `PenState`; `pytest tests/test_guidance.py`; `python -m src.sim --real tracker,guidance` (fake IMU/voice/brain drive the flow; you watch whether the pen position and cues make sense).
**Done when:** pen tracked at ≥ 25 FPS within ~0.5 cm of truth over the page, survives a hand covering the pen for 300 ms, guidance phrases match what a human would say, `layout.json` committed for the real test sheet, sim passes with `--real tracker,guidance`.

### DEV 3 — Brain & Voice: state machine, voice, phrases
**Owns:** `src/state_machine.py`, `src/voice.py`, `src/speech.py`, `data/questions.json`, `tests/test_state_machine.py`.
**Mission:** the logic that makes the whole thing behave, and the voice that talks to the student.

| # | Task | Est |
|---|---|---|
| 3.1 | `StateMachine` from `ReferenceStateMachine` (copy it, then harden) — §4 table | 2.0 h |
| 3.2 | Error paths: pen lost, retry limit, tap debounce, out-of-order events, timeouts, skip/repeat/done | 1.5 h |
| 3.3 | `tests/test_state_machine.py`: scripted `Inputs` sequences for every row of §4 + each error path | 2.0 h |
| 3.4 | `VoiceEngine` TTS: `eleven_flash_v2_5` streaming, queue, `is_speaking` (no gaps), `silence()`, `interrupt`, `say` fallback | 2.5 h |
| 3.5 | STT: `poll_command()`, keywords (repeat / skip / next / done), mic mute while speaking and during WRITING | 1.0 h |
| 3.6 | `speech.py` phrase library (short, clear, one place) + write 3 demo FRQs in `questions.json` | 1.0 h |

**Test without anyone else:** `pytest tests/test_state_machine.py` (pure logic, no hardware); `python -m src.sim --real brain` (your brain vs the fake world — the full flow, instantly); `python -m src.sim --real voice` to hear real audio driven by the scripted flow.
**Done when:** every transition-table row has a passing test, `--real brain` passes the sim, voice latency from `speak()` to first audio is < 1 s, `is_speaking()` has no gaps between queued phrases, mic never transcribes the TTS.

### DEV 4 — Integration, HUD, Auditor, Demo  ·  *contract owner*
**Owns:** `src/main.py`, `src/hud.py`, `src/audit.py`, `src/system.py`, `src/contracts.py` (arbiter), `src/factory.py`, `src/sim.py`, `demo/`.
**Mission:** make the pieces plug together, make it visible to judges, check the answers, and own the demo.

| # | Task | Est |
|---|---|---|
| 4.1 | Drop in contract files, get the team to run `python -m src.sim` (H12). Replace the broken `main.py`: CLI `--real`, RealClock, 50 Hz loop, clean shutdown | 1.5 h |
| 4.2 | `--debug-keys` injector (keys inject motion states and tap 3) so any dead sensor can be bypassed on stage | 0.5 h |
| 4.3 | HUD: camera view + boxes + pen + trail, phase banner, question text | 2.0 h |
| 4.4 | HUD IMU panel: motion state, accel waveform, tap flashes, haptic log, audit result | 1.5 h |
| 4.5 | `GeminiAuditor`: worker thread, crop + structured Gemini call, pixel fallback, tests on saved photos | 2.5 h |
| 4.6 | Demo ops: printed test sheet (numbered thick boxes, 3 FRQs), demo script, **backup video**, README + submission text | 1.5 h |
| 4.7 | Integration referee: run checkpoints C1–C3, triage who fixes what, own `git merge` to main | 1.0 h |

**Test without anyone else:** the HUD runs on all fakes (`python -m src.main` with no `--real`); auditor tested on 6 saved photos (ink in box / outside / empty).
**Done when:** the full real system runs from one command, HUD is legible from 2 m, auditor agrees with a human on 6/6 photos, a screen-recorded successful run exists.

---

## 7. Schedule and checkpoints

| Hours | Everyone | Gate |
|---|---|---|
| **12.0 – 12.5** | Pull contract files, `python -m src.sim` passes on every laptop, create branches | **C0:** all four see `SIM PASS` |
| 12.5 – 16 | Build your piece against fakes (see your table). Commit small, merge only if sim passes | — |
| **16.0** | Each dev demos their standalone test | **C1:** each real class passes `--real <mine>` |
| 16 – 18 | **Pair integration:** (Dev 1 + Dev 3) = `--real imu,brain,voice` — this is the IMU-driven core; (Dev 2 + Dev 4) = `--real tracker,guidance,auditor` | **C2:** both pairs pass |
| 18 – 20 | Full real: `--real imu,tracker,guidance,voice,brain,auditor`. Fix, tune thresholds | **C3:** one full question, end to end, on the real sheet |
| **20.0** | **FEATURE FREEZE.** Only bug fixes, no new features | — |
| 20 – 23 | Rehearse with a blindfolded teammate ×3. Record the backup video on the first clean run | — |
| 23 – 24 | Submission, slides, charge laptops, pack | — |

## 8. Kill switches (decide NOW, use on stage without panic)

| If this breaks | Do this | Cost |
|---|---|---|
| WRITING vs MOVING unreliable | firmware emits STILL/MOVING/LIFTED only; **tap 3 = done** | none visible |
| Taps unreliable | voice "done" (already mapped) | slightly less pure |
| Accelerometer dead | `--debug-keys` injection from a teammate's keyboard | cheating, but the demo runs |
| Pen marker lost / bad light | `--real` without tracker, use the fake world + hand-steer | no live guidance |
| Gemini down / slow | pixel-fraction fallback in the auditor; layout.json is pre-saved | none |
| ElevenLabs down | macOS `say` fallback in `voice.py` | worse voice |
| Arduino won't flash on stage | pre-recorded backup video | — |

## 9. Demo script (90 seconds)

1. Blindfolded teammate holds the pen over the sheet. **STILL** → "Question 1. …" reads aloud.
2. Hand moves: buzzers pulse left/right, voice says "Move 2 inches Down." (**MOVING** → guidance on.)
3. Pen enters the box, stops: double-tone **LOCK**, "In the answer box. Start writing."
4. Writing: voice goes **silent**. Pen drifts past the margin → harsh **WARN**; correct → stops. (HUD shows `WRITING` from the IMU while the camera loses the pen under the hand — *this is the pitch*.)
5. Triple tap → audit → "Answer recorded." ascending **COMPLETE** tone.
6. Show the HUD IMU panel: "the pen knows what it's doing; the camera knows where it is."

**Say latency honestly:** "sub-10 ms haptic response on the device; ~50 ms end-to-end guidance" (the camera is 30 FPS ≈ 33 ms per frame).

## 10. Risks, ranked

1. **Classifier quality** (Dev 1) — whole demo leans on it. Mitigation: record data in hour 1, hard deadline H15, kill switches above.
2. **Marker tracking under the hand** (Dev 2) — mitigation: bright marker, occlusion hold, IMU doesn't depend on it.
3. **Integration surprises** — mitigation: contract + fakes + sim gate; pair integration at H16.
4. **Network at the venue** — mitigation: committed `layout.json`, fallbacks for Gemini/ElevenLabs.
5. **Mounting changes after calibration** — don't touch the probe after recordings start.

## 11. Workload balance

| Dev | Area | Hours | Cross-dependencies while building |
|---|---|---|---|
| 1 | Firmware + serial link | 10.0 | none (serial monitor + sim) |
| 2 | Tracker + calibration + guidance + prescan | 10.0 | none (live view + unit tests + sim) |
| 3 | State machine + voice + phrases | 10.0 | none (pure logic tests + sim) |
| 4 | Integration + HUD + auditor + demo | 10.5 | none (all-fakes HUD + saved photos) |

First time anyone depends on anyone: checkpoint C2 (H16), and by then each side has already passed its own tests.

## 12. Hardware notes (Dev 1)

Changes from the original plan, found while bringing up the hardware (Oct 2026):

| Item | Original plan | Actual |
|---|---|---|
| Pen board | Arduino 101 | **Arduino UNO Q** (the 101 would not accept uploads: "Timed out waiting for Arduino 101") |
| Accelerometer | LIS3DH, I²C `0x18` | **MMA7660, I²C `0x4C`** (Grove 3-Axis Digital Accelerometer), 6-bit, about 21.33 counts/g, ±1.5 g, up to 120 Hz |
| Upload path | USB DFU | UNO Q uploads over `adb`; board package `arduino:zephyr:unoq`, serial port shows up as COM4 on Windows |

Working code: `accel_stream/accel_stream.ino` streams `A,<ms>,<x_g>,<y_g>,<z_g>` at 50 Hz and was confirmed on the real UNO Q. Task 1.1 turns it into the contract's `READY` / `S,…` format with the A0 light value added.

Known issues and open questions:
- **Uploads to the UNO Q are intermittent**: sometimes `adb.exe: device offline`, then working again with no clear change. Workaround: unplug, wait about 60 s for the board to boot, try again; close the Serial Monitor before using the port from scripts.
- **Coarse accelerometer.** About 47 mg per step with a ±1.5 g range. WRITING jitter and 0.4 g tap spikes may be harder to see or may clip. Record data early (task 1.2) before trusting any threshold. The kill switches in §8 still apply.
- **Not yet verified on the UNO Q:** that `tone()` can drive both buzzers (D3, D4) at the same time, and the range of the A0 light-sensor reading (it may differ from the 10-bit values the old test sketch assumed).
