# Illumin / TactileReader

A blind or low-vision student takes a **paper** free-response test with no scribe.
A **pen probe** (Arduino 101: accelerometer + light sensor + 2 buzzers) and an **overhead camera** guide the pen to each answer box, keep the writing inside it, and a **voice** reads the questions and confirms the answer.

> **The camera knows WHERE the pen is. The accelerometer knows WHAT the pen is doing. The brain acts on the combination.**

Full design, contract, transition table and per-dev workstreams: **[PLAN.md](PLAN.md)**.

## Quick start
```bash
pip install -r requirements.txt            # + `pip install pytest` for the tests
cp .env.example .env                       # GEMINI_API_KEY, ELEVENLABS_API_KEY, ARDUINO_PORT, CAMERA_INDEX

python -m src.sim                          # headless full-flow check, ~1 s  -> must print SIM PASS
pytest -q                                  # unit + integration tests
python -m src.main                         # HUD on ALL FAKES (scripted world)
```

## Running real components
Every component is the real class or a fake, chosen with one flag (names: `imu tracker guidance voice brain auditor`, or `all`):
```bash
python -m src.sim  --real brain                          # your brain vs the fake world (instant, asserted)
python -m src.main --real imu,brain,voice                # IMU-driven core
python -m src.main --real tracker,guidance,auditor       # camera pair
python -m src.main --real all --debug-keys               # full system, keyboard rescue enabled
python -m src.main --real all --record demo/backup_run.mp4   # record the HUD (backup video)
```
`--debug-keys`: `1` STILL · `2` MOVING · `3` WRITING · `4` LIFTED · `r` tap 1 (read / repeat) · `t` or `n` tap 2 (next question) · `0` release override. `q`/ESC quits, `s` saves a screenshot.
Other flags: `--headless` (no window), `--fast` (fakes only, simulated clock), `--max-seconds N`, `--exit-on-complete`.

## Browser UI
`python -m src.main --web` serves a live view at http://localhost:8765 (camera with labeled boxes, voice waveform, pen probe,
buzzers, gesture cues, audit). `ui/index.html` also opens by itself with demo data. See `ui/README.md`.

## Layout
| Path | What | Owner |
|---|---|---|
| `src/contracts.py` | types, serial protocol, interfaces (single source of truth) | Dev 4 |
| `src/fakes.py`, `src/factory.py`, `src/sim.py`, `src/system.py` | fakes, real/fake wiring, headless sim, the glue loop | Dev 4 |
| `src/main.py`, `src/hud.py`, `src/web.py`, `ui/` | CLI + 50 Hz loop, judges' HUD, browser UI + its bridge | Dev 4 |
| `src/audit.py` | Gemini answer auditor with pixel fallback | Dev 4 |
| `src/arduino_link.py`, `firmware/` | pen probe link + firmware | Dev 1 |
| `src/tracker.py`, `src/guidance.py`, `src/prescan.py` | vision + geometry | Dev 2 |
| `src/state_machine.py`, `src/voice.py`, `src/speech.py` | brain + voice | Dev 3 |
| `demo/` | test sheet, demo script, integration checklist, submission text | Dev 4 |

## Demo kit
```bash
pip install -r demo/requirements-demo.txt
python -m demo.make_test_sheet                         # -> demo/test_sheet.pdf (print at 100 %), layout.sheet.json, questions.demo.json
python -m demo.make_audit_samples                      # synthetic "saved photos" for the auditor
python -m demo.audit_photos --dir demo/audit_samples   # auditor vs labelled photos (add --no-gemini for offline)
```
See `demo/DEMO_SCRIPT.md` (90 s script + kill switches), `demo/INTEGRATION.md` (checkpoints, triage), `demo/SUBMISSION.md`.

## Contract rules (short)
Everything talks through `src/contracts.py`. You own your files; don't edit others'. Develop against fakes. `python -m src.sim` must pass before every merge.
